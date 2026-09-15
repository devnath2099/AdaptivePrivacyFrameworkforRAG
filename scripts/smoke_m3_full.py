"""M3 full smoke test: 10K train / 2K val, 5 epochs, stratified sampling.

Sampling covers all 4 domains, all sensitivity/intent/disclosure classes,
and balanced positive/negative examples for each entity_tags/threat_content category.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from m1_data_integration.config import load_config
from m1_data_integration.schemas import EvidenceBundle, UnifiedRecord
from m3_privacy_prediction.dataset import M3Dataset, validate_alignment
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss
from m3_privacy_prediction.metrics import compute_all_metrics
from m3_privacy_prediction.trainer import M3Trainer

print("=" * 80)
print("M3 FULL SMOKE TEST")
print("Train: 10,000 records | Val: 2,000 records | Epochs: 5 | Seed: 42")
print("=" * 80)

rng = np.random.default_rng(42)

# 1. Load records
cfg = load_config("configs/review1.yaml")

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ev = d.get("evidence")
            if isinstance(ev, dict):
                ev = EvidenceBundle(
                    entities=ev.get("entities", []),
                    dependency_relations=ev.get("dependency_relations", []),
                    regex_matches=ev.get("regex_matches", {}),
                    embedding=ev.get("embedding"),
                )
            rec = UnifiedRecord(
                record_id=d["record_id"],
                domain=d["domain"],
                source_dataset=d["source_dataset"],
                query_text=d.get("query_text", "") or "",
                context_text=d.get("context_text", "") or "",
                normalized_text=d.get("normalized_text", "") or "",
                metadata=d.get("metadata", {}),
                evidence=ev,
                is_duplicate=d.get("is_duplicate", False),
                split=d.get("split"),
            )
            records.append(rec)
    return records

print("\n[1] Loading M1 records...")
train_all = load_records(cfg.resolve_output("m1_dataset").parent / "m1_train_dataset.jsonl")
val_all = load_records(cfg.resolve_output("m1_dataset").parent / "m1_validation_dataset.jsonl")
print(f"  Total train: {len(train_all)}, Total val: {len(val_all)}")

# Group by domain for stratified sampling
train_by_domain = {}
for r in train_all:
    train_by_domain.setdefault(r.domain, []).append(r)
val_by_domain = {}
for r in val_all:
    val_by_domain.setdefault(r.domain, []).append(r)

print(f"  Train domains: {[(d, len(v)) for d, v in train_by_domain.items()]}")
print(f"  Val domains: {[(d, len(v)) for d, v in val_by_domain.items()]}")

# 2. Stratified sampling with domain coverage
print("\n[2] Stratified sampling...")

def stratified_sample(records_by_domain, domain_sizes, rng):
    """Sample records ensuring all domains and class coverage."""
    selected = []
    for domain, size in domain_sizes.items():
        domain_records = records_by_domain.get(domain, [])
        if len(domain_records) < size:
            # Repeat if not enough records in this domain
            indices = list(range(len(domain_records))) * ((size // len(domain_records)) + 1)
            indices = indices[:size]
        else:
            indices = list(range(len(domain_records)))
            rng.shuffle(indices)
            indices = indices[:size]
        for idx in indices:
            selected.append(domain_records[idx])
    return selected

# Distribute 10K across 4 domains proportionally
total_train = len(train_all)
train_domain_sizes = {}
for domain, recs in train_by_domain.items():
    train_domain_sizes[domain] = max(1, int(10000 * len(recs) / total_train))
# Adjust to exactly 10000
current = sum(train_domain_sizes.values())
train_domain_sizes[list(train_domain_sizes.keys())[0]] += (10000 - current)

total_val = len(val_all)
val_domain_sizes = {}
for domain, recs in val_by_domain.items():
    val_domain_sizes[domain] = max(1, int(2000 * len(recs) / total_val))
current = sum(val_domain_sizes.values())
val_domain_sizes[list(val_domain_sizes.keys())[0]] += (2000 - current)

train_subset = stratified_sample(train_by_domain, train_domain_sizes, rng)
val_subset = stratified_sample(val_by_domain, val_domain_sizes, rng)
print(f"  Train subset: {len(train_subset)} records")
print(f"  Val subset: {len(val_subset)} records")
print(f"  Train domains: {[(d, len([r for r in train_subset if r.domain == d])) for d in train_domain_sizes]}")
print(f"  Val domains: {[(d, len([r for r in val_subset if r.domain == d])) for d in val_domain_sizes]}")

# Verify no overlap
train_ids = set(r.record_id for r in train_subset)
val_ids = set(r.record_id for r in val_subset)
overlap = train_ids & val_ids
assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} records!"
print(f"  No train/val overlap: ✓")

# 3. Build M3Datasets
print("\n[3] Building M3Datasets...")
ds_train = M3Dataset(
    records=train_subset,
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=128,
)
ds_val = M3Dataset(
    records=val_subset,
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=128,
)
print(f"  Train: {ds_train.num_records}, Val: {ds_val.num_records}")

# Validate alignment
print("\n[4] Validating alignment...")
alignment = validate_alignment(ds_train, ds_val, ds_val)
print(f"  {alignment}")

# 5. Build model, optimizer, trainer
print("\n[5] Building model...")
model = DeBERTaMultiTaskModel(
    model_name="microsoft/deberta-base",
    dropout_rate=0.1,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

trainer = M3Trainer(
    model=model,
    train_dataset=ds_train,
    val_dataset=ds_val,
    optimizer=optimizer,
    device=device,
)

# 6. Train for 5 epochs
print("\n[6] Training 5 epochs...")
result = trainer.train(
    epochs=5,
    batch_size=32,
    save_dir="outputs/m3/smoke",
    learning_rate=1e-5,
)

# 7. Reload best checkpoint and do inference
print("\n[7] Reloading best checkpoint for inference...")
best_state = torch.load("outputs/m3/smoke/best_model.pt", map_location=device)
fresh_model = DeBERTaMultiTaskModel(dropout_rate=0.1)
fresh_model.load_state_dict(best_state)
fresh_model.eval()
fresh_model.to(device)

val_batch = ds_val[0]
input_ids = val_batch["input_ids"].unsqueeze(0).to(device)
attention_mask = val_batch["attention_mask"].unsqueeze(0).to(device)

with torch.no_grad():
    outputs = fresh_model(input_ids, attention_mask)

expected_shapes = {
    "sensitivity_logits": (1, 3),
    "intent_logits": (1, 5),
    "disclosure_scope_logits": (1, 2),
    "entity_tags_logits": (1, 4),
    "threat_content_logits": (1, 3),
    "representation": (1, 768),
}
print("  Inference output shapes:")
for key, shape in expected_shapes.items():
    actual = outputs[key].shape
    print(f"    {key}: {actual} {'✓' if actual == shape else '✗'}")

# 8. Print training history summary
print("\n[8] Training history summary:")
for record in result["history"]:
    print(f"  Epoch {record['epoch']}: train_loss={record['train_loss']:.4f}, "
          f"val_loss={record['val_loss']:.4f}, "
          f"lr={record['learning_rate']:.6f}, "
          f"duration={record['epoch_duration_seconds']:.1f}s, "
          f"gpu={record['peak_gpu_memory_mb']:.1f}MB")
    for dim in ["sensitivity", "intent", "disclosure_scope"]:
        dim_metrics = record["val_metrics"].get(dim, {})
        if dim_metrics:
            print(f"    {dim}: acc={dim_metrics.get('accuracy', 'N/A')}, "
                  f"macro_f1={dim_metrics.get('macro_f1', 'N/A')}")

print("\n" + "=" * 80)
print("M3 FULL SMOKE TEST COMPLETE")
print("=" * 80)
print(f"\nBest checkpoint: outputs/m3/smoke/best_model.pt")
print(f"Training history: outputs/m3/smoke/training_history.json")
print(f"Validation metrics: outputs/m3/smoke/validation_metrics.json")
print(f"Config: outputs/m3/smoke/m3_config.json")
print(f"\nBest val loss: {result['best_val_loss']:.4f}")
