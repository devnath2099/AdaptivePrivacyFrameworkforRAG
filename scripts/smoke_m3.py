"""Lightweight smoke test for M3 components.

Verifies the full M3 flow without running the full training loop:
1. Record alignment (M1 train/val/test → M2 soft labels)
2. Tokenizer output shapes
3. Model forward pass with all five heads
4. Loss computation and backward
5. Checkpoint save/load
"""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from m1_data_integration.config import load_config
from m1_data_integration.schemas import EvidenceBundle, UnifiedRecord
from m3_privacy_prediction.dataset import M3Dataset, validate_alignment
from m3_privacy_prediction.losses import compute_total_loss, soft_cross_entropy, binary_cross_entropy_with_logits
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.metrics import compute_all_metrics

print("=" * 80)
print("M3 DEVELOPMENT SMOKE TEST")
print("=" * 80)

# 1. Load M1 records and verify alignment with M2
print("\n[1] Loading M1 records...")
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
                record_id=d["record_id"], domain=d["domain"],
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

import json
train_records = load_records(cfg.resolve_output("m1_dataset").parent / "m1_train_dataset.jsonl")
val_records = load_records(cfg.resolve_output("m1_dataset").parent / "m1_validation_dataset.jsonl")
test_records = load_records(cfg.resolve_output("m1_dataset").parent / "m1_test_dataset.jsonl")
print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

# Use small subsets for smoke test
train_subset = train_records[:50]
val_subset = val_records[:25]
test_subset = test_records[:25]

# 2. Build M3Dataset with small subsets
print("\n[2] Building M3Dataset...")
ds_train = M3Dataset(
    records=train_subset,
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=64,
)
ds_val = M3Dataset(
    records=val_subset,
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=64,
)
print(f"  Train: {ds_train.num_records}, Val: {ds_val.num_records}")

# 3. Validate alignment (train vs val vs test)
print("\n[3] Validating alignment...")
ds_test = M3Dataset(
    records=test_subset,
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=64,
)
alignment = validate_alignment(ds_train, ds_val, ds_test)
print(f"  {alignment}")

# 4. Tokenizer output shapes
print("\n[4] Tokenizer output shapes...")
item = ds_train[0]
print(f"  input_ids: {item['input_ids'].shape}")
print(f"  attention_mask: {item['attention_mask'].shape}")
print(f"  soft_targets: {[(k, v.shape) for k, v in item['soft_targets'].items()]}")

# 5. Model forward pass (shape check) and prepare for backward
print("\n[5] Model forward pass...")
model = DeBERTaMultiTaskModel(model_name="microsoft/deberta-base", dropout_rate=0.1)
model.train()
input_ids = item["input_ids"].unsqueeze(0)
attention_mask = item["attention_mask"].unsqueeze(0)

outputs = model(input_ids, attention_mask)

expected_shapes = {
    "sensitivity_logits": (1, 3),
    "intent_logits": (1, 5),
    "disclosure_scope_logits": (1, 2),
    "entity_tags_logits": (1, 4),
    "threat_content_logits": (1, 3),
    "representation": (1, 768),
}
print("  Shape check:")
all_ok = True
for key, shape in expected_shapes.items():
    actual = outputs[key].shape
    ok = actual == shape
    all_ok = all_ok and ok
    print(f"    {key}: {actual} {'OK' if ok else 'FAIL'}")
assert all_ok, "Shape mismatch!"

# 6. Loss computation and backward (train mode for gradients)
print("\n[6] Loss computation and backward...")
model.train()
targets = {k: v.unsqueeze(0) for k, v in item["soft_targets"].items()}
total_loss, task_losses = compute_total_loss(outputs, targets)
print(f"  Total loss: {total_loss.item():.4f}")
for k, v in task_losses.items():
    print(f"    {k}: {v.item():.4f}")
total_loss.backward()
grad_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print(f"  Gradients finite: {grad_ok}")
assert torch.isfinite(total_loss), "Loss is not finite!"
assert grad_ok, "Gradients are not finite!"

# 7. Checkpoint save/load
print("\n[7] Checkpoint save/load...")
model.eval()
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "test_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    new_model = DeBERTaMultiTaskModel(dropout_rate=0.1)
    new_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    new_model.eval()

    with torch.no_grad():
        out1 = model(input_ids, attention_mask)
        out2 = new_model(input_ids, attention_mask)

    all_close = all(torch.allclose(out1[k], out2[k], atol=1e-6) for k in out1)
    print(f"  Checkpoint save/load matches: {all_close}")
    assert all_close, "Checkpoint save/load failed!"

# 8. Metrics computation
print("\n[8] Metrics computation...")
# Simulate predictions
logits_sensitivity = np.random.randn(25, 3)
targets_sensitivity = np.random.rand(25, 3)
metrics = compute_all_metrics(
    {"sensitivity": logits_sensitivity, "entity_tags": np.random.randn(25, 4)},
    {"sensitivity": targets_sensitivity, "entity_tags": np.random.rand(25, 4)},
)
print(f"  sensitivity: {metrics['sensitivity']}")
print(f"  entity_tags: {metrics['entity_tags']}")

print("\n" + "=" * 80)
print("ALL SMOKE TESTS PASSED")
print("=" * 80)
print(f"\nSummary:")
print(f"  Input text: normalized_text")
print(f"  Max length: 64 (dev)")
print(f"  Model: microsoft/deberta-base")
print(f"  Hidden size: 768")
print(f"  Train records: {len(train_subset)} (dev), {len(train_records)} (full)")
print(f"  Val records: {len(val_subset)} (dev), {len(val_records)} (full)")
print(f"  Test records: {len(test_subset)} (dev), {len(test_records)} (full)")
print(f"  Loss functions: soft_cross_entropy (sensitivity/intent/disclosure_scope), binary_cross_entropy_with_logits (entity_tags/threat_content)")
print(f"  Task weights: equal (1.0 each)")
print(f"  Checkpoint strategy: best by validation loss")
