"""Run M4: FGSM adversarial fine-tuning of M3 model.

Loads M3 checkpoint, performs 3 epochs of FGSM adversarial training,
evaluates on clean and adversarial validation sets, saves robust model.
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
from m3_privacy_prediction.dataset import M3Dataset, validate_alignment, build_m3_datasets
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss
from m3_privacy_prediction.metrics import compute_all_metrics
from m4_adversarial_training.trainer import M4Trainer

print("=" * 80)
print("M4 ADVERSARIAL TRAINING")
print("=" * 80)

# 1. Load config
cfg = load_config("configs/review1.yaml")
m3_cfg = cfg.raw.get("m3", {})
m4_cfg = cfg.raw.get("m4", {})

epsilon = float(m4_cfg.get("epsilon", 1e-3))
lambda_adv = float(m4_cfg.get("lambda_adv", 1.0))
epochs = int(m4_cfg.get("epochs", 3))
batch_size = int(m4_cfg.get("batch_size", 32))
learning_rate = float(m4_cfg.get("learning_rate", 1e-5))
max_length = int(m3_cfg.get("max_length", 128))
adv_val_batch_size = int(m4_cfg.get("adv_val_batch_size", 16))

print(f"Config: epsilon={epsilon}, lambda_adv={lambda_adv}, epochs={epochs}")
print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
print(f"Adversarial val batch size: {adv_val_batch_size}")

# 2. Load M3 checkpoint
print("\n[1] Loading M3 checkpoint...")
model = DeBERTaMultiTaskModel(
    model_name=m3_cfg.get("model_name", "microsoft/deberta-base"),
    dropout_rate=m3_cfg.get("dropout_rate", 0.1),
)
checkpoint = torch.load("outputs/m3/smoke/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.requires_grad_(True)
print("M3 checkpoint loaded successfully.")

# 3. Build M3 datasets (reuse same population)
print("\n[2] Building M3 datasets...")
train_all = M3Dataset(
    records=[],  # placeholder, will use build_m3_datasets
    weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
    text_field="normalized_text",
    max_length=max_length,
)
# Use build_m3_datasets for full data
datasets = build_m3_datasets(cfg)

# Use subset for M4 (controlled comparison)
# Use same 10K train / 2K val as M3 smoke test
rng = np.random.default_rng(42)
train_indices = rng.choice(len(datasets["train"]), size=10000, replace=False)
val_indices = rng.choice(len(datasets["val"]), size=2000, replace=False)

ds_train = Subset(datasets["train"], train_indices)
ds_val = Subset(datasets["val"], val_indices)

print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}")

# Validate alignment
alignment = validate_alignment(datasets["train"], datasets["val"], datasets["test"])
print(f"  Alignment: {alignment}")

# 4. Setup optimizer and trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3] Device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

trainer = M4Trainer(
    model=model,
    train_dataset=ds_train,
    val_dataset=ds_val,
    optimizer=optimizer,
    device=device,
)

# 5. Run M4 training
print(f"\n[4] Training M4 for {epochs} epochs...")
result = trainer.train(
    epochs=epochs,
    batch_size=batch_size,
    save_dir="outputs/m4",
    learning_rate=learning_rate,
    epsilon=epsilon,
    lambda_adv=lambda_adv,
    adv_val_batch_size=adv_val_batch_size,
)

# 6. Load best M4 checkpoint and evaluate
print("\n[5] Evaluating M4 on clean and adversarial validation...")
best_state = torch.load("outputs/m4/best_model.pt", map_location=device)
model.load_state_dict(best_state)

val_result = trainer.validate(adversarial=False)
adv_val_result = trainer.validate(adversarial=True, epsilon=epsilon,
                                 adv_val_batch_size=adv_val_batch_size)

print("\nM4 Clean Validation Metrics:")
for dim, m in val_result["metrics"].items():
    print(f"  {dim}: {m}")

print("\nM4 Adversarial Validation Metrics:")
for dim, m in adv_val_result["metrics"].items():
    print(f"  {dim}: {m}")

# 7. Load M3 checkpoint for comparison
m3_model = DeBERTaMultiTaskModel(
    model_name=m3_cfg.get("model_name", "microsoft/deberta-base"),
    dropout_rate=m3_cfg.get("dropout_rate", 0.1),
)
m3_checkpoint = torch.load("outputs/m3/smoke/best_model.pt", map_location=device)
m3_model.load_state_dict(m3_checkpoint)
m3_model.eval()
m3_trainer = M4Trainer(
    model=m3_model, train_dataset=ds_train, val_dataset=ds_val,
    optimizer=optimizer, device=device
)
m3_val = m3_trainer.validate(adversarial=False)
m3_adv = m3_trainer.validate(adversarial=True, epsilon=epsilon,
                             adv_val_batch_size=adv_val_batch_size)

print("\nM3 vs M4 Robustness Comparison:")
print(f"  M3 Clean: val_loss={m3_val['loss']:.4f}")
print(f"  M3 Adversarial: val_loss={m3_adv['loss']:.4f}")
print(f"  M4 Clean: val_loss={val_result['loss']:.4f}")
print(f"  M4 Adversarial: val_loss={adv_val_result['loss']:.4f}")

# 8. Save robustness comparison
robustness_comparison = {
    "m3_clean_val_loss": round(m3_val["loss"], 4),
    "m3_adversarial_val_loss": round(m3_adv["loss"], 4),
    "m4_clean_val_loss": round(val_result["loss"], 4),
    "m4_adversarial_val_loss": round(adv_val_result["loss"], 4),
    "epsilon": epsilon,
    "lambda_adv": lambda_adv,
    "m3_clean_metrics": {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                             for kk, vv in v.items()} for k, v in m3_val["metrics"].items()},
    "m3_adversarial_metrics": {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                                   for kk, vv in v.items()} for k, v in m3_adv["metrics"].items()},
    "m4_clean_metrics": {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                             for kk, vv in v.items()} for k, v in val_result["metrics"].items()},
    "m4_adversarial_metrics": {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                                   for kk, vv in v.items()} for k, v in adv_val_result["metrics"].items()},
}

with open("outputs/m4/robustness_comparison.json", "w") as f:
    json.dump(robustness_comparison, f, indent=2, default=str)

# 9. Save clean and adversarial validation metrics
with open("outputs/m4/clean_validation_metrics.json", "w") as f:
    json.dump(val_result["metrics"], f, indent=2, default=str)

with open("outputs/m4/adversarial_validation_metrics.json", "w") as f:
    json.dump(adv_val_result["metrics"], f, indent=2, default=str)

print("\nM4 training complete!")
print(f"Best checkpoint: outputs/m4/best_model.pt")