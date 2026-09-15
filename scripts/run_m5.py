"""Run M5: MC-Dropout uncertainty quantification and calibration.

Loads M4 best checkpoint, performs stochastic inference,
computes uncertainty metrics and ECE calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from tqdm import tqdm

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import M3Dataset, build_m3_datasets
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.metrics import compute_all_metrics
from m5_uncertainty.mc_dropout import (
    set_dropout_training,
    convert_to_probabilities,
    compute_predictive_mean,
    compute_predictive_variance,
    compute_uncertainty_summary,
    run_m5_inference,
)
from m5_uncertainty.calibration import (
    compute_categorical_ece,
    compute_multilabel_ece,
    compute_correct_vs_incorrect_uncertainty,
    save_high_uncertainty_examples,
)

print("=" * 80)
print("M5 MC-DROPOUT UNCERTAINTY & CALIBRATION")
print("=" * 80)

# 1. Load config
cfg = load_config("configs/review1.yaml")
m3_cfg = cfg.raw.get("m3", {})
m5_cfg = cfg.raw.get("m5", {})

mc_passes = int(m5_cfg.get("mc_passes", 20))
calibration_bins = int(m5_cfg.get("calibration_bins", 10))
multi_label_threshold = float(m5_cfg.get("multi_label_threshold", 0.5))
batch_size = int(m5_cfg.get("batch_size", 32))
max_length = int(m3_cfg.get("max_length", 128))

print(f"Config: mc_passes={mc_passes}, bins={calibration_bins}, threshold={multi_label_threshold}")

# 2. Load M4 checkpoint
print("\n[1] Loading M4 checkpoint...")
model = DeBERTaMultiTaskModel(
    model_name=m3_cfg.get("model_name", "microsoft/deberta-base"),
    dropout_rate=m3_cfg.get("dropout_rate", 0.1),
)
checkpoint = torch.load("outputs/m4/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint)
print("M4 checkpoint loaded successfully.")

# 3. Build validation dataset
print("\n[2] Building validation dataset...")
datasets = build_m3_datasets(cfg)
ds_val = datasets["val"]

# Use subset for faster M5
rng = np.random.default_rng(42)
val_indices = rng.choice(len(ds_val), size=min(2000, len(ds_val)), replace=False)
ds_val_subset = torch.utils.data.Subset(ds_val, val_indices)
print(f"  Val subset: {len(ds_val_subset)} records")

# 4. Run MC-Dropout inference
print(f"\n[3] Running MC-Dropout inference (T={mc_passes})...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

results, uncertainty_summary = run_m5_inference(
    model, ds_val_subset, mc_passes=mc_passes, batch_size=batch_size, device=device
)

print("\nUncertainty Summary:")
for dim, stats in uncertainty_summary.items():
    print(f"  {dim}: entropy={stats['mean_entropy']:.4f}, "
          f"variance={stats['mean_variance']:.4f}, "
          f"confidence={stats['mean_confidence']:.4f}")

# 5. Compute calibration metrics
print(f"\n[4] Computing ECE calibration (bins={calibration_bins})...")

# Build predictions and targets for calibration
all_preds = {}
all_targets = {}
all_record_ids = []

for r in results:
    all_record_ids.append(r["record_id"])
    for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]:
        if dim not in all_preds:
            all_preds[dim] = []
            all_targets[dim] = []
        all_preds[dim].append(np.array(r[f"{dim}_mean"]))

for dim in all_preds:
    all_preds[dim] = np.array(all_preds[dim])

# Get targets from dataset
for i, idx in enumerate(val_indices):
    batch = ds_val[idx]
    for dim in batch["soft_targets"]:
        if dim not in all_targets:
            all_targets[dim] = []
        all_targets[dim].append(batch["soft_targets"][dim].numpy())

for dim in all_targets:
    all_targets[dim] = np.array(all_targets[dim])

# Categorical ECE
categorical_ece = {}
for dim in ["sensitivity", "intent", "disclosure_scope"]:
    ece, bin_stats = compute_categorical_ece(all_preds[dim], all_targets[dim], calibration_bins)
    categorical_ece[dim] = {"ece": ece, "bin_stats": bin_stats}
    print(f"  {dim} ECE: {ece:.4f}")

# Multi-label ECE
multilabel_ece = {}
for dim in ["entity_tags", "threat_content"]:
    macro_ece, bin_stats, per_cat = compute_multilabel_ece(
        all_preds[dim], all_targets[dim], multi_label_threshold, calibration_bins
    )
    multilabel_ece[dim] = {"macro_ece": macro_ece, "per_category_ece": per_cat}
    print(f"  {dim} macro ECE: {macro_ece:.4f}")

# Correct vs incorrect uncertainty
all_variances = {
    dim: np.array([r[f"{dim}_variance"] for r in results])
    for dim in all_preds
}
correct_vs_incorrect = compute_correct_vs_incorrect_uncertainty(
    all_preds, all_variances, all_targets, threshold=multi_label_threshold
)

# 6. Save high uncertainty examples
high_uncertainty = save_high_uncertainty_examples(results, top_k=10)

# 7. Save outputs
print("\n[5] Saving outputs...")
Path("outputs/m5").mkdir(parents=True, exist_ok=True)

# uncertainty_predictions.jsonl
with open("outputs/m5/uncertainty_predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# uncertainty_summary.json
with open("outputs/m5/uncertainty_summary.json", "w") as f:
    json.dump(uncertainty_summary, f, indent=2)

# calibration_metrics.json
calibration_metrics = {
    "categorical_ece": categorical_ece,
    "multilabel_ece": multilabel_ece,
    "correct_vs_incorrect_uncertainty": correct_vs_incorrect,
}
with open("outputs/m5/calibration_metrics.json", "w") as f:
    json.dump(calibration_metrics, f, indent=2, default=str)

# high_uncertainty_examples.json
with open("outputs/m5/high_uncertainty_examples.json", "w") as f:
    json.dump(high_uncertainty, f, indent=2, default=str)

# m5_config.json
m5_config = {
    "mc_passes": mc_passes,
    "calibration_bins": calibration_bins,
    "multi_label_threshold": multi_label_threshold,
    "batch_size": batch_size,
    "model_loaded_from": "outputs/m4/best_model.pt",
}
with open("outputs/m5/m5_config.json", "w") as f:
    json.dump(m5_config, f, indent=2)

print("\nM5 complete!")
print(f"  Outputs: outputs/m5/")
