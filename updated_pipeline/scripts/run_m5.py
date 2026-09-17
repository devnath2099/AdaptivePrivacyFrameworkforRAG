"""Run M5: MC-Dropout uncertainty quantification with temperature scaling.

Loads M4 best checkpoint, fits temperature scaling on calibration split,
performs stochastic inference, computes uncertainty metrics and ECE,
and applies the deterministic decision engine.

Path-robust: uses PROJECT_ROOT env var or resolves from file location.
Outputs to outputs/m5/full_m4/ to avoid overwriting smoke-test artifacts.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Path-robust project root
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import build_m3_datasets
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m5_uncertainty.mc_dropout import run_m5_inference
from m5_uncertainty.temperature_scaling import fit_temperature_scaling
from m5_uncertainty.calibration import (
    compute_categorical_ece,
    compute_multilabel_ece,
    compute_correct_vs_incorrect_uncertainty,
    save_high_uncertainty_examples,
)
from m6_decision import RiskPolicyDecisionEngine

print("=" * 80)
print("M5 MC-DROPOUT UNCERTAINTY & TEMPERATURE CALIBRATION + M6 DECISION")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")

# 1. Load config
cfg = load_config(str(PROJECT_ROOT / "configs" / "review1.yaml"))
m3_cfg = cfg.raw.get("m3", {})
m5_cfg = cfg.raw.get("m5", {})
m6_cfg = cfg.raw.get("m6_decision", {})
temp_cfg = m5_cfg.get("temperature", {})

mc_passes = int(m5_cfg.get("mc_passes", 20))
calibration_bins = int(m5_cfg.get("calibration_bins", 10))
multi_label_threshold = float(m5_cfg.get("multi_label_threshold", 0.5))
batch_size = int(m5_cfg.get("batch_size", 32))
num_iterations = int(temp_cfg.get("num_iterations", 100))
optimizer_type = temp_cfg.get("optimizer_type", "lbfgs")
lr = float(temp_cfg.get("learning_rate", 0.05))
calib_mc_passes = int(temp_cfg.get("mc_passes", 5))
calibration_fraction = float(temp_cfg.get("calibration_fraction", 0.3))
fit_on_calibration_only = temp_cfg.get("fit_on_calibration_only", True)

print(f"Config: mc_passes={mc_passes}, bins={calibration_bins}")
print(f"  Temperature: optimizer={optimizer_type}, lr={lr}, iterations={num_iterations}")
print(f"  Calibration fraction: {calibration_fraction}")
print(f"  Fit on calibration only: {fit_on_calibration_only}")

# 2. Load M4 checkpoint
print("\n[1] Loading M4 checkpoint...")
model = DeBERTaMultiTaskModel(
    model_name=m3_cfg.get("model_name", "microsoft/deberta-base"),
    dropout_rate=m3_cfg.get("dropout_rate", 0.1),
)
ckpt_path = PROJECT_ROOT / "outputs" / "m4" / "best_model.pt"
if not ckpt_path.exists():
    ckpt_path = PROJECT_ROOT / "outputs" / "m4" / "full" / "best_model.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(f"M4 checkpoint not found at {ckpt_path}")
checkpoint = torch.load(str(ckpt_path), map_location="cpu")
model.load_state_dict(checkpoint)
print(f"M4 checkpoint loaded from {ckpt_path}.")

# 3. Build validation dataset
print("\n[2] Building validation dataset...")
datasets = build_m3_datasets(cfg)
ds_val = datasets["val"]

rng = np.random.default_rng(42)
val_size = min(2000, len(ds_val))
val_indices = rng.choice(len(ds_val), size=val_size, replace=False)

# Split into calibration and evaluation
n_calib = max(1, int(val_size * calibration_fraction))
calib_indices = val_indices[:n_calib]
eval_indices = val_indices[n_calib:]

ds_calib = Subset(ds_val, calib_indices.tolist())
ds_eval = Subset(ds_val, eval_indices.tolist())
print(f"  Val subset: {val_size} records")
print(f"  Calibration: {len(ds_calib)}, Evaluation: {len(ds_eval)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Fit temperature scaling on calibration split only
print(f"\n[3] Fitting temperature scaling on calibration split ({len(ds_calib)} records)...")
temperatures, ece_before, ece_after, nll_before, nll_after = fit_temperature_scaling(
    model, ds_calib, ds_eval, device=device,
    num_iterations=num_iterations, optimizer_type=optimizer_type,
    lr=lr, mc_passes=calib_mc_passes, batch_size=batch_size,
)
print(f"  Temperatures: {temperatures}")
print(f"  ECE before: {ece_before['overall']:.4f}")
print(f"  ECE after: {ece_after['overall']:.4f}")
print(f"  NLL before: {nll_before:.4f}")
print(f"  NLL after: {nll_after:.4f}")

# 5. Run M5 inference on evaluation split with calibrated temperatures
print(f"\n[4] Running M5 inference on evaluation split (T={temperatures})...")
results, uncertainty_summary = run_m5_inference(
    model, ds_eval, mc_passes=mc_passes,
    batch_size=batch_size, device=device, temperature=temperatures,
)

print("\nUncertainty Summary (after temperature scaling):")
for dim, stats in uncertainty_summary.items():
    print(f"  {dim}: entropy={stats['mean_entropy']:.4f}, "
          f"variance={stats['mean_variance']:.4f}, "
          f"confidence={stats['mean_confidence']:.4f}")

# 6. Compute calibration metrics
print(f"\n[5] Computing ECE calibration on evaluation split...")
all_preds = {}
all_targets = {}

for r in results:
    for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]:
        if dim not in all_preds:
            all_preds[dim] = []
            all_targets[dim] = []
        all_preds[dim].append(np.array(r[f"{dim}_mean"]))

for dim in all_preds:
    all_preds[dim] = np.array(all_preds[dim])

# Get targets from evaluation subset
for i, idx in enumerate(eval_indices):
    batch = ds_val[idx]
    for dim in batch["soft_targets"]:
        if dim not in all_targets:
            all_targets[dim] = []
        all_targets[dim].append(batch["soft_targets"][dim].numpy())

for dim in all_targets:
    all_targets[dim] = np.array(all_targets[dim])

categorical_ece = {}
for dim in ["sensitivity", "intent", "disclosure_scope"]:
    ece, bin_stats = compute_categorical_ece(all_preds[dim], all_targets[dim], calibration_bins)
    categorical_ece[dim] = {"ece": ece, "bin_stats": bin_stats}
    print(f"  {dim} ECE: {ece:.4f}")

multilabel_ece = {}
for dim in ["entity_tags", "threat_content"]:
    macro_ece, bin_stats, per_cat = compute_multilabel_ece(
        all_preds[dim], all_targets[dim], multi_label_threshold, calibration_bins
    )
    multilabel_ece[dim] = {"macro_ece": macro_ece, "per_category_ece": per_cat}
    print(f"  {dim} macro ECE: {macro_ece:.4f}")

all_variances = {
    dim: np.array([r[f"{dim}_variance"] for r in results])
    for dim in all_preds
}
correct_vs_incorrect = compute_correct_vs_incorrect_uncertainty(
    all_preds, all_variances, all_targets, threshold=multi_label_threshold
)

# 7. Apply M6/M7 decision engine
print(f"\n[6] Applying M6/M7 decision engine...")
decision_engine = RiskPolicyDecisionEngine(config=m6_cfg)

# Add composite entropy to results
for r in results:
    entropies = [r.get(f"{dim}_entropy", 0) for dim in
        ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]]
    r["composite_entropy"] = sum(entropies) / len(entropies)

decision_results = decision_engine.predict_batch(results)
decision_summary = decision_engine.summarize(decision_results)

print("\nDecision Summary:")
print(f"  Total records: {decision_summary['total_records']}")
print(f"  Risk distribution: {decision_summary['risk_distribution']}")
print(f"  Policy distribution: {decision_summary['policy_distribution']}")
print(f"  Fallback triggered: {decision_summary['fallback_count']}")

# 8. Save outputs to separate full_m4 directory
print("\n[7] Saving outputs...")
full_m5_dir = PROJECT_ROOT / "outputs" / "m5" / "full_m4"
full_m5_dir.mkdir(parents=True, exist_ok=True)
Path("outputs/m6").mkdir(parents=True, exist_ok=True)

# uncertainty_predictions.jsonl
with open(full_m5_dir / "uncertainty_predictions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# uncertainty_summary.json
with open(full_m5_dir / "uncertainty_summary.json", "w") as f:
    json.dump(uncertainty_summary, f, indent=2)

# calibration_metrics.json (with temperature info)
calibration_metrics = {
    "temperatures": temperatures,
    "nll_before": nll_before,
    "nll_after": nll_after,
    "ece_before": ece_before,
    "ece_after": ece_after,
    "categorical_ece": categorical_ece,
    "multilabel_ece": multilabel_ece,
    "correct_vs_incorrect_uncertainty": correct_vs_incorrect,
    "calibration_fraction": calibration_fraction,
    "fit_on_calibration_only": fit_on_calibration_only,
    "optimizer": optimizer_type,
    "num_iterations": num_iterations,
}
with open(full_m5_dir / "calibration_metrics.json", "w") as f:
    json.dump(calibration_metrics, f, indent=2, default=str)

# high_uncertainty_examples.json
high_uncertainty = save_high_uncertainty_examples(results, top_k=10)
with open(full_m5_dir / "high_uncertainty_examples.json", "w") as f:
    json.dump(high_uncertainty, f, indent=2, default=str)

# m5_config.json (manifest)
m5_config = {
    "mc_passes": mc_passes,
    "calibration_bins": calibration_bins,
    "multi_label_threshold": multi_label_threshold,
    "batch_size": batch_size,
    "temperature": temperatures,
    "nll_before": nll_before,
    "nll_after": nll_after,
    "ece_before": ece_before,
    "ece_after": ece_after,
    "model_loaded_from": str(ckpt_path),
    "calibration_split": f"{len(ds_calib)} of {val_size} records",
    "evaluation_split": f"{len(ds_eval)} of {val_size} records",
    "optimizer": optimizer_type,
    "fit_on_calibration_only": fit_on_calibration_only,
}
with open(full_m5_dir / "m5_config.json", "w") as f:
    json.dump(m5_config, f, indent=2)

# decision_results.jsonl
with open(PROJECT_ROOT / "outputs" / "m6" / "decision_results.jsonl", "w") as f:
    for r in decision_results:
        f.write(json.dumps(r) + "\n")

with open(PROJECT_ROOT / "outputs" / "m6" / "decision_summary.json", "w") as f:
    json.dump(decision_summary, f, indent=2)

m6_config_out = {
    "uncertainty_threshold": m6_cfg.get("uncertainty_threshold", 0.5),
    "conservative_fallback": m6_cfg.get("conservative_fallback", True),
    "risk_thresholds": m6_cfg.get("risk_score_thresholds", {}),
    "task_weights": m6_cfg.get("task_weights", {}),
}
with open(PROJECT_ROOT / "outputs" / "m6" / "m6_config.json", "w") as f:
    json.dump(m6_config_out, f, indent=2)

# Combined output
with open(full_m5_dir / "uncertainty_predictions_with_decision.jsonl", "w") as f:
    for r, d in zip(results, decision_results):
        combined = {**r, **d}
        f.write(json.dumps(combined) + "\n")

# Print manifest
print("\n" + "=" * 80)
print("MANIFEST")
print("=" * 80)
print(f"  M4 checkpoint: {ckpt_path}")
print(f"  M5 outputs: {full_m5_dir}/")
print(f"    - uncertainty_predictions.jsonl ({len(results)} records)")
print(f"    - calibration_metrics.json (T={temperatures})")
print(f"    - m5_config.json (manifest)")
print(f"  M6 outputs: outputs/m6/")
print(f"    - decision_results.jsonl")
print(f"    - decision_summary.json")
print(f"  M5 evaluation split: {len(ds_eval)} records (from {val_size} total val)")
print(f"  M5 calibration split: {len(ds_calib)} records (NOT used for evaluation)")
print("\nM5 + M6 complete!")