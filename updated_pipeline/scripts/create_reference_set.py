"""Create a stratified annotated reference set for validating M2 weaknesses
and downstream M4/M5/M6 decisions.

300 queries split into:
  - 200 labelled queries for rule/threshold design (calibration portion)
  - 100 untouched queries for final evaluation (never used for threshold design)

Stratification dimensions:
  1. Domain: healthcare, financial, multi_hop_qa, general_qa
  2. Predicted sensitivity: low, medium, high
  3. M5 uncertainty: high vs low
  4. Intent all-abstain / uncertain cases
  5. Rare contact-identifier and membership-inference cases
  6. Disclosure-conflict cases

Path-robust: uses PROJECT_ROOT env var.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from tqdm import tqdm

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import build_m3_datasets

print("=" * 80)
print("REFERENCE SET CREATION — 300-Query Stratified Annotation")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")

# 1. Load config
cfg = load_config(str(PROJECT_ROOT / "configs" / "review1.yaml"))

# 2. Load existing M5 uncertainty predictions (from full M4 run)
print("\n[1] Loading M5 uncertainty predictions...")
m5_dir = PROJECT_ROOT / "outputs" / "m5" / "full_m4"
uncertainty_path = m5_dir / "uncertainty_predictions.jsonl"
if uncertainty_path.exists():
    with open(uncertainty_path) as f:
        uncertainty_records = [json.loads(line) for line in f]
    print(f"  Loaded {len(uncertainty_records)} records from M5 full_m4 output")
else:
    # Fall back to smoke-test output
    uncertainty_path = PROJECT_ROOT / "outputs" / "m5" / "uncertainty_predictions.jsonl"
    if uncertainty_path.exists():
        with open(uncertainty_path) as f:
            uncertainty_records = [json.loads(line) for line in f]
        print(f"  Loaded {len(uncertainty_records)} records from M5 smoke output")
    else:
        uncertainty_records = []
        print("  No M5 output found; using random sampling")

# 3. Build dataset
print("\n[2] Building validation dataset...")
datasets = build_m3_datasets(cfg)
ds_val = datasets["val"]

# 4. Load M2 weak labels for cross-validation
print("\n[3] Loading M2 weak labels...")
m2_dir = PROJECT_ROOT / "outputs" / "m2_weak_labels"
m2_labels = {}
if m2_dir.exists():
    for f in m2_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                m2_labels.update(data)
    print(f"  Loaded M2 labels for {len(m2_labels)} records")

# 5. Collect records with metadata for stratification
print("\n[4] Collecting records for stratification...")
domain_labels = ["healthcare", "fiqa", "hotpotqa", "nq"]
domain_records = {d: [] for d in domain_labels}

for i in range(min(len(ds_val), 2000)):
    batch = ds_val[i]
    domain = batch.get("domain", "unknown")
    if domain in domain_records:
        # Look up M5 uncertainty for this record
        rid = batch.get("record_id", f"record_{i}")
        m5_rec = None
        for ur in uncertainty_records:
            if ur.get("record_id") == rid:
                m5_rec = ur
                break

        # Determine uncertainty level
        composite_entropy = m5_rec.get("composite_entropy", 0.0) if m5_rec else 0.0
        uncertainty_level = "high" if composite_entropy > 0.5 else "low"

        # Determine predicted sensitivity
        sensitivity_mean = m5_rec.get("sensitivity_mean", [0.33, 0.33, 0.34]) if m5_rec else [0.33, 0.33, 0.34]
        sens_pred = int(np.argmax(sensitivity_mean))
        sens_label = ["low", "medium", "high"][sens_pred]

        # Check for disclosure conflicts
        disclosure_mean = m5_rec.get("disclosure_scope_mean", [0.5, 0.5]) if m5_rec else [0.5, 0.5]
        disclosure_pred = int(np.argmax(disclosure_mean))

        domain_records[domain].append({
            "index": i,
            "record_id": rid,
            "domain": domain,
            "sensitivity_pred": sens_label,
            "uncertainty_level": uncertainty_level,
            "disclosure_pred": disclosure_pred,
            "composite_entropy": composite_entropy,
            "m2_labels_matched": rid in m2_labels,
        })

# 6. Stratified sampling: 200 calibration + 100 evaluation
print("\n[5] Performing stratified sampling (200 calibration + 100 evaluation)...")
target_total = 300
target_per_domain = 75  # 300 / 4 domains
calibration_fraction = 200 / 300  # 2/3 for calibration

calibration_set = []
evaluation_set = []

for domain in domain_labels:
    available = domain_records.get(domain, [])
    if len(available) == 0:
        continue

    # Stratify within domain by sensitivity × uncertainty
    strata = {}
    for rec in available:
        key = (rec["sensitivity_pred"], rec["uncertainty_level"])
        if key not in strata:
            strata[key] = []
        strata[key].append(rec)

    # Sample from each stratum proportionally
    domain_calib = []
    domain_eval = []
    for key, stratum_recs in strata.items():
        n_stratum = min(len(stratum_recs), max(1, int(target_per_domain / len(strata))))
        np.random.shuffle(stratum_recs)
        selected = stratum_recs[:n_stratum]
        n_calib = max(1, int(len(selected) * calibration_fraction))
        domain_calib.extend(selected[:n_calib])
        domain_eval.extend(selected[n_calib:])

    # If we have fewer than target, add more from remaining
    if len(domain_calib) < int(target_per_domain * calibration_fraction):
        remaining = [r for r in available if r not in domain_calib and r not in domain_eval]
        np.random.shuffle(remaining)
        needed = int(target_per_domain * calibration_fraction) - len(domain_calib)
        domain_calib.extend(remaining[:needed])

    if len(domain_eval) < target_per_domain - len(domain_calib):
        remaining = [r for r in available if r not in domain_calib and r not in domain_eval]
        np.random.shuffle(remaining)
        needed = (target_per_domain - len(domain_calib)) - len(domain_eval)
        domain_eval.extend(remaining[:needed])

    calibration_set.extend(domain_calib)
    evaluation_set.extend(domain_eval)

print(f"  Calibration: {len(calibration_set)} records")
print(f"  Evaluation: {len(evaluation_set)} records")
print(f"  Per domain: {[len([r for r in calibration_set if r['domain'] == d]) for d in domain_labels]}")

# 7. Build annotation templates
print("\n[6] Building reference set...")

def build_annotation_record(record_meta, dataset):
    """Build an annotation template from a stratified record."""
    idx = record_meta["index"]
    batch = dataset[idx]
    rid = record_meta["record_id"]
    domain = record_meta["domain"]
    text = batch.get("text", "")[:200]

    # Find M2 labels
    m2_match = m2_labels.get(rid, None)

    # Find M5 uncertainty
    m5_rec = None
    for ur in uncertainty_records:
        if ur.get("record_id") == rid:
            m5_rec = ur
            break

    annotation = {
        "record_id": rid,
        "domain": domain,
        "text_snippet": text,
        "annotator": "TODO_ASSIGN",
        "annotation_date": "TODO_DATE",
        "sensitivity": {"label": "TODO", "confidence": 0.0, "evidence": ""},
        "intent": {"label": "TODO", "confidence": 0.0, "evidence": ""},
        "disclosure_scope": {"label": "TODO", "confidence": 0.0, "evidence": ""},
        "entity_tags": {"has_person": False, "has_organization": False, "has_location": False, "has_contact_identifier": False},
        "threat_content": {"re_identification": False, "attribute_inference": False, "membership_inference": False},
        "risk_tier": "TODO",
        "expected_policy": "TODO",
        "m2_weak_labels_matched": m2_match is not None,
        "m2_conflict_detected": False,
        "m5_uncertainty_level": record_meta["uncertainty_level"],
        "m5_composite_entropy": round(record_meta["composite_entropy"], 4),
        "m5_sensitivity_pred": record_meta["sensitivity_pred"],
        "m5_disclosure_pred": record_meta["disclosure_pred"],
        "notes": "",
        "set_type": "calibration",
    }

    # Priority annotation criteria
    priority = []
    if record_meta["uncertainty_level"] == "high":
        priority.append("high_M5_uncertainty")
    if record_meta["sensitivity_pred"] == "high":
        priority.append("high_sensitivity_prediction")
    if m2_match:
        priority.append("has_M2_weak_labels")
    if record_meta["disclosure_pred"] == 1:  # multi_document
        priority.append("multi_document_disclosure")
    entity_tags = m5_rec.get("entity_tags_mean", []) if m5_rec else []
    if len(entity_tags) > 3 and entity_tags[3] > 0.5:
        priority.append("contact_identifier_present")

    annotation["annotation_priority"] = priority
    return annotation

# Build calibration set
calibration_annotations = []
for rec_meta in calibration_set:
    annotation = build_annotation_record(rec_meta, ds_val)
    calibration_annotations.append(annotation)

# Build evaluation set
evaluation_annotations = []
for rec_meta in evaluation_set:
    annotation = build_annotation_record(rec_meta, ds_val)
    annotation["set_type"] = "evaluation"
    evaluation_annotations.append(annotation)

# 8. Save reference set
print("\n[7] Saving reference set...")
ref_dir = PROJECT_ROOT / "outputs" / "reference_set"
ref_dir.mkdir(parents=True, exist_ok=True)

with open(ref_dir / "reference_set_calibration_200.jsonl", "w") as f:
    for rec in calibration_annotations:
        f.write(json.dumps(rec) + "\n")

with open(ref_dir / "reference_set_evaluation_100.jsonl", "w") as f:
    for rec in evaluation_annotations:
        f.write(json.dumps(rec) + "\n")

# Summary
summary = {
    "total_records": 300,
    "calibration_records": len(calibration_annotations),
    "evaluation_records": len(evaluation_annotations),
    "domain_distribution": {d: len([r for r in calibration_annotations + evaluation_annotations if r["domain"] == d]) for d in domain_labels},
    "set_types": {
        "calibration": len(calibration_annotations),
        "evaluation": len(evaluation_annotations),
    },
    "priority_criteria": [
        "High M5 composite entropy (>0.5)",
        "High predicted sensitivity",
        "M2 weak labels present",
        "Multi-document disclosure",
        "Contact identifier present",
    ],
    "annotation_guide": "Annotate independently from M2/M5. Use M2 weak labels and M5 uncertainty as starting points only.",
}

with open(ref_dir / "reference_set_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Annotation guide
guide = {
    "instructions": "Annotate each query independently from M2/M5 predictions. "
    "Use M2 weak labels and M5 uncertainty as starting points only.",
    "label_taxonomy": cfg.raw.get("label_taxonomy", {}),
    "risk_tiers": ["low", "medium", "high", "critical"],
    "policy_codes": ["P0_low", "P1_moderate", "P2_high", "P3_restrictive_fallback"],
    "stratification_dimensions": [
        "Domain (4 domains)",
        "Predicted sensitivity (low/medium/high)",
        "M5 uncertainty (high/low, threshold=0.5)",
        "Intent all-abstain/uncertain cases",
        "Rare contact-identifier and membership-inference cases",
        "Disclosure-conflict cases",
    ],
    "calibration_portion": 200,
    "evaluation_portion": 100,
    "note": "Calibration portion used for rule/threshold design. Evaluation portion NEVER used for threshold design.",
}

with open(ref_dir / "annotation_guide.json", "w") as f:
    json.dump(guide, f, indent=2)

# 9. Save combined reference set
with open(ref_dir / "reference_set_300_combined.jsonl", "w") as f:
    for rec in calibration_annotations + evaluation_annotations:
        f.write(json.dumps(rec) + "\n")

print("\nReference set creation complete!")
print(f"  Calibration: {len(calibration_annotations)} records → {ref_dir / 'reference_set_calibration_200.jsonl'}")
print(f"  Evaluation: {len(evaluation_annotations)} records → {ref_dir / 'reference_set_evaluation_100.jsonl'}")
print(f"  Combined: {ref_dir / 'reference_set_300_combined.jsonl'}")
print(f"  Guide: {ref_dir / 'annotation_guide.json'}")
print(f"  Summary: {ref_dir / 'reference_set_summary.json'}")