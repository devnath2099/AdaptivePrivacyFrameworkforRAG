"""End-to-end M4→M5→M6 pipeline runner.

Path-robust: uses PROJECT_ROOT env var.
Outputs to separate directories to avoid overwriting smoke-test artifacts.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 80)
print("M4 → M5 → M6/7 END-TO-END PIPELINE")
print("=" * 80)
print(f"Project root: {PROJECT_ROOT}")

# Step 1: Run M5 with temperature scaling and decision engine
print("\n[1] Running M5 (temperature calibration + uncertainty quantification)...")
import subprocess
result = subprocess.run(
    [sys.executable, str(SCRIPTS_DIR / "run_m5.py")],
    env={**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)},
    cwd=str(PROJECT_ROOT),
)
if result.returncode != 0:
    print(f"ERROR: M5 pipeline failed with return code {result.returncode}")
    sys.exit(1)
print("M5 pipeline complete.")

# Step 2: Verify outputs
print("\n[2] Verifying outputs...")
m5_dir = PROJECT_ROOT / "outputs" / "m5" / "full_m4"
m6_dir = PROJECT_ROOT / "outputs" / "m6"

required_m5 = ["uncertainty_predictions.jsonl", "calibration_metrics.json", "m5_config.json"]
required_m6 = ["decision_results.jsonl", "decision_summary.json", "m6_config.json"]

for f in required_m5:
    path = m5_dir / f
    if not path.exists():
        print(f"WARNING: Missing {path}")
    else:
        print(f"  OK: {path}")

for f in required_m6:
    path = m6_dir / f
    if not path.exists():
        print(f"WARNING: Missing {path}")
    else:
        print(f"  OK: {path}")

print("\nPipeline complete!")
print(f"  M5 outputs: outputs/m5/full_m4/")
print(f"  M6 outputs: outputs/m6/")

# Load and print summary
with open(m5_dir / "m5_config.json") as f:
    m5_cfg = json.load(f)
with open(m6_dir / "decision_summary.json") as f:
    m6_summary = json.load(f)

print(f"\nSummary:")
print(f"  Temperature: {m5_cfg.get('temperature', {})}")
print(f"  NLL before/after: {m5_cfg.get('nll_before'):.4f} / {m5_cfg.get('nll_after'):.4f}")
print(f"  ECE before/after: {m5_cfg.get('ece_before', {}).get('overall'):.4f} / {m5_cfg.get('ece_after', {}).get('overall'):.4f}")
print(f"  Risk distribution: {m6_summary.get('risk_distribution')}")
print(f"  Policy distribution: {m6_summary.get('policy_distribution')}")
print(f"  Fallback count: {m6_summary.get('fallback_count')}")