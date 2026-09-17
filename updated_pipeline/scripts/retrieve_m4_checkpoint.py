"""Retrieve full M4 checkpoint from Kaggle.

Usage on Kaggle notebook:
  python scripts/retrieve_m4_checkpoint.py

This exports the full M4 checkpoint from Kaggle training to a
downloadable location (Google Drive or local /kaggle/working).

After retrieval, verify the checkpoint config/metrics before proceeding.
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

import torch

print("=" * 80)
print("M4 CHECKPOINT RETRIEVAL")
print("=" * 80)

# 1. Check if M4 checkpoint exists locally
print("\n[1] Checking for M4 checkpoint...")
ckpt_paths = [
    PROJECT_ROOT / "outputs" / "m4" / "best_model.pt",
    PROJECT_ROOT / "outputs" / "m4" / "full" / "best_model.pt",
    Path("/kaggle/working/outputs/m4/best_model.pt"),
    Path("/content/outputs/m4/best_model.pt"),
]

found_ckpt = None
for p in ckpt_paths:
    if p.exists():
        found_ckpt = p
        break

if found_ckpt:
    print(f"  Found checkpoint at {found_ckpt}")
    print(f"  Size: {found_ckpt.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print("  No checkpoint found locally.")
    print("  Run this script on the Kaggle notebook where M4 training completed.")
    print("  Expected location: /kaggle/working/outputs/m4/best_model.pt")

# 2. If found, verify checkpoint config/metrics
if found_ckpt:
    print(f"\n[2] Verifying checkpoint...")
    try:
        checkpoint = torch.load(str(found_ckpt), map_location="cpu")
        print(f"  Checkpoint loaded successfully.")
        print(f"  Keys: {list(checkpoint.keys())[:5]}...")

        # Save checkpoint info
        ckpt_info = {
            "path": str(found_ckpt),
            "size_bytes": found_ckpt.stat().st_size,
            "checkpoint_keys": list(checkpoint.keys()),
            "verified": True,
        }

        # Check for M4 config in outputs/m4/
        m4_dir = PROJECT_ROOT / "outputs" / "m4"
        config_files = list(m4_dir.glob("*.json")) if m4_dir.exists() else []
        ckpt_info["config_files"] = [str(f) for f in config_files]

        with open(m4_dir / "checkpoint_info.json", "w") as f:
            json.dump(ckpt_info, f, indent=2)

        print(f"  Checkpoint info saved to {m4_dir / 'checkpoint_info.json'}")
        print(f"\n  READY TO PROCEED: Run run_m5.py for full M5 pipeline.")

    except Exception as e:
        print(f"  ERROR verifying checkpoint: {e}")

# 3. If not found locally, copy from Kaggle Drive
else:
    print(f"\n[3] Attempting to copy from Google Drive...")
    drive_path = Path("/content/drive/MyDrive/review1_cache")
    if drive_path.exists():
        files = list(drive_path.glob("**/best_model.pt"))
        if files:
            print(f"  Found {len(files)} checkpoint files on Drive")
            for f in files:
                print(f"    {f} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print("  No checkpoints found on Drive")
    else:
        print("  Google Drive path not accessible")

    print("\n  ACTION REQUIRED: On the Kaggle notebook, run:")
    print(f"    cp /kaggle/working/outputs/m4/best_model.pt {PROJECT_ROOT}/outputs/m4/best_model.pt")
    print(f"    or")
    print(f"    gdown <file_id> --output {PROJECT_ROOT}/outputs/m4/best_model.pt")