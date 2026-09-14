"""Run M3: DeBERTa multi-task privacy prediction.

Usage:
    python scripts/run_m3.py [--dev] [--config configs/review1.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config
from m3_privacy_prediction.pipeline import run_m3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 training")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "review1.yaml"))
    parser.add_argument("--dev", action="store_true", help="Run in dev/smoke mode")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"=== M3 Training ===")
    print(f"Config: {args.config}")
    print(f"Dev mode: {args.dev}")
    print()

    result = run_m3(cfg, dev_mode=args.dev)

    print(f"\n=== Results ===")
    print(f"Train: {result['train_records']}, Val: {result['val_records']}, Test: {result['test_records']}")
    print(f"Best val loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
