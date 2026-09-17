"""Merge all partition checkpoints into final M1 dataset.

Usage:
    python scripts/merge_partitions.py --num-partitions 4

Run this after ALL partitions have completed on all notebooks.
"""
import argparse
import sys

sys.path.insert(0, "src")

from m1_data_integration.pipeline import merge_partitions
from m1_data_integration.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Merge M1 partition outputs")
    parser.add_argument("--num-partitions", type=int, default=4, help="Total number of partitions")
    parser.add_argument("--config", type=str, default="configs/review1.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Merging {args.num_partitions} partitions...")
    result = merge_partitions(cfg, num_partitions=args.num_partitions)
    print(f"Merge complete: {result.statistics['final_record_count']} records")
    print(f"Train: {result.statistics['split']['train_count']}")
    print(f"Validation: {result.statistics['split']['validation_count']}")
    print(f"Test: {result.statistics['split']['test_count']}")
    print(f"Leakage check: {result.statistics['split_leakage']}")
    print(f"Reconciliation: {result.statistics['reconciliation']['checks_passed']}")


if __name__ == "__main__":
    main()
