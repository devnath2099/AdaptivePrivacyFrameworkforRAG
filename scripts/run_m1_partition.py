"""Run M1 on a single partition for multi-notebook parallel processing.

Usage:
    python scripts/run_m1_partition.py --partition-id 0 --num-partitions 4
    python scripts/run_m1_partition.py --partition-id 1 --num-partitions 4
    python scripts/run_m1_partition.py --partition-id 2 --num-partitions 4
    python scripts/run_m1_partition.py --partition-id 3 --num-partitions 4
"""
import argparse
import sys

sys.path.insert(0, "src")

from m1_data_integration.pipeline import run_m1
from m1_data_integration.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run M1 on a partition")
    parser.add_argument("--partition-id", type=int, required=True, help="Partition index (0-based)")
    parser.add_argument("--num-partitions", type=int, default=4, help="Total number of partitions")
    parser.add_argument("--config", type=str, default="configs/review1.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Running M1 partition {args.partition_id}/{args.num_partitions}")
    result = run_m1(cfg, partition_id=args.partition_id, num_partitions=args.num_partitions)
    print(f"Partition {args.partition_id} complete: {len(result.records)} records")


if __name__ == "__main__":
    main()
