#!/usr/bin/env python
"""Single entry point: runs M1 -> M2 -> validation -> artifact generation.

Usage:
    python scripts/run_review1.py [--config configs/review1.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config  # noqa: E402
from m1_data_integration.pipeline import run_m1, save_m1_outputs  # noqa: E402
from m2_label_generation.pipeline import run_m2, save_m2_outputs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_review1")


def _stage_logger(module_name: str):
    def _cb(stage: str, status: str, payload):
        logger.info("[%s] %s -> %s", module_name, stage, status.upper())
    return _cb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Review 1 (M1 + M2) pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "review1.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("=== Running M1: Multi-Domain Privacy & Evidence Integration ===")
    m1_result = run_m1(cfg, on_stage=_stage_logger("M1"))
    save_m1_outputs(m1_result, cfg)
    logger.info("M1 complete: %d unified records produced", len(m1_result.records))

    logger.info("=== Running M2: Snorkel-Based Privacy Label Synthesis ===")
    m2_result = run_m2(m1_result.records, cfg, on_stage=_stage_logger("M2"))
    record_ids = [r.record_id for r in m1_result.records]
    save_m2_outputs(m2_result, cfg, record_ids)
    logger.info("M2 complete: weak labels produced for %d dimensions", len(m2_result.dimension_results))

    logger.info("=== Summary ===")
    for dim, diag in m2_result.diagnostics.items():
        logger.info("  %-18s n=%-5d mean_entropy=%.3f mean_confidence=%.3f uncertain=%d",
                    dim, diag.get("n_records", 0), diag.get("mean_entropy", 0.0),
                    diag.get("mean_max_confidence", 0.0), diag.get("n_uncertain_records", 0))

    logger.info("Outputs written under: %s", cfg.resolve_output("m1_dataset").parent)


if __name__ == "__main__":
    main()
