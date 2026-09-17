"""Shared pytest fixtures: small/fast config + M1 results for reuse across tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config  # noqa: E402
from m1_data_integration.pipeline import run_m1  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "review1.yaml"


@pytest.fixture(scope="session")
def small_cfg():
    cfg = load_config(CONFIG_PATH)
    # keep tests fast regardless of the checked-in config's sample sizes
    raw = cfg.raw.get("max_samples_per_dataset", {})
    for key in raw:
        raw[key] = 12
    return cfg


@pytest.fixture(scope="session")
def m1_result(small_cfg):
    return run_m1(small_cfg)
