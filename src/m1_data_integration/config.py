"""Configuration loading for Review 1 (M1 + M2).

A single YAML file drives dataset locations, sample limits, evidence
extraction settings, embedding model choice and the label taxonomy so
that none of these are hard-coded across the source tree.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass
class ReviewConfig:
    """Typed accessor around the raw YAML config dictionary."""

    raw: Dict[str, Any]
    path: Path

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def datasets(self) -> Dict[str, Any]:
        return self.raw["datasets"]

    @property
    def max_samples(self) -> Dict[str, Optional[int]]:
        raw = self.raw["max_samples_per_dataset"]
        return {k: None if v is None else int(v) for k, v in raw.items()}

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def evidence_cfg(self) -> Dict[str, Any]:
        return self.raw.get("evidence", {})

    @property
    def embeddings_cfg(self) -> Dict[str, Any]:
        return self.raw.get("embeddings", {})

    @property
    def label_taxonomy(self) -> Dict[str, List[str]]:
        return self.raw["label_taxonomy"]

    @property
    def output_paths(self) -> Dict[str, str]:
        return self.raw["output_paths"]

    @property
    def data_cache(self) -> Dict[str, Any]:
        return self.raw.get("data_cache", {})

    @property
    def sampling_strategy(self) -> str:
        return self.raw.get("sampling_strategy", "natural")

    @property
    def split_cfg(self) -> Dict[str, Any]:
        return self.raw.get("split", {
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
            "stratify_by_domain": True,
        })

    @property
    def mode(self) -> str:
        """Determine execution mode: 'development' if any max_samples limit is set, else 'full'."""
        max_samples = self.max_samples
        if max_samples and any(v is not None for v in max_samples.values()):
            return "development"
        return "full"

    @property
    def local_datasets(self) -> Dict[str, str]:
        return self.data_cache.get("local_datasets", {})

    def resolve_output(self, key: str) -> Path:
        """Resolve an output path relative to the config file's project root."""
        project_root = self.path.parent.parent
        rel = self.output_paths[key]
        out = project_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out


def load_config(path: str | Path) -> ReviewConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = ReviewConfig(raw=raw, path=path)
    seed_everything(cfg.seed)
    return cfg


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across libraries in use."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)