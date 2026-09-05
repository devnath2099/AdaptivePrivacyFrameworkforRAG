"""Common record schema shared by all four Review 1 datasets.

M2 (and downstream modules) must be able to consume any record without
dataset-specific branching, so every loader/harmonizer ultimately
produces a `UnifiedRecord`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RawRecord:
    """Minimal, loader-produced representation before harmonization."""

    source_dataset: str
    domain: str
    query_text: str
    context_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """Semantic evidence E_A = (eta, delta, rho, epsilon) for one record."""

    entities: List[Dict[str, str]] = field(default_factory=list)          # eta
    dependency_relations: List[Dict[str, str]] = field(default_factory=list)  # delta
    regex_matches: Dict[str, List[str]] = field(default_factory=dict)     # rho
    embedding: Optional[List[float]] = None                               # epsilon

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedRecord:
    """The common schema D that M2 operates on, D = {d_1, ..., d_n}."""

    record_id: str
    domain: str
    source_dataset: str
    query_text: str
    context_text: str
    normalized_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: Optional[EvidenceBundle] = None
    is_duplicate: bool = False
    split: Optional[str] = None  # train/validation/test

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
