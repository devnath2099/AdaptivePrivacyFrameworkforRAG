"""Constructs the labeling-function matrix Lambda and computes LF diagnostics.

Lambda in ({0,...,K-1} union {ABSTAIN})^(n x m) for a single dimension,
as specified in the SynthesizeWeakLabels algorithm (step 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from m1_data_integration.schemas import UnifiedRecord

from .labeling_functions import LabelingFunction
from .taxonomy import ABSTAIN, DimensionSpec


@dataclass
class LFMatrixResult:
    dimension: str
    lf_names: List[str]
    matrix: np.ndarray  # shape (n_records, n_lfs), values in {ABSTAIN, 0..K-1}
    coverage: Dict[str, float] = field(default_factory=dict)
    conflict_stats: Dict[str, float] = field(default_factory=dict)
    abstention_stats: Dict[str, float] = field(default_factory=dict)


def build_lf_matrix(
    records: List[UnifiedRecord],
    lfs: List[LabelingFunction],
    spec: DimensionSpec,
) -> LFMatrixResult:
    n, m = len(records), len(lfs)
    matrix = np.full((n, m), ABSTAIN, dtype=int)

    for j, lf in enumerate(lfs):
        for i, record in enumerate(records):
            try:
                vote = lf(record, spec)
            except Exception:  # noqa: BLE001 - a single LF failure must not break the run
                vote = ABSTAIN
            if vote != ABSTAIN and not (0 <= vote < spec.num_classes):
                vote = ABSTAIN  # guard: keep LF outputs inside the valid label space
            matrix[i, j] = vote

    lf_names = [getattr(lf, "__name__", f"lf_{j}") for j, lf in enumerate(lfs)]
    result = LFMatrixResult(dimension=spec.name, lf_names=lf_names, matrix=matrix)
    result.coverage = _compute_coverage(matrix, lf_names)
    result.abstention_stats = _compute_abstention(matrix, lf_names)
    result.conflict_stats = _compute_conflicts(matrix)
    return result


def _compute_coverage(matrix: np.ndarray, lf_names: List[str]) -> Dict[str, float]:
    n = matrix.shape[0]
    if n == 0:
        return {name: 0.0 for name in lf_names}
    return {
        lf_names[j]: float(np.mean(matrix[:, j] != ABSTAIN))
        for j in range(matrix.shape[1])
    }


def _compute_abstention(matrix: np.ndarray, lf_names: List[str]) -> Dict[str, float]:
    n = matrix.shape[0]
    if n == 0:
        return {name: 1.0 for name in lf_names}
    per_lf = {
        lf_names[j]: float(np.mean(matrix[:, j] == ABSTAIN))
        for j in range(matrix.shape[1])
    }
    per_lf["overall_row_all_abstain_ratio"] = float(
        np.mean(np.all(matrix == ABSTAIN, axis=1))
    ) if matrix.size else 1.0
    return per_lf


def _compute_conflicts(matrix: np.ndarray) -> Dict[str, float]:
    """A record has a 'conflict' if at least two non-abstaining LFs disagree."""
    n = matrix.shape[0]
    if n == 0:
        return {"conflict_ratio": 0.0}
    conflict_rows = 0
    for row in matrix:
        active = row[row != ABSTAIN]
        if active.size >= 2 and len(set(active.tolist())) > 1:
            conflict_rows += 1
    return {"conflict_ratio": conflict_rows / n, "n_conflicting_records": conflict_rows}


def _compute_conflicts_multi_label(matrix: np.ndarray) -> Dict[str, float]:
    """Computes conflict ratio within a single binary category matrix."""
    n = matrix.shape[0]
    if n == 0:
        return {"conflict_ratio": 0.0}
    conflict_rows = 0
    for row in matrix:
        active = row[row != ABSTAIN]
        if active.size >= 2 and len(set(active.tolist())) > 1:
            conflict_rows += 1
    return {"conflict_ratio": conflict_rows / n, "n_conflicting_records": conflict_rows}