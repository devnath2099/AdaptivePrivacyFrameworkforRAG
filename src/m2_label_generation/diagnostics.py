"""Research-style diagnostics for the weak-supervision outputs.

No gold labels are assumed to exist for Review 1, so only
weak-supervision diagnostics are reported (coverage, conflict,
abstention, entropy/confidence of L_w, class distribution) -- never an
"accuracy" figure, per the project's research-honesty rule.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .weak_labels import DimensionWeakLabelResult

_EPS = 1e-12


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def dimension_diagnostics(result: DimensionWeakLabelResult, uncertain_threshold: float = 0.6) -> Dict[str, Any]:
    probs = result.weak_labels
    n = probs.shape[0]
    if n == 0:
        return {"dimension": result.dimension, "n_records": 0}

    ent = _entropy(probs)
    max_prob = probs.max(axis=1)
    predicted = probs.argmax(axis=1)

    label_distribution = {
        result.label_names[k]: int(np.sum(predicted == k)) for k in range(probs.shape[1])
    }

    return {
        "dimension": result.dimension,
        "n_records": n,
        "generative_model_method": result.generative_result.method,
        "lf_coverage": result.lf_matrix_result.coverage,
        "lf_abstention": result.lf_matrix_result.abstention_stats,
        "lf_conflict": result.lf_matrix_result.conflict_stats,
        "label_distribution": label_distribution,
        "mean_entropy": float(np.mean(ent)),
        "mean_max_confidence": float(np.mean(max_prob)),
        "n_uncertain_records": int(np.sum(max_prob < uncertain_threshold)),
        "uncertain_threshold": uncertain_threshold,
    }


def full_diagnostics(results: Dict[str, DimensionWeakLabelResult]) -> Dict[str, Any]:
    return {dim: dimension_diagnostics(res) for dim, res in results.items()}
