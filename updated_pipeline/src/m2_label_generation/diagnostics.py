"""Research-style diagnostics for the weak-supervision outputs.

No gold labels are assumed to exist for Review 1, so only
weak-supervision diagnostics are reported (coverage, conflict,
abstention, entropy/confidence of L_w, class distribution) -- never an
"accuracy" figure, per the project's research-honesty rule.

Additional feature: per-domain breakdowns when records carry a `domain`
field (from M1), enabling domain-imbalance analysis.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .weak_labels import DimensionWeakLabelResult

_EPS = 1e-12


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _safe_mean(vals: np.ndarray) -> float:
    return float(np.mean(vals)) if vals.size else 0.0


def _class_distribution(predicted: np.ndarray, num_classes: int) -> Dict[str, int]:
    """Count occurrences per class label."""
    counts: Dict[int, int] = {}
    for k in predicted:
        counts[int(k)] = counts.get(int(k), 0) + 1
    return {f"class_{k}": counts.get(k, 0) for k in range(num_classes)}


_THRESHOLDS = (0.5, 0.6, 0.8, 0.9)
_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _per_category_stats(
    label_names: List[str], probs: np.ndarray
) -> Dict[str, Any]:
    """Compute per-category probability statistics for a multi-label dimension.

    Reports, for each independent category:
      - count / proportion above the thresholds {0.5, 0.6, 0.8, 0.9}
      - mean / median / std / min / max probability
      - quantiles at 1%, 5%, 25%, 50%, 75%, 95%, 99%
    """
    n = probs.shape[0]
    stats: Dict[str, Any] = {}
    for k, label in enumerate(label_names):
        col = probs[:, k].astype(float)
        if n == 0:
            col = np.zeros(0)
        entry: Dict[str, Any] = {}
        entry["mean"] = float(np.mean(col)) if n else 0.0
        entry["median"] = float(np.median(col)) if n else 0.0
        entry["std"] = float(np.std(col)) if n else 0.0
        entry["min"] = float(np.min(col)) if n else 0.0
        entry["max"] = float(np.max(col)) if n else 0.0
        entry["quantiles"] = {
            f"q{int(q * 100)}": float(np.quantile(col, q)) if n else 0.0
            for q in _QUANTILES
        }
        entry["above_threshold"] = {}
        for t in _THRESHOLDS:
            count = int(np.sum(col > t))
            entry["above_threshold"][f"gt_{t}"] = {
                "count": count,
                "proportion": float(count / n) if n else 0.0,
            }
        stats[label] = entry
    return stats


def _co_occurrence(probs: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    positive = probs >= threshold
    n_positives_per_row = positive.sum(axis=1).astype(int)
    n = probs.shape[0]
    return {
        "zero_positive_records": int(np.sum(n_positives_per_row == 0)),
        "exactly_one_positive_record": int(np.sum(n_positives_per_row == 1)),
        "two_positive_records": int(np.sum(n_positives_per_row == 2)),
        "three_or_more_positive_records": int(np.sum(n_positives_per_row >= 3)),
    }


def _build_label_summary(
    label_names: List[str],
    probs: np.ndarray,
    predicted: np.ndarray,
    is_multi_label: bool,
) -> Dict[str, Any]:
    """Build a label-distribution summary appropriate to the dimension type.

    - Single-label / multi-class: counts of the argmax class per label.
      Every record contributes to exactly one class, so the totals sum to n.

    - Multi-label (independent binary categories): the argmax across
      categories is MEANINGLESS. Instead report, per category, independent
      probability statistics (mean/median/std/min/max/quantiles, and the
      number/proportion of records above thresholds 0.5/0.6/0.8/0.9), plus
      co-occurrence counts (records positive for zero/one/two/many
      categories). This preserves the independence of the categories rather
      than collapsing them into a fake exclusive assignment.
    """
    n = probs.shape[0]
    if not is_multi_label:
        return {
            label_names[k]: int(np.sum(predicted == k)) for k in range(probs.shape[1])
        }

    per_category = _per_category_stats(label_names, probs)
    co_occurrence = _co_occurrence(probs, threshold=0.5)

    return {
        "per_category_statistics": per_category,
        "co_occurrence": co_occurrence,
    }


def dimension_diagnostics(
    result: DimensionWeakLabelResult,
    records: Optional[List] = None,
    uncertain_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Compute diagnostics for a dimension, with optional per-domain breakdown.

    If `records` (list of UnifiedRecord with .domain field) are provided,
    the returned dict includes a "per_domain" sub-dict.
    """
    probs = result.weak_labels
    n = probs.shape[0]
    if n == 0:
        return {
            "dimension": result.dimension,
            "n_records": 0,
            "per_domain": {},
            "generative_model_method": result.generative_result.method,
            "category_backends": getattr(result, "category_backends", None),
            "lf_coverage": result.lf_matrix_result.coverage,
            "lf_abstention": result.lf_matrix_result.abstention_stats,
            "lf_conflict": result.lf_matrix_result.conflict_stats,
            "label_distribution": {},
            "mean_entropy": 0.0,
            "mean_max_confidence": 0.0,
            "n_uncertain_records": 0,
            "uncertain_threshold": uncertain_threshold,
        }

    ent = _entropy(probs)
    max_prob = probs.max(axis=1)
    predicted = probs.argmax(axis=1)

    is_multi_label = getattr(result, "_is_multi_label", False)
    label_distribution = _build_label_summary(
        result.label_names, probs, predicted, is_multi_label
    )

    base: Dict[str, Any] = {
        "dimension": result.dimension,
        "n_records": n,
        "generative_model_method": result.generative_result.method,
        "category_backends": getattr(result, "category_backends", None),
        "lf_coverage": result.lf_matrix_result.coverage,
        "lf_abstention": result.lf_matrix_result.abstention_stats,
        "lf_conflict": result.lf_matrix_result.conflict_stats,
        "label_distribution": label_distribution,
        "mean_entropy": float(np.mean(ent)),
        "mean_max_confidence": float(np.mean(max_prob)),
        "n_uncertain_records": int(np.sum(max_prob < uncertain_threshold)),
        "uncertain_threshold": uncertain_threshold,
    }

    # Per-domain breakdown if records with .domain are provided
    if records is not None:
        domains = sorted(set(getattr(r, "domain", None) for r in records if getattr(r, "domain", None) is not None))
        per_domain: Dict[str, Dict[str, Any]] = {}
        for d in domains:
            # Select records belonging to this domain
            domain_records_indices = [i for i, r in enumerate(records) if r.domain == d]
            if not domain_records_indices:
                per_domain[d] = {
                    "n_records": 0,
                    "mean_entropy": 0.0,
                    "mean_max_confidence": 0.0,
                    "n_uncertain_records": 0,
                    "label_distribution": {},
                    "lf_coverage": {},
                    "lf_abstention": {},
                    "lf_conflict": {"conflict_ratio": 0.0},
                }
                continue

            d_probs = probs[domain_records_indices]
            d_ent = _entropy(d_probs)
            d_max_prob = d_probs.max(axis=1)
            d_predicted = d_probs.argmax(axis=1)

            d_label_dist = _build_label_summary(
                result.label_names, d_probs, d_predicted, is_multi_label
            )

            per_domain[d] = {
                "n_records": len(domain_records_indices),
                "mean_entropy": _safe_mean(d_ent),
                "mean_max_confidence": _safe_mean(d_max_prob),
                "n_uncertain_records": int(np.sum(d_max_prob < uncertain_threshold)),
                "label_distribution": d_label_dist,
                "lf_coverage": result.lf_matrix_result.coverage,  # overall LF coverage (same matrix)
                "lf_abstention": result.lf_matrix_result.abstention_stats,
                "lf_conflict": result.lf_matrix_result.conflict_stats,
            }
        base["per_domain"] = per_domain

    return base


def full_diagnostics(
    results: Dict[str, DimensionWeakLabelResult],
    records: Optional[List] = None,
) -> Dict[str, Any]:
    return {dim: dimension_diagnostics(res, records=records) for dim, res in results.items()}
