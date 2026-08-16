"""SynthesizeWeakLabels(D, E_A, LF, K) -- ties together LF matrix
construction and generative-model posterior inference to produce
L_w in [0,1]^(n x K) for every privacy dimension.

`entity_tags` and `threat_content` are both multi-label: each category
within the dimension is run as an independent binary (K=2)
SynthesizeWeakLabels instance, and the resulting P(category=1) columns
are stacked into one (n x |labels|) matrix for that dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from m1_data_integration.schemas import UnifiedRecord

from .generative_model import GenerativeModelResult, fit_and_infer
from .labeling_functions import DIMENSION_LFS, ENTITY_TAG_LFS, THREAT_CONTENT_LFS
from .lf_engine import LFMatrixResult, build_lf_matrix
from .taxonomy import DimensionSpec


@dataclass
class DimensionWeakLabelResult:
    dimension: str
    lf_matrix_result: LFMatrixResult
    generative_result: GenerativeModelResult
    weak_labels: np.ndarray  # L_w, shape (n, K)
    label_names: List[str]


def synthesize_weak_labels_for_dimension(
    records: List[UnifiedRecord], spec: DimensionSpec, seed: int,
) -> DimensionWeakLabelResult:
    """Unified path for multi-class and multi-label dimensions."""
    if spec.is_multi_label:
        return _synthesize_multi_label(records, spec, seed)
    return _synthesize_multi_class(records, spec, seed)


def _synthesize_multi_class(
    records: List[UnifiedRecord], spec: DimensionSpec, seed: int,
) -> DimensionWeakLabelResult:
    lfs = DIMENSION_LFS[spec.name]
    lf_result = build_lf_matrix(records, lfs, spec)
    gen_result = fit_and_infer(lf_result.matrix, spec.num_classes, seed=seed)
    return DimensionWeakLabelResult(
        dimension=spec.name,
        lf_matrix_result=lf_result,
        generative_result=gen_result,
        weak_labels=gen_result.probs,
        label_names=spec.labels,
    )


def _synthesize_multi_label(
    records: List[UnifiedRecord], spec: DimensionSpec, seed: int,
) -> DimensionWeakLabelResult:
    n = len(records)
    category_probs = np.zeros((n, len(spec.labels)))
    combined_lf_names: List[str] = []
    combined_matrices = []
    coverage: Dict[str, float] = {}
    abstention: Dict[str, float] = {}
    conflict_ratios = []
    lf_accuracies = []

    lfs_dict = THREAT_CONTENT_LFS if spec.name == "threat_content" else ENTITY_TAG_LFS

    for k, category in enumerate(spec.labels):
        lfs = lfs_dict[category]
        binary_spec = DimensionSpec(name=f"{spec.name}_{category}", labels=["absent", "present"], is_multi_label=False)
        lf_result = build_lf_matrix(records, lfs, binary_spec)
        gen_result = fit_and_infer(lf_result.matrix, num_classes=2, seed=seed)
        category_probs[:, k] = gen_result.probs[:, 1] if gen_result.probs.shape[0] else 0.0

        combined_lf_names.extend([f"{category}::{name}" for name in lf_result.lf_names])
        combined_matrices.append(lf_result.matrix)
        coverage.update({f"{category}::{k2}": v for k2, v in lf_result.coverage.items()})
        abstention.update({f"{category}::{k2}": v for k2, v in lf_result.abstention_stats.items()})
        conflict_ratios.append(lf_result.conflict_stats.get("conflict_ratio", 0.0))
        lf_accuracies.append(gen_result.lf_accuracy)

    stacked_matrix = np.concatenate(combined_matrices, axis=1) if combined_matrices else np.zeros((n, 0))
    combined_lf_result = LFMatrixResult(
        dimension=spec.name,
        lf_names=combined_lf_names,
        matrix=stacked_matrix,
        coverage=coverage,
        abstention_stats=abstention,
        conflict_stats={"conflict_ratio": float(np.mean(conflict_ratios)) if conflict_ratios else 0.0},
    )
    combined_gen_result = GenerativeModelResult(
        method="snorkel_label_model_per_category",
        class_priors=category_probs.mean(axis=0) if n else np.zeros(len(spec.labels)),
        lf_accuracy=np.concatenate(lf_accuracies) if lf_accuracies else np.zeros(0),
        probs=category_probs,
    )
    return DimensionWeakLabelResult(
        dimension=spec.name,
        lf_matrix_result=combined_lf_result,
        generative_result=combined_gen_result,
        weak_labels=category_probs,
        label_names=spec.labels,
    )


def synthesize_all_dimensions(
    records: List[UnifiedRecord], taxonomy: Dict[str, DimensionSpec], seed: int,
) -> Dict[str, DimensionWeakLabelResult]:
    results: Dict[str, DimensionWeakLabelResult] = {}
    for dim_name, spec in taxonomy.items():
        results[dim_name] = synthesize_weak_labels_for_dimension(records, spec, seed)
    return results