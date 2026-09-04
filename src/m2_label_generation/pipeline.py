"""M2 orchestration: (D, E_A) -> Privacy Dimensions -> Labeling Functions
-> LF Matrix Lambda -> Snorkel Generative Model -> Posterior Inference -> L_w.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from m1_data_integration.config import ReviewConfig
from m1_data_integration.schemas import UnifiedRecord

from .diagnostics import full_diagnostics
from .taxonomy import build_taxonomy
from .weak_labels import DimensionWeakLabelResult, synthesize_all_dimensions

logger = logging.getLogger(__name__)

M2_STAGES = [
    "privacy_dimension_setup",
    "labeling_functions",
    "lf_matrix_construction",
    "generative_model_fitting",
    "posterior_inference",
]


@dataclass
class M2Result:
    dimension_results: Dict[str, DimensionWeakLabelResult]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stage_snapshots: Dict[str, Any] = field(default_factory=dict)


def run_m2(records: List[UnifiedRecord], cfg: ReviewConfig,
           on_stage: Optional[Callable] = None) -> M2Result:
    def emit(stage: str, status: str, payload: Any = None):
        if on_stage is not None:
            on_stage(stage, status, payload)

    snapshots: Dict[str, Any] = {}

    emit("privacy_dimension_setup", "running")
    taxonomy = build_taxonomy(cfg.label_taxonomy)
    snapshots["privacy_dimension_setup"] = {
        dim: {"labels": spec.labels, "num_classes": spec.num_classes, "multi_label": spec.is_multi_label}
        for dim, spec in taxonomy.items()
    }
    emit("privacy_dimension_setup", "completed", snapshots["privacy_dimension_setup"])

    emit("labeling_functions", "running")
    from .labeling_functions import DIMENSION_LFS, ENTITY_TAG_LFS, THREAT_CONTENT_LFS
    lf_counts = {dim: len(lfs) for dim, lfs in DIMENSION_LFS.items()}
    lf_counts.update({f"entity_tags::{cat}": len(lfs) for cat, lfs in ENTITY_TAG_LFS.items()})
    lf_counts.update({f"threat_content::{cat}": len(lfs) for cat, lfs in THREAT_CONTENT_LFS.items()})
    snapshots["labeling_functions"] = {"lf_counts_per_dimension": lf_counts}
    emit("labeling_functions", "completed", snapshots["labeling_functions"])

    # lf_matrix_construction, generative_model_fitting, posterior_inference all
    # happen inside synthesize_all_dimensions per-dimension; we emit around the
    # whole block since the sub-steps are tightly coupled per the spec's flow.
    emit("lf_matrix_construction", "running")
    dimension_results = synthesize_all_dimensions(records, taxonomy, cfg.seed)
    matrix_shapes = {dim: list(res.lf_matrix_result.matrix.shape) for dim, res in dimension_results.items()}
    snapshots["lf_matrix_construction"] = {"matrix_shapes": matrix_shapes}
    emit("lf_matrix_construction", "completed", snapshots["lf_matrix_construction"])

    emit("generative_model_fitting", "running")
    gen_methods = {dim: res.generative_result.method for dim, res in dimension_results.items()}
    snapshots["generative_model_fitting"] = {"backend_per_dimension": gen_methods}
    emit("generative_model_fitting", "completed", snapshots["generative_model_fitting"])

    emit("posterior_inference", "running")
    weak_label_shapes = {dim: list(res.weak_labels.shape) for dim, res in dimension_results.items()}
    sample_record_labels = None
    if records:
        sample_record_labels = {
            dim: res.weak_labels[0].tolist() for dim, res in dimension_results.items()
        }
    snapshots["posterior_inference"] = {
        "weak_label_shapes": weak_label_shapes,
        "sample_record_posterior": sample_record_labels,
    }
    emit("posterior_inference", "completed", snapshots["posterior_inference"])

    diagnostics = full_diagnostics(dimension_results, records=records)
    return M2Result(dimension_results=dimension_results, diagnostics=diagnostics, stage_snapshots=snapshots)


def save_m2_outputs(result: M2Result, cfg: ReviewConfig, record_ids: List[str]) -> None:
    lf_dir = cfg.resolve_output("m2_lf_matrix_dir")
    lf_dir.mkdir(parents=True, exist_ok=True)
    weak_dir = cfg.resolve_output("m2_weak_labels_dir")
    weak_dir.mkdir(parents=True, exist_ok=True)

    for dim, res in result.dimension_results.items():
        np.save(lf_dir / f"{dim}_lambda_matrix.npy", res.lf_matrix_result.matrix)
        np.save(weak_dir / f"{dim}_weak_labels.npy", res.weak_labels)

        with open(weak_dir / f"{dim}_weak_labels.jsonl", "w", encoding="utf-8") as fh:
            for rid, row in zip(record_ids, res.weak_labels):
                fh.write(json.dumps({
                    "record_id": rid,
                    "dimension": dim,
                    "label_names": res.label_names,
                    "probabilities": row.tolist(),
                }) + "\n")

    diag_path = cfg.resolve_output("m2_diagnostics")
    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(result.diagnostics, fh, indent=2, default=str)

    logger.info("M2 outputs saved under %s and %s", lf_dir, weak_dir)