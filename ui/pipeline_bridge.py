"""Bridge between the Streamlit demo UI and the M1/M2 computational modules.

This file contains NO privacy/labeling logic of its own. Every stage
below calls the exact same functions used by `scripts/run_review1.py`
(the CLI/notebook entry point); it only decides *which raw records*
to feed in based on the user's UI selections (one dataset + optional
manual text / uploaded rows) and packages per-stage snapshots for
visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.cleaner import clean_record  # noqa: E402
from m1_data_integration.config import ReviewConfig, load_config  # noqa: E402
from m1_data_integration.deduplicator import deduplicate  # noqa: E402
from m1_data_integration.embeddings import build_embeddings_all  # noqa: E402
from m1_data_integration.evidence import build_evidence_all  # noqa: E402
from m1_data_integration.harmonizer import harmonize_record  # noqa: E402
from m1_data_integration.loaders import LOADER_REGISTRY, LocalDatasetCache  # noqa: E402
from m1_data_integration.schemas import RawRecord, UnifiedRecord  # noqa: E402
from m2_label_generation.diagnostics import full_diagnostics  # noqa: E402
from m2_label_generation.taxonomy import build_taxonomy  # noqa: E402
from m2_label_generation.weak_labels import synthesize_all_dimensions  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "review1.yaml"


def get_config() -> ReviewConfig:
    return load_config(CONFIG_PATH)


def build_raw_records(cfg: ReviewConfig, dataset_key: str, n_samples: int,
                       manual_text: Optional[str] = None,
                       uploaded_rows: Optional[List[str]] = None) -> List[RawRecord]:
    """Load n_samples raw records for the selected dataset via the *same*
    loader classes used by the CLI pipeline, optionally appended with a
    manually entered query and/or uploaded rows as ad-hoc 'custom' records.
    """
    loader_cls = LOADER_REGISTRY[dataset_key]
    dataset_cfg = cfg.datasets[dataset_key]
    cache = LocalDatasetCache.from_config(cfg)  # same local cache used by the CLI pipeline
    loader = loader_cls(dataset_cfg, max(n_samples, 0), cfg.seed, cache)
    records = loader.load() if n_samples > 0 else []

    if manual_text:
        records.append(RawRecord(
            source_dataset="manual_entry", domain="custom",
            query_text=manual_text, context_text="",
            metadata={"source_mode": "manual_entry"},
        ))
    if uploaded_rows:
        for i, row_text in enumerate(uploaded_rows):
            records.append(RawRecord(
                source_dataset="uploaded_file", domain="custom",
                query_text=row_text, context_text="",
                metadata={"source_mode": "uploaded_file", "row_index": i},
            ))
    return records


def run_m1_for_ui(cfg: ReviewConfig, dataset_key: str, n_samples: int,
                   manual_text: Optional[str] = None,
                   uploaded_rows: Optional[List[str]] = None,
                   on_stage=None) -> Dict[str, Any]:
    """Runs the M1 flow using the exact M1 functions, on a UI-selected subset
    of records. Returns kept records plus per-record, per-stage snapshots.
    """
    def emit(stage, status, payload=None):
        if on_stage:
            on_stage(stage, status, payload)

    emit("dataset_loading", "running")
    raw_records = build_raw_records(cfg, dataset_key, n_samples, manual_text, uploaded_rows)
    emit("dataset_loading", "completed", {"n_raw": len(raw_records)})

    emit("schema_harmonization", "running")
    unified: List[UnifiedRecord] = [
        harmonize_record(raw, dataset_key, idx) for idx, raw in enumerate(raw_records)
    ]
    emit("schema_harmonization", "completed", {"n_unified": len(unified)})

    emit("cleaning_normalization", "running")
    for r in unified:
        clean_record(r, cfg.preprocessing)
    emit("cleaning_normalization", "completed", {"n_cleaned": len(unified)})

    emit("domain_assignment", "running")
    domain_dist: Dict[str, int] = {}
    for r in unified:
        domain_dist[r.domain] = domain_dist.get(r.domain, 0) + 1
    emit("domain_assignment", "completed", {"domain_distribution": domain_dist})

    emit("deduplication", "running")
    kept, dedup_stats = deduplicate(unified, cfg.preprocessing.get("lowercase_for_dedup", True))
    emit("deduplication", "completed", dedup_stats)

    emit("semantic_evidence_extraction", "running")
    evidence_cfg = cfg.evidence_cfg
    build_evidence_all(kept, evidence_cfg.get("spacy_model", "en_core_web_sm"),
                        evidence_cfg.get("regex_patterns", []))
    emb_cfg = cfg.embeddings_cfg
    build_embeddings_all(kept, emb_cfg.get("model_name"), emb_cfg.get("batch_size", 16),
                          emb_cfg.get("device", "cpu"))
    emit("semantic_evidence_extraction", "completed", {"n_with_evidence": len(kept)})

    return {
        "all_unified_records": unified,   # includes duplicates, for transparency
        "kept_records": kept,             # what flows into M2
        "dedup_stats": dedup_stats,
        "domain_distribution": domain_dist,
    }


def run_m2_for_ui(cfg: ReviewConfig, records: List[UnifiedRecord], on_stage=None) -> Dict[str, Any]:
    """Runs the M2 flow using the exact M2 functions on `records`."""
    def emit(stage, status, payload=None):
        if on_stage:
            on_stage(stage, status, payload)

    emit("privacy_dimension_setup", "running")
    taxonomy = build_taxonomy(cfg.label_taxonomy)
    emit("privacy_dimension_setup", "completed", {"dimensions": list(taxonomy.keys())})

    emit("labeling_functions", "running")
    from m2_label_generation.labeling_functions import DIMENSION_LFS, ENTITY_TAG_LFS, THREAT_CONTENT_LFS
    lf_counts = {dim: len(lfs) for dim, lfs in DIMENSION_LFS.items()}
    lf_counts.update({f"entity_tags::{c}": len(lfs) for c, lfs in ENTITY_TAG_LFS.items()})
    lf_counts.update({f"threat_content::{c}": len(lfs) for c, lfs in THREAT_CONTENT_LFS.items()})
    emit("labeling_functions", "completed", {"lf_counts": lf_counts})

    emit("lf_matrix_construction", "running")
    dimension_results = synthesize_all_dimensions(records, taxonomy, cfg.seed)
    emit("lf_matrix_construction", "completed", {
        dim: list(res.lf_matrix_result.matrix.shape) for dim, res in dimension_results.items()
    })

    emit("generative_model_fitting", "running")
    emit("generative_model_fitting", "completed", {
        dim: res.generative_result.method for dim, res in dimension_results.items()
    })

    emit("posterior_inference", "running")
    emit("posterior_inference", "completed", {
        dim: list(res.weak_labels.shape) for dim, res in dimension_results.items()
    })

    diagnostics = full_diagnostics(dimension_results)
    return {"dimension_results": dimension_results, "diagnostics": diagnostics, "taxonomy": taxonomy}


def record_inspection_view(record: UnifiedRecord, index_in_kept: Optional[int],
                            dimension_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble everything known about one record for the inspection panel."""
    view: Dict[str, Any] = {
        "record_id": record.record_id,
        "domain": record.domain,
        "source_dataset": record.source_dataset,
        "query_text": record.query_text,
        "context_text": record.context_text,
        "normalized_text": record.normalized_text,
        "is_duplicate": record.is_duplicate,
        "metadata": record.metadata,
        "evidence": record.evidence.to_dict() if record.evidence else None,
    }
    if dimension_results is not None and index_in_kept is not None:
        view["weak_labels"] = {}
        view["lf_votes"] = {}
        for dim, res in dimension_results.items():
            probs = res.weak_labels[index_in_kept].tolist()
            view["weak_labels"][dim] = dict(zip(res.label_names, probs))
            votes = res.lf_matrix_result.matrix[index_in_kept].tolist()
            view["lf_votes"][dim] = dict(zip(res.lf_matrix_result.lf_names, votes))
    return view


# ---------------------------------------------------------------------------
# Small-batch, stage-by-stage BEFORE/AFTER trace.
#
# Unlike run_m1_for_ui/run_m2_for_ui (which run each module end-to-end and
# return only the final artifacts), this walks the exact same underlying
# functions ONE STAGE AT A TIME and snapshots every record in a small batch
# both immediately before and immediately after each stage runs. Nothing
# here reimplements M1/M2 logic -- it only adds "copy the record, then call
# the real function, then copy again" around each existing stage function.
# ---------------------------------------------------------------------------
import copy  # noqa: E402


def _raw_snapshot(raw: RawRecord) -> Dict[str, Any]:
    return {
        "source_dataset": raw.source_dataset,
        "domain": raw.domain,
        "query_text": raw.query_text,
        "context_text": raw.context_text,
        "metadata": dict(raw.metadata),
    }


def _unified_snapshot(rec: UnifiedRecord, fields: List[str]) -> Dict[str, Any]:
    """Snapshot only the requested fields of a UnifiedRecord (keeps each
    stage's before/after panel focused on what that stage actually changes,
    instead of dumping the whole record every time).
    """
    snap: Dict[str, Any] = {"record_id": rec.record_id}
    for f in fields:
        if f == "evidence":
            snap["evidence"] = rec.evidence.to_dict() if rec.evidence else None
        else:
            snap[f] = getattr(rec, f)
    return snap


def run_batch_trace_for_ui(cfg: ReviewConfig, dataset_key: str, batch_size: int,
                            manual_text: Optional[str] = None,
                            uploaded_rows: Optional[List[str]] = None,
                            on_stage=None) -> Dict[str, Any]:
    """Run M1 -> M2 one stage at a time on a small batch, capturing a
    BEFORE and AFTER snapshot of every record at every stage.

    Returns {"stages": [ {name, before:[...], after:[...] }, ... ],
             "final_records": [...], "dimension_results": {...}}
    so the UI can render one stage block at a time, each with a
    before/after comparison for every record in the batch.
    """
    def emit(stage, status, payload=None):
        if on_stage:
            on_stage(stage, status, payload)

    stages: List[Dict[str, Any]] = []

    # ---- Stage 0: Raw input -------------------------------------------------
    emit("raw_input", "running")
    # Reserve room in the batch for manual/uploaded rows so they are never
    # silently truncated away by the batch-size cap below.
    n_extra = (1 if manual_text else 0) + len(uploaded_rows or [])
    n_from_loader = max(batch_size - n_extra, 0)
    raw_records = build_raw_records(cfg, dataset_key, n_from_loader, manual_text, uploaded_rows)
    raw_records = raw_records[:batch_size]
    stages.append({
        "name": "Raw Input",
        "before": [None for _ in raw_records],
        "after": [_raw_snapshot(r) for r in raw_records],
    })
    emit("raw_input", "completed", {"n": len(raw_records)})

    # ---- Stage 1: Schema harmonization --------------------------------------
    emit("schema_harmonization", "running")
    before = [_raw_snapshot(r) for r in raw_records]
    unified: List[UnifiedRecord] = [
        harmonize_record(raw, dataset_key, idx) for idx, raw in enumerate(raw_records)
    ]
    after = [_unified_snapshot(r, ["record_id", "domain", "source_dataset",
                                    "query_text", "context_text"]) for r in unified]
    stages.append({"name": "Harmonize Schema (raw -> unified record)", "before": before, "after": after})
    emit("schema_harmonization", "completed", {"n": len(unified)})

    # ---- Stage 2: Cleaning / normalization ----------------------------------
    emit("cleaning_normalization", "running")
    before = [_unified_snapshot(r, ["query_text", "context_text"]) for r in unified]
    for r in unified:
        clean_record(r, cfg.preprocessing)
    after = [_unified_snapshot(r, ["normalized_text"]) for r in unified]
    stages.append({"name": "Clean / Normalize Text", "before": before, "after": after})
    emit("cleaning_normalization", "completed", {"n": len(unified)})

    # ---- Stage 3: Deduplication ----------------------------------------------
    emit("deduplication", "running")
    before = [_unified_snapshot(r, ["normalized_text"]) | {"is_duplicate": False} for r in unified]
    kept, dedup_stats = deduplicate(unified, cfg.preprocessing.get("lowercase_for_dedup", True))
    kept_ids = {r.record_id for r in kept}
    after = [
        _unified_snapshot(r, []) | {
            "is_duplicate": r.record_id not in kept_ids,
            "status": "kept -> continues to evidence extraction" if r.record_id in kept_ids
                      else "REMOVED (duplicate of an earlier record in this batch)",
        }
        for r in unified
    ]
    stages.append({"name": "Deduplicate Records", "before": before, "after": after,
                    "stats": dedup_stats})
    emit("deduplication", "completed", dedup_stats)

    # ---- Stage 4: Semantic evidence extraction -------------------------------
    emit("semantic_evidence_extraction", "running")
    before = [_unified_snapshot(r, ["normalized_text"]) | {"evidence": None} for r in kept]
    evidence_cfg = cfg.evidence_cfg
    build_evidence_all(kept, evidence_cfg.get("spacy_model", "en_core_web_sm"),
                        evidence_cfg.get("regex_patterns", []))
    emb_cfg = cfg.embeddings_cfg
    build_embeddings_all(kept, emb_cfg.get("model_name"), emb_cfg.get("batch_size", 16),
                          emb_cfg.get("device", "cpu"))
    after = []
    for r in kept:
        ev = r.evidence.to_dict() if r.evidence else {}
        after.append({
            "record_id": r.record_id,
            "entities (eta)": ev.get("entities", []),
            "regex_matches (rho)": ev.get("regex_matches", {}),
            "dependency_relations (delta)": ev.get("dependency_relations", [])[:5],
            "embedding_dim (epsilon)": len(ev.get("embedding") or []),
        })
    stages.append({"name": "Extract Semantic Evidence", "before": before, "after": after})
    emit("semantic_evidence_extraction", "completed", {"n": len(kept)})

    # ---- Stage 5: Apply labeling functions + fit generative model -----------
    emit("labeling_and_inference", "running")
    before = [{"record_id": r.record_id, "evidence_only": True} for r in kept]
    taxonomy = build_taxonomy(cfg.label_taxonomy)
    dimension_results = synthesize_all_dimensions(kept, taxonomy, cfg.seed) if kept else {}
    after = []
    for i, r in enumerate(kept):
        row: Dict[str, Any] = {"record_id": r.record_id}
        for dim, res in dimension_results.items():
            votes = dict(zip(res.lf_matrix_result.lf_names, res.lf_matrix_result.matrix[i].tolist()))
            fired = {k: v for k, v in votes.items() if v != -1}
            row[f"{dim} :: LF votes (non-abstain)"] = fired if fired else "all LFs abstained"
        after.append(row)
    stages.append({"name": "Apply Labeling Functions", "before": before, "after": after})
    emit("labeling_and_inference", "completed", {"n_dimensions": len(dimension_results)})

    # ---- Stage 6: Posterior inference (weak labels) --------------------------
    emit("posterior_inference", "running")
    before = [{"record_id": r.record_id, "L_w": "not yet inferred"} for r in kept]
    after = []
    for i, r in enumerate(kept):
        row = {"record_id": r.record_id}
        for dim, res in dimension_results.items():
            probs = res.weak_labels[i].tolist()
            row[dim] = {name: round(p, 3) for name, p in zip(res.label_names, probs)}
        after.append(row)
    stages.append({"name": "Infer Posterior Weak Labels (L_w)", "before": before, "after": after})
    emit("posterior_inference", "completed", {"n": len(kept)})

    return {"stages": stages, "final_records": kept, "dimension_results": dimension_results}