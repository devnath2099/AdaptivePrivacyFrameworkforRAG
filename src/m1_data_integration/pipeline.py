"""M1 orchestration: Datasets -> Integration -> Cleaning -> Domain Assignment
-> Deduplication -> Semantic Evidence Extraction -> (D, E_A).

`run_m1` returns a `M1Result` that also carries per-stage snapshots so
that a UI layer can visualize "Pending -> Running -> Completed" without
re-implementing any M1 logic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cleaner import clean_all
from .config import ReviewConfig
from .deduplicator import deduplicate
from .embeddings import build_embeddings_all
from .evidence import build_evidence_all
from .harmonizer import harmonize_all
from .loaders import load_all_datasets
from .schemas import UnifiedRecord

logger = logging.getLogger(__name__)

M1_STAGES = [
    "dataset_loading",
    "schema_harmonization",
    "cleaning_normalization",
    "domain_assignment",
    "deduplication",
    "semantic_evidence_extraction",
]


@dataclass
class M1Result:
    records: List[UnifiedRecord]
    statistics: Dict[str, Any] = field(default_factory=dict)
    stage_snapshots: Dict[str, Any] = field(default_factory=dict)

    def balanced_sample(self, seed: int = 42) -> List[UnifiedRecord]:
        return sample_balanced_by_domain(self.records, seed)


def _domain_distribution(records: List[UnifiedRecord]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for r in records:
        dist[r.domain] = dist.get(r.domain, 0) + 1
    return dist


def sample_balanced_by_domain(records: List[UnifiedRecord], seed: int = 42) -> List[UnifiedRecord]:
    """Return a balanced sample where each domain is represented proportionally.

    Aims for roughly equal representation per domain. If a domain has fewer
    records than the smallest domain, all its records are kept and others are
    down-sampled to match. If a domain has more, it is down-sampled.

    The result is seeded for reproducibility.
    """
    import random
    import numpy as np

    rng = random.Random(seed)

    # Group records by domain
    by_domain: Dict[str, List[UnifiedRecord]] = {}
    for r in records:
        by_domain.setdefault(r.domain, []).append(r)

    if not by_domain:
        return []

    # Find the smallest domain size
    sizes = {d: len(recs) for d, recs in by_domain.items()}
    target_size = min(sizes.values())

    # Sample each domain to target_size
    balanced: List[UnifiedRecord] = []
    for domain, recs in by_domain.items():
        if len(recs) <= target_size:
            balanced.extend(recs)
        else:
            shuffled = recs[:]
            rng.shuffle(shuffled)
            balanced.extend(shuffled[:target_size])

    return balanced


def run_m1(cfg: ReviewConfig, on_stage: Optional[callable] = None) -> M1Result:
    """Execute the full M1 flow.

    `on_stage(stage_name, status, payload)` is an optional callback used
    by the UI to render live Pending/Running/Completed status; it must
    never change pipeline behaviour.
    """
    def emit(stage: str, status: str, payload: Any = None):
        if on_stage is not None:
            on_stage(stage, status, payload)

    stats: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}

    # 1. dataset loading
    emit("dataset_loading", "running")
    raw_by_dataset = load_all_datasets(cfg)
    raw_counts = {k: len(v) for k, v in raw_by_dataset.items()}
    stats["raw_records_per_dataset"] = raw_counts
    snapshots["dataset_loading"] = {
        "raw_counts": raw_counts,
        "sample_raw_record": _sample_raw(raw_by_dataset),
    }
    emit("dataset_loading", "completed", snapshots["dataset_loading"])

    # 2. schema harmonization
    emit("schema_harmonization", "running")
    unified = harmonize_all(raw_by_dataset)
    snapshots["schema_harmonization"] = {
        "n_records": len(unified),
        "sample_record": unified[0].to_dict() if unified else None,
    }
    emit("schema_harmonization", "completed", snapshots["schema_harmonization"])

    # 3. cleaning / normalization
    emit("cleaning_normalization", "running")
    before_clean = len(unified)
    cleaned = clean_all(unified, cfg.preprocessing)
    stats["records_removed_by_cleaning"] = before_clean - len(cleaned)
    snapshots["cleaning_normalization"] = {
        "n_before": before_clean,
        "n_after": len(cleaned),
        "sample_record": cleaned[0].to_dict() if cleaned else None,
    }
    emit("cleaning_normalization", "completed", snapshots["cleaning_normalization"])

    # 4. domain assignment (already set at load time; verify/report here)
    emit("domain_assignment", "running")
    domain_dist = _domain_distribution(cleaned)
    stats["domain_distribution"] = domain_dist
    snapshots["domain_assignment"] = {"domain_distribution": domain_dist}
    emit("domain_assignment", "completed", snapshots["domain_assignment"])

    # 5. deduplication
    emit("deduplication", "running")
    deduped, dedup_stats = deduplicate(cleaned, cfg.preprocessing.get("lowercase_for_dedup", True))
    stats["deduplication"] = dedup_stats
    snapshots["deduplication"] = dedup_stats
    emit("deduplication", "completed", snapshots["deduplication"])

    # 6. semantic evidence extraction: eta, delta, rho, epsilon
    emit("semantic_evidence_extraction", "running")
    evidence_cfg = cfg.evidence_cfg
    build_evidence_all(deduped, evidence_cfg.get("spacy_model", "en_core_web_sm"),
                        evidence_cfg.get("regex_patterns", []))
    emb_cfg = cfg.embeddings_cfg
    build_embeddings_all(deduped, emb_cfg.get("model_name"), emb_cfg.get("batch_size", 16),
                          emb_cfg.get("device", "cpu"))

    entity_counts = sum(len(r.evidence.entities) for r in deduped if r.evidence)
    regex_hit_counts: Dict[str, int] = {}
    for r in deduped:
        if not r.evidence:
            continue
        for k, v in r.evidence.regex_matches.items():
            regex_hit_counts[k] = regex_hit_counts.get(k, 0) + len(v)
    embedding_dims = {len(r.evidence.embedding) for r in deduped if r.evidence and r.evidence.embedding}

    stats["evidence_statistics"] = {
        "total_entities_detected": entity_counts,
        "regex_hit_counts": regex_hit_counts,
        "embedding_dimensions_seen": list(embedding_dims),
    }
    snapshots["semantic_evidence_extraction"] = {
        "sample_evidence": deduped[0].evidence.to_dict() if deduped and deduped[0].evidence else None,
        **stats["evidence_statistics"],
    }
    emit("semantic_evidence_extraction", "completed", snapshots["semantic_evidence_extraction"])

    stats["final_record_count"] = len(deduped)
    return M1Result(records=deduped, statistics=stats, stage_snapshots=snapshots)


def _sample_raw(raw_by_dataset: Dict[str, List]) -> Optional[Dict[str, Any]]:
    for records in raw_by_dataset.values():
        if records:
            r = records[0]
            return {
                "source_dataset": r.source_dataset,
                "domain": r.domain,
                "query_text": r.query_text,
                "context_text": r.context_text,
                "metadata": r.metadata,
            }
    return None


def save_m1_outputs(result: M1Result, cfg: ReviewConfig) -> None:
    dataset_path = cfg.resolve_output("m1_dataset")
    with open(dataset_path, "w", encoding="utf-8") as fh:
        for r in result.records:
            fh.write(json.dumps(r.to_dict(), default=str) + "\n")

    stats_path = cfg.resolve_output("m1_stats")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(result.statistics, fh, indent=2, default=str)

    logger.info("M1 outputs saved: %s, %s", dataset_path, stats_path)
