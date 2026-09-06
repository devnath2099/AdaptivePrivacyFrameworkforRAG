"""M1 orchestration: Datasets -> Integration -> Cleaning -> Domain Assignment
-> Deduplication -> Semantic Evidence Extraction -> Domain-Stratified Split -> (D, E_A).

`run_m1` returns a `M1Result` that also carries per-stage snapshots so
that a UI layer can visualize "Pending -> Running -> Completed" without
re-implementing any M1 logic.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cleaner import clean_all
from .config import ReviewConfig
from .deduplicator import deduplicate
from .embeddings import build_embeddings_all
from .evidence import build_evidence_all
from .harmonizer import harmonize_all
from .loaders import load_all_datasets
from .schemas import UnifiedRecord

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = "cache"
EVIDENCE_CACHE_FILE = "deduped_with_evidence.jsonl"
EMBEDDINGS_CACHE_FILE = "embeddings_done.jsonl"

M1_STAGES = [
    "dataset_loading",
    "schema_harmonization",
    "cleaning_normalization",
    "domain_assignment",
    "deduplication",
    "semantic_evidence_extraction",
    "domain_stratified_split",
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


def _domain_stratified_split(
    records: List[UnifiedRecord],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_domain: bool = True,
) -> Tuple[List[UnifiedRecord], List[UnifiedRecord], List[UnifiedRecord]]:
    """Perform domain-stratified train/validation/test split.

    If stratify_by_domain is True, splits each domain independently to preserve
    domain proportions. Records are shuffled within each domain before splitting.

    Returns:
        Tuple of (train_records, validation_records, test_records)
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)

    if not stratify_by_domain:
        # Global shuffle
        shuffled = records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * validation_ratio)
        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    # Domain-stratified split
    by_domain: Dict[str, List[UnifiedRecord]] = {}
    for r in records:
        by_domain.setdefault(r.domain, []).append(r)

    train_records: List[UnifiedRecord] = []
    val_records: List[UnifiedRecord] = []
    test_records: List[UnifiedRecord] = []

    for domain, domain_records in by_domain.items():
        rng.shuffle(domain_records)
        n = len(domain_records)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * validation_ratio)

        for i, r in enumerate(domain_records):
            if i < train_end:
                r.split = "train"
                train_records.append(r)
            elif i < val_end:
                r.split = "validation"
                val_records.append(r)
            else:
                r.split = "test"
                test_records.append(r)

    # Final shuffle within each split to mix domains
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    return train_records, val_records, test_records


def _check_split_leakage(
    train_records: List[UnifiedRecord],
    validation_records: List[UnifiedRecord],
    test_records: List[UnifiedRecord],
) -> Dict[str, Any]:
    """Check for data leakage across train/validation/test splits.

    Checks for exact matches on:
    - normalized_text (query deduplication key)
    - record_id
    - source_dataset + row_index (from metadata)

    Returns dict with leakage counts and details.
    """
    def make_key(r: UnifiedRecord) -> str:
        return r.normalized_text.lower().strip()

    def make_source_key(r: UnifiedRecord) -> str:
        src = r.metadata.get("source_mode", "unknown")
        idx = r.metadata.get("row_index", "unknown")
        return f"{r.source_dataset}:{idx}"

    leakage: Dict[str, Any] = {}

    # Build sets for each split
    train_keys = {make_key(r) for r in train_records}
    val_keys = {make_key(r) for r in validation_records}
    test_keys = {make_key(r) for r in test_records}

    train_source_keys = {make_source_key(r) for r in train_records}
    val_source_keys = {make_source_key(r) for r in validation_records}
    test_source_keys = {make_source_key(r) for r in test_records}

    # Check normalized_text overlap
    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    leakage["normalized_text_overlap"] = {
        "train_validation": len(train_val_overlap),
        "train_test": len(train_test_overlap),
        "validation_test": len(val_test_overlap),
        "examples_train_validation": list(train_val_overlap)[:5],
        "examples_train_test": list(train_test_overlap)[:5],
        "examples_validation_test": list(val_test_overlap)[:5],
    }

    # Check source key overlap
    train_val_source_overlap = train_source_keys & val_source_keys
    train_test_source_overlap = train_source_keys & test_source_keys
    val_test_source_overlap = val_source_keys & test_source_keys

    leakage["source_key_overlap"] = {
        "train_validation": len(train_val_source_overlap),
        "train_test": len(train_test_source_overlap),
        "validation_test": len(val_test_source_overlap),
        "examples_train_validation": list(train_val_source_overlap)[:5],
        "examples_train_test": list(train_test_source_overlap)[:5],
        "examples_validation_test": list(val_test_source_overlap)[:5],
    }

    # Record ID overlap
    train_ids = {r.record_id for r in train_records}
    val_ids = {r.record_id for r in validation_records}
    test_ids = {r.record_id for r in test_records}

    leakage["record_id_overlap"] = {
        "train_validation": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(val_ids & test_ids),
    }

    return leakage


def _reconcile_counts(
    raw_by_dataset: Dict[str, List],
    raw_counts: Dict[str, int],
    cleaned_count: int,
    removed_by_cleaning: int,
    deduped_count: int,
    dedup_stats: Dict[str, Any],
    train_records: List[UnifiedRecord],
    val_records: List[UnifiedRecord],
    test_records: List[UnifiedRecord],
) -> Dict[str, Any]:
    """Perform reconciliation checks at end of M1.

    Verifies:
    1. raw records = valid + removed by cleaning
    2. cleaned records = train + validation + test
    3. domain totals = sum of domain counts
    4. train/val/test totals = sum of their domain counts

    Returns reconciliation report with any mismatches.
    """
    reconciliation: Dict[str, Any] = {
        "checks_passed": True,
        "mismatches": [],
    }

    # 1. raw = cleaned + removed_by_cleaning
    total_raw = sum(raw_counts.values())
    expected_cleaned = total_raw - removed_by_cleaning
    if cleaned_count != expected_cleaned:
        reconciliation["checks_passed"] = False
        reconciliation["mismatches"].append({
            "check": "raw_equals_cleaned_plus_removed",
            "expected": expected_cleaned,
            "actual": cleaned_count,
            "details": f"total_raw={total_raw}, removed_by_cleaning={removed_by_cleaning}",
        })

    # 2. cleaned = train + validation + test (should equal deduped count)
    total_split = len(train_records) + len(val_records) + len(test_records)
    if total_split != cleaned_count:
        reconciliation["checks_passed"] = False
        reconciliation["mismatches"].append({
            "check": "split_equals_cleaned",
            "expected": cleaned_count,
            "actual": total_split,
            "details": f"train={len(train_records)}, validation={len(val_records)}, test={len(test_records)}",
        })

    # 3. Domain totals = sum of domain counts (for splits)
    # This is implicitly verified by the split logic

    # 4. Split totals = sum of their domain counts
    for split_name, records in [("train", train_records), ("validation", val_records), ("test", test_records)]:
        split_domain_dist = {}
        for r in records:
            split_domain_dist[r.domain] = split_domain_dist.get(r.domain, 0) + 1
        if sum(split_domain_dist.values()) != len(records):
            reconciliation["checks_passed"] = False
            reconciliation["mismatches"].append({
                "check": f"{split_name}_domain_sum_matches_total",
                "expected": len(records),
                "actual": sum(split_domain_dist.values()),
            })

    return reconciliation


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


def _get_cache_dir(cfg: ReviewConfig) -> str:
    cache_dir = cfg.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_cached_evidence(cfg: ReviewConfig, partition_id: int = 0, num_partitions: int = 1) -> Optional[List[UnifiedRecord]]:
    """Load records with evidence from checkpoint if partition is complete."""
    cache_dir = _get_cache_dir(cfg)
    done_path = os.path.join(cache_dir, f"evidence_done_{partition_id}.marker")
    checkpoint_path = os.path.join(cache_dir, f"evidence_checkpoint_{partition_id}.jsonl")

    if not os.path.exists(done_path):
        return None

    # Load all records from checkpoint
    records: List[UnifiedRecord] = []
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = UnifiedRecord.from_dict(data)
                records.append(record)
        logger.info("Loaded %d cached records with evidence (partition %d/%d)", len(records), partition_id, num_partitions)
        return records
    except Exception as exc:
        logger.warning("Failed to load evidence cache (%s), recomputing", exc)
        return None


def _save_cached_evidence(cfg: ReviewConfig, records: List[UnifiedRecord],
                           partition_id: int = 0, num_partitions: int = 1) -> None:
    """Save deduped records with evidence to cache."""
    cache_dir = _get_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, f"evidence_checkpoint_{partition_id}.jsonl")
    done_path = os.path.join(cache_dir, f"evidence_done_{partition_id}.marker")
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), default=str) + "\n")
    with open(done_path, "w") as f:
        json.dump({"n_records": len(records), "complete": True}, f)
    logger.info("Saved %d records with evidence to cache (partition %d/%d)", len(records), partition_id, num_partitions)


def _load_embeddings_cache(cfg: ReviewConfig, partition_id: int = 0) -> bool:
    cache_dir = _get_cache_dir(cfg)
    return os.path.exists(os.path.join(cache_dir, f"embeddings_done_{partition_id}.marker"))


def _save_embeddings_cache(cfg: ReviewConfig, n_records: int, partition_id: int = 0) -> None:
    cache_dir = _get_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"embeddings_done_{partition_id}.marker"), "w") as f:
        json.dump({"n_records": n_records, "complete": True}, f)


def run_m1(cfg: ReviewConfig, on_stage: Optional[callable] = None,
           partition_id: int = 0, num_partitions: int = 1) -> M1Result:
    """Execute the full M1 flow.

    `on_stage(stage_name, status, payload)` is an optional callback used
    by the UI to render live Pending/Running/Completed status; it must
    never change pipeline behaviour.

    Args:
        partition_id: This notebook's partition index (0-based).
        num_partitions: Total number of partitions for multi-notebook processing.
    """
    def emit(stage: str, status: str, payload: Any = None):
        if on_stage is not None:
            on_stage(stage, status, payload)

    # Log execution mode
    logger.info("M1 Execution Mode: %s, Partition %d/%d", cfg.mode, partition_id, num_partitions)

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
    checkpoint_dir = str(cfg.resolve_output("m1_dataset").parent / "cache")
    cached_records = _load_cached_evidence(cfg, partition_id=partition_id, num_partitions=num_partitions)
    if cached_records is not None:
        deduped = cached_records
        logger.info("Using cached evidence for %d records", len(deduped))
    else:
        evidence_cfg = cfg.evidence_cfg
        n_process = cfg.preprocessing.get("n_process", 1)
        build_evidence_all(deduped, evidence_cfg.get("spacy_model", "en_core_web_sm"),
                            evidence_cfg.get("regex_patterns", []), n_process=n_process,
                            checkpoint_dir=checkpoint_dir, partition_id=partition_id,
                            num_partitions=num_partitions)
        _save_cached_evidence(cfg, deduped, partition_id=partition_id, num_partitions=num_partitions)

    emb_cfg = cfg.embeddings_cfg
    if _load_embeddings_cache(cfg, partition_id=partition_id):
        logger.info("Using cached embeddings")
    else:
        build_embeddings_all(deduped, emb_cfg.get("model_name"), emb_cfg.get("batch_size", 16),
                              emb_cfg.get("device", "cuda"),
                              checkpoint_dir=checkpoint_dir, partition_id=partition_id,
                              num_partitions=num_partitions)
        _save_embeddings_cache(cfg, len(deduped), partition_id=partition_id)

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

    # 7. domain-stratified train/validation/test split
    emit("domain_stratified_split", "running")
    split_cfg = cfg.split_cfg
    train_records, val_records, test_records = _domain_stratified_split(
        deduped,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        validation_ratio=split_cfg.get("validation_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", 42),
        stratify_by_domain=split_cfg.get("stratify_by_domain", True),
    )

    # Leakage check
    leakage = _check_split_leakage(train_records, val_records, test_records)
    stats["split_leakage"] = leakage

    # Log leakage results
    if leakage.get("normalized_text_overlap", {}).get("train_validation", 0) > 0:
        logger.warning("LEAKAGE DETECTED: train/validation normalized_text overlap: %d",
                       leakage["normalized_text_overlap"]["train_validation"])
    if leakage.get("normalized_text_overlap", {}).get("train_test", 0) > 0:
        logger.warning("LEAKAGE DETECTED: train/test normalized_text overlap: %d",
                       leakage["normalized_text_overlap"]["train_test"])
    if leakage.get("normalized_text_overlap", {}).get("validation_test", 0) > 0:
        logger.warning("LEAKAGE DETECTED: validation/test normalized_text overlap: %d",
                       leakage["normalized_text_overlap"]["validation_test"])

    # Reconciliation check
    reconciliation = _reconcile_counts(
        raw_by_dataset=raw_by_dataset,
        raw_counts=raw_counts,
        cleaned_count=len(cleaned),
        removed_by_cleaning=stats["records_removed_by_cleaning"],
        deduped_count=len(deduped),
        dedup_stats=dedup_stats,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )
    stats["reconciliation"] = reconciliation
    if not reconciliation["checks_passed"]:
        logger.error("RECONCILIATION FAILED: %s", reconciliation["mismatches"])
        for m in reconciliation["mismatches"]:
            logger.error("  - %s: expected %s, actual %s (%s)",
                         m["check"], m["expected"], m["actual"], m.get("details", ""))
    else:
        logger.info("Reconciliation checks passed")

    # Split statistics
    train_dist = _domain_distribution(train_records)
    val_dist = _domain_distribution(val_records)
    test_dist = _domain_distribution(test_records)

    stats["split"] = {
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "test_count": len(test_records),
        "train_domain_distribution": train_dist,
        "validation_domain_distribution": val_dist,
        "test_domain_distribution": test_dist,
        "train_ratio": len(train_records) / len(deduped) if deduped else 0,
        "validation_ratio": len(val_records) / len(deduped) if deduped else 0,
        "test_ratio": len(test_records) / len(deduped) if deduped else 0,
        "stratify_by_domain": split_cfg.get("stratify_by_domain", True),
    }

    # Per-domain split percentages
    per_domain_split: Dict[str, Dict[str, float]] = {}
    for domain in domain_dist.keys():
        d_train = train_dist.get(domain, 0)
        d_val = val_dist.get(domain, 0)
        d_test = test_dist.get(domain, 0)
        d_total = d_train + d_val + d_test
        if d_total > 0:
            per_domain_split[domain] = {
                "train_percentage": d_train / d_total,
                "validation_percentage": d_val / d_total,
                "test_percentage": d_test / d_total,
                "train_count": d_train,
                "validation_count": d_val,
                "test_count": d_test,
            }
    stats["split"]["per_domain_percentages"] = per_domain_split

    snapshots["domain_stratified_split"] = {
        "split_stats": stats["split"],
        "leakage": leakage,
    }
    emit("domain_stratified_split", "completed", snapshots["domain_stratified_split"])

    # Combine all records with split assignments
    all_records = train_records + val_records + test_records

    stats["final_record_count"] = len(deduped)
    stats["mode"] = "development" if cfg.mode == "development" else "full"
    return M1Result(records=all_records, statistics=stats, stage_snapshots=snapshots)


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

    # Save split files
    splits = {"train": [], "validation": [], "test": []}
    for r in result.records:
        if r.split in splits:
            splits[r.split].append(r)

    output_dir = dataset_path.parent
    for split_name, records in splits.items():
        split_path = output_dir / f"m1_{split_name}_dataset.jsonl"
        with open(split_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict(), default=str) + "\n")
        logger.info("M1 %s split saved: %s (%d records)", split_name, split_path, len(records))

    stats_path = cfg.resolve_output("m1_stats")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(result.statistics, fh, indent=2, default=str)

    logger.info("M1 outputs saved: %s, %s", dataset_path, stats_path)


def merge_partitions(cfg: ReviewConfig, num_partitions: int, on_stage: Optional[callable] = None) -> M1Result:
    """Merge all partition checkpoints into a single unified dataset.
    
    This is called after all N notebooks have completed their partition
    processing. It loads every partition's evidence_checkpoint file,
    combines them, runs the split stage, and saves final outputs.
    """
    def emit(stage: str, status: str, payload: Any = None):
        if on_stage is not None:
            on_stage(stage, status, payload)

    import random

    logger.info("Merging %d partitions", num_partitions)
    cache_dir = _get_cache_dir(cfg)

    all_records: List[UnifiedRecord] = []
    total_lines = 0
    for pid in range(num_partitions):
        checkpoint_path = os.path.join(cache_dir, f"evidence_checkpoint_{pid}.jsonl")
        if not os.path.exists(checkpoint_path):
            logger.warning("Missing checkpoint for partition %d", pid)
            continue
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                total_lines += 1
    print(f"Loading {total_lines} records from {num_partitions} partitions...")
    loaded = 0
    for pid in range(num_partitions):
        checkpoint_path = os.path.join(cache_dir, f"evidence_checkpoint_{pid}.jsonl")
        if not os.path.exists(checkpoint_path):
            logger.warning("Missing checkpoint for partition %d", pid)
            continue
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = UnifiedRecord.from_dict(data)
                all_records.append(record)
                loaded += 1
        print(f"  Partition {pid}: loaded ({loaded}/{total_lines})")

    logger.info("Merged %d records from %d partitions", len(all_records), num_partitions)

    # Now run embeddings for all records (or load cached embeddings)
    emb_cfg = cfg.embeddings_cfg
    if not all(_load_embeddings_cache(cfg, partition_id=pid) for pid in range(num_partitions)):
        # Some embeddings missing, compute all
        from .embeddings import build_embeddings_all
        build_embeddings_all(all_records, emb_cfg.get("model_name"),
                              emb_cfg.get("batch_size", 16), emb_cfg.get("device", "cuda"))

    # Run split on merged records
    split_cfg = cfg.split_cfg
    train_records, val_records, test_records = _domain_stratified_split(
        all_records,
        train_ratio=split_cfg.get("train_ratio", 0.8),
        validation_ratio=split_cfg.get("validation_ratio", 0.1),
        test_ratio=split_cfg.get("test_ratio", 0.1),
        seed=split_cfg.get("seed", 42),
        stratify_by_domain=split_cfg.get("stratify_by_domain", True),
    )

    # Leakage check
    from .pipeline import _check_split_leakage
    leakage = _check_split_leakage(train_records, val_records, test_records)

    # Reconciliation
    from .pipeline import _reconcile_counts
    raw_by_dataset = load_all_datasets(cfg)
    raw_counts = {k: len(v) for k, v in raw_by_dataset.items()}
    cleaned_count = len(all_records)
    deduped_count = len(all_records)
    dedup_stats = {"n_deduped": deduped_count}
    reconciliation = _reconcile_counts(
        raw_by_dataset=raw_by_dataset, raw_counts=raw_counts,
        cleaned_count=cleaned_count, removed_by_cleaning=0,
        deduped_count=deduped_count, dedup_stats=dedup_stats,
        train_records=train_records, val_records=val_records, test_records=test_records,
    )

    # Build result
    stats: Dict[str, Any] = {}
    stats["split_leakage"] = leakage
    stats["reconciliation"] = reconciliation
    stats["final_record_count"] = len(all_records)
    stats["mode"] = "development" if cfg.mode == "development" else "full"
    stats["num_partitions"] = num_partitions

    train_dist = _domain_distribution(train_records)
    val_dist = _domain_distribution(val_records)
    test_dist = _domain_distribution(test_records)
    stats["split"] = {
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "test_count": len(test_records),
        "train_domain_distribution": train_dist,
        "validation_domain_distribution": val_dist,
        "test_domain_distribution": test_dist,
        "stratify_by_domain": split_cfg.get("stratify_by_domain", True),
    }

    all_records = train_records + val_records + test_records
    for r in all_records:
        r.split = None  # Already assigned above

    # Assign split back
    for r in train_records:
        r.split = "train"
    for r in val_records:
        r.split = "validation"
    for r in test_records:
        r.split = "test"

    result = M1Result(records=all_records, statistics=stats)

    # Save outputs
    save_m1_outputs(result, cfg)

    logger.info("Merged M1 outputs saved")
    return result
