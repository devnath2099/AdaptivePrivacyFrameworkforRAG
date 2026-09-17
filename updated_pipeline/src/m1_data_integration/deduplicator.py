"""Duplicate detection over the normalized text of unified records.

Uses an exact-match strategy on a case-normalized key, which is
sufficient and transparent for a research prototype; near-duplicate /
fuzzy detection is out of scope for Review 1.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import UnifiedRecord


def _dedup_key(record: UnifiedRecord, lowercase: bool) -> str:
    text = record.normalized_text
    return text.lower() if lowercase else text


def deduplicate(records: List[UnifiedRecord], lowercase: bool = True) -> Tuple[List[UnifiedRecord], Dict[str, int]]:
    """Mark exact-duplicate records and return (deduplicated_list, stats)."""
    seen: Dict[str, str] = {}
    kept: List[UnifiedRecord] = []
    n_duplicates = 0

    for record in records:
        key = _dedup_key(record, lowercase)
        if key in seen:
            record.is_duplicate = True
            n_duplicates += 1
            continue
        seen[key] = record.record_id
        kept.append(record)

    stats = {
        "total_input_records": len(records),
        "unique_records": len(kept),
        "duplicates_removed": n_duplicates,
        "duplicate_ratio": (n_duplicates / len(records)) if records else 0.0,
    }
    return kept, stats
