"""Schema harmonization: RawRecord (per-dataset) -> UnifiedRecord (common schema).

This stage does NOT clean text (that is `cleaner.py`'s job); it only
maps heterogeneous per-dataset fields onto the single `UnifiedRecord`
representation so that every later stage is dataset-agnostic.
"""
from __future__ import annotations

import uuid
from typing import Dict, List

from .schemas import RawRecord, UnifiedRecord


def harmonize_record(raw: RawRecord, dataset_key: str, index: int) -> UnifiedRecord:
    record_id = f"{dataset_key}_{index:06d}_{uuid.uuid4().hex[:8]}"
    return UnifiedRecord(
        record_id=record_id,
        domain=raw.domain,
        source_dataset=raw.source_dataset,
        query_text=raw.query_text or "",
        context_text=raw.context_text or "",
        normalized_text="",  # filled in by cleaner.py
        metadata=dict(raw.metadata),
    )


def harmonize_all(raw_by_dataset: Dict[str, List[RawRecord]]) -> List[UnifiedRecord]:
    unified: List[UnifiedRecord] = []
    for dataset_key, records in raw_by_dataset.items():
        for idx, raw in enumerate(records):
            unified.append(harmonize_record(raw, dataset_key, idx))
    return unified
