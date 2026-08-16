"""Cleaning / normalization stage.

Handles: missing-value handling, whitespace normalization, basic
Unicode normalization, and construction of the `normalized_text` field
that all later evidence-extraction and embedding steps consume.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

from .schemas import UnifiedRecord

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    return unicodedata.normalize(form, text)


def handle_missing_value(text: str | None) -> str:
    return text if isinstance(text, str) else ""


def clean_text(text: str | None, unicode_form: str = "NFKC") -> str:
    text = handle_missing_value(text)
    text = normalize_unicode(text, unicode_form)
    text = normalize_whitespace(text)
    return text


def clean_record(record: UnifiedRecord, preprocessing_cfg: Dict[str, Any]) -> UnifiedRecord:
    unicode_form = preprocessing_cfg.get("unicode_normalize_form", "NFKC")
    min_len = preprocessing_cfg.get("min_text_length", 3)

    record.query_text = clean_text(record.query_text, unicode_form)
    record.context_text = clean_text(record.context_text, unicode_form)

    combined = (record.query_text + " " + record.context_text).strip()
    record.normalized_text = combined

    record.metadata["is_short_text"] = len(record.query_text) < min_len
    return record


def clean_all(records: List[UnifiedRecord], preprocessing_cfg: Dict[str, Any]) -> List[UnifiedRecord]:
    cleaned = [clean_record(r, preprocessing_cfg) for r in records]
    # drop records with essentially empty query text after cleaning
    valid = [r for r in cleaned if len(r.query_text) >= preprocessing_cfg.get("min_text_length", 3)]
    return valid
