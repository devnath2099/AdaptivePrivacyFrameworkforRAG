"""Validation tests for M1 -- Multi-Domain Privacy & Evidence Integration."""
from __future__ import annotations

from m1_data_integration.cleaner import clean_text, normalize_whitespace
from m1_data_integration.deduplicator import deduplicate
from m1_data_integration.loaders import load_all_datasets
from m1_data_integration.schemas import UnifiedRecord


def test_all_four_datasets_load(small_cfg):
    raw = load_all_datasets(small_cfg)
    assert set(raw.keys()) == {"healthcare", "fiqa", "hotpotqa", "nq"}
    for key, records in raw.items():
        assert len(records) > 0, f"dataset {key} produced no records"


def test_common_schema_produced(m1_result):
    assert len(m1_result.records) > 0
    for r in m1_result.records[:5]:
        assert isinstance(r, UnifiedRecord)


def test_required_fields_exist(m1_result):
    required = ["record_id", "domain", "source_dataset", "query_text", "normalized_text"]
    for r in m1_result.records[:10]:
        d = r.to_dict()
        for field_name in required:
            assert field_name in d
            assert d[field_name] is not None


def test_whitespace_normalization_works():
    assert normalize_whitespace("hello    \n  world\t!") == "hello world !"


def test_clean_text_handles_missing_value():
    assert clean_text(None) == ""


def test_duplicate_removal_works():
    r1 = UnifiedRecord(record_id="a", domain="d", source_dataset="s",
                        query_text="hello", context_text="", normalized_text="hello world")
    r2 = UnifiedRecord(record_id="b", domain="d", source_dataset="s",
                        query_text="hello", context_text="", normalized_text="hello world")
    r3 = UnifiedRecord(record_id="c", domain="d", source_dataset="s",
                        query_text="different", context_text="", normalized_text="different text")
    kept, stats = deduplicate([r1, r2, r3])
    assert len(kept) == 2
    assert stats["duplicates_removed"] == 1


def test_domain_labels_are_correct(m1_result):
    valid_domains = {"medical", "financial", "multi_hop_qa", "general_qa"}
    for r in m1_result.records:
        assert r.domain in valid_domains


def test_evidence_extraction_produces_valid_structures(m1_result):
    for r in m1_result.records[:10]:
        assert r.evidence is not None
        assert isinstance(r.evidence.entities, list)
        assert isinstance(r.evidence.dependency_relations, list)
        assert isinstance(r.evidence.regex_matches, dict)


def test_embedding_dimensions_consistent(m1_result):
    dims = {len(r.evidence.embedding) for r in m1_result.records if r.evidence and r.evidence.embedding}
    assert len(dims) <= 1, f"inconsistent embedding dimensions found: {dims}"
    assert len(dims) == 1


def test_no_duplicate_records_remain(m1_result):
    seen = set()
    for r in m1_result.records:
        key = r.normalized_text.lower()
        assert key not in seen, "deduplication left a duplicate normalized_text"
        seen.add(key)


def test_reconciliation_accounts_for_duplicates():
    """The reconciliation must treat dedup removals separately from the split,
    so a run with duplicates (split != cleaned) still reconciles."""
    from m1_data_integration.pipeline import _reconcile_counts

    # raw: 10 records, cleaning removes none, dedup removes 2 -> 8 survive.
    raw_counts = {"d": 10}
    raw_by_dataset = {"d": []}
    cleaned_count = 10
    removed_by_cleaning = 0
    deduped_count = 8
    dedup_stats = {"total_input_records": 10, "duplicates_removed": 2}
    # split into 6 train / 1 val / 1 test = 8 (matches deduped)
    train = [UnifiedRecord(record_id=f"t{i}", domain="d", source_dataset="s",
                            query_text="q", context_text="", normalized_text="n") for i in range(6)]
    val = [UnifiedRecord(record_id="v0", domain="d", source_dataset="s",
                          query_text="q", context_text="", normalized_text="n")]
    test = [UnifiedRecord(record_id="e0", domain="d", source_dataset="s",
                           query_text="q", context_text="", normalized_text="n")]

    rep = _reconcile_counts(
        raw_by_dataset=raw_by_dataset, raw_counts=raw_counts,
        cleaned_count=cleaned_count, removed_by_cleaning=removed_by_cleaning,
        deduped_count=deduped_count, dedup_stats=dedup_stats,
        train_records=train, val_records=val, test_records=test,
    )
    assert rep["checks_passed"], f"reconciliation failed: {rep['mismatches']}"


def test_reconciliation_detects_missing_dedup_accounting():
    """Regression guard: if the split does not equal the deduped count, the
    reconciliation must fail (rather than silently comparing to cleaned count)."""
    from m1_data_integration.pipeline import _reconcile_counts

    raw_counts = {"d": 10}
    raw_by_dataset = {"d": []}
    cleaned_count = 10
    deduped_count = 8
    dedup_stats = {"total_input_records": 10, "duplicates_removed": 2}
    # split totals 7 != deduped 8 -> must fail
    train = [UnifiedRecord(record_id=f"t{i}", domain="d", source_dataset="s",
                            query_text="q", context_text="", normalized_text="n") for i in range(5)]
    val = [UnifiedRecord(record_id="v0", domain="d", source_dataset="s",
                          query_text="q", context_text="", normalized_text="n")]
    test = [UnifiedRecord(record_id="e0", domain="d", source_dataset="s",
                           query_text="q", context_text="", normalized_text="n")]

    rep = _reconcile_counts(
        raw_by_dataset=raw_by_dataset, raw_counts=raw_counts,
        cleaned_count=cleaned_count, removed_by_cleaning=0,
        deduped_count=deduped_count, dedup_stats=dedup_stats,
        train_records=train, val_records=val, test_records=test,
    )
    assert not rep["checks_passed"]
    assert any(m["check"] == "split_equals_deduped" for m in rep["mismatches"])
