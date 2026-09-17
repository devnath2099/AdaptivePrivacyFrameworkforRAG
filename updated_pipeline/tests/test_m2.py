"""Validation tests for M2 -- Snorkel-Based Privacy Label Synthesis."""
from __future__ import annotations

import numpy as np
import pytest

from m2_label_generation.labeling_functions import DIMENSION_LFS, ENTITY_TAG_LFS, THREAT_CONTENT_LFS
from m2_label_generation.lf_engine import build_lf_matrix
from m2_label_generation.pipeline import run_m2
from m2_label_generation.taxonomy import ABSTAIN, build_taxonomy


@pytest.fixture(scope="module")
def m2_result(small_cfg, m1_result):
    return run_m2(m1_result.records, small_cfg)


def test_every_lf_returns_valid_label_or_abstain(small_cfg, m1_result):
    taxonomy = build_taxonomy(small_cfg.label_taxonomy)
    for dim, lfs in DIMENSION_LFS.items():
        spec = taxonomy[dim]
        result = build_lf_matrix(m1_result.records, lfs, spec)
        valid_values = set(range(spec.num_classes)) | {ABSTAIN}
        assert set(np.unique(result.matrix).tolist()).issubset(valid_values)


def test_lf_matrix_has_correct_shape(small_cfg, m1_result):
    taxonomy = build_taxonomy(small_cfg.label_taxonomy)
    spec = taxonomy["sensitivity"]
    result = build_lf_matrix(m1_result.records, DIMENSION_LFS["sensitivity"], spec)
    assert result.matrix.shape == (len(m1_result.records), len(DIMENSION_LFS["sensitivity"]))


def test_lf_labels_stay_inside_valid_label_space(small_cfg, m1_result):
    taxonomy = build_taxonomy(small_cfg.label_taxonomy)
    for dim, lfs in DIMENSION_LFS.items():
        spec = taxonomy[dim]
        result = build_lf_matrix(m1_result.records, lfs, spec)
        assert result.matrix.max() < spec.num_classes


def test_weak_label_probabilities_correct_dimensions(m2_result, small_cfg):
    taxonomy = build_taxonomy(small_cfg.label_taxonomy)
    for dim, res in m2_result.dimension_results.items():
        expected_k = taxonomy[dim].num_classes if not taxonomy[dim].is_multi_label else len(taxonomy[dim].labels)
        assert res.weak_labels.shape[1] == expected_k


def test_probabilities_non_negative(m2_result):
    for res in m2_result.dimension_results.values():
        assert np.all(res.weak_labels >= 0.0)


def test_multiclass_rows_sum_to_one(m2_result, small_cfg):
    taxonomy = build_taxonomy(small_cfg.label_taxonomy)
    for dim, res in m2_result.dimension_results.items():
        if taxonomy[dim].is_multi_label:
            continue
        row_sums = res.weak_labels.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_no_nan_or_inf_in_weak_labels(m2_result):
    for res in m2_result.dimension_results.values():
        assert not np.any(np.isnan(res.weak_labels))
        assert not np.any(np.isinf(res.weak_labels))


def test_all_five_dimensions_produce_output(m2_result):
    expected = {"entity_tags", "sensitivity", "intent", "disclosure_scope", "threat_content"}
    assert set(m2_result.dimension_results.keys()) == expected


def test_multi_label_threat_output_shape(m2_result, m1_result):
    res = m2_result.dimension_results["threat_content"]
    assert res.weak_labels.shape == (len(m1_result.records), 3)
    assert np.all(res.weak_labels >= 0.0) and np.all(res.weak_labels <= 1.0)


def test_threat_content_labeling_functions_exist():
    expected_categories = {"re_identification", "attribute_inference", "membership_inference"}
    assert set(THREAT_CONTENT_LFS.keys()) == expected_categories
    for lfs in THREAT_CONTENT_LFS.values():
        assert len(lfs) >= 1


def test_entity_tag_labeling_functions_exist():
    expected_categories = {"has_person", "has_organization", "has_location", "has_contact_identifier"}
    assert set(ENTITY_TAG_LFS.keys()) == expected_categories
    for lfs in ENTITY_TAG_LFS.values():
        assert len(lfs) >= 1


def test_multi_label_entity_tags_output_shape(m2_result, m1_result):
    res = m2_result.dimension_results["entity_tags"]
    assert res.weak_labels.shape == (len(m1_result.records), 4)
    assert np.all(res.weak_labels >= 0.0) and np.all(res.weak_labels <= 1.0)


def test_multi_label_lf_matrices_valid(small_cfg, m1_result):
    """Every LF in the multi-label dicts (entity_tags, threat_content) returns
    only {ABSTAIN, 0, 1} -- the binary presence/absence vote space -- since
    these are no longer routed through DIMENSION_LFS/build_lf_matrix directly
    with a shared multi-class spec.
    """
    from m2_label_generation.lf_engine import build_lf_matrix
    from m2_label_generation.taxonomy import DimensionSpec

    for lfs_dict in (ENTITY_TAG_LFS, THREAT_CONTENT_LFS):
        for category, lfs in lfs_dict.items():
            binary_spec = DimensionSpec(name=f"test_{category}", labels=["absent", "present"],
                                         is_multi_label=False)
            result = build_lf_matrix(m1_result.records, lfs, binary_spec)
            assert set(np.unique(result.matrix).tolist()).issubset({ABSTAIN, 0, 1})


def test_multi_label_categories_have_independent_probabilities(m2_result):
    """Multi-label dimensions must retain independent per-category probabilities:
    a record is allowed to be positive for multiple categories, so the weak-label
    rows must NOT be forced into one exclusive argmax assignment, and the columns
    must be mutually independent (a row may sum to > 1.0)."""
    for dim, n_categories in (("entity_tags", 4), ("threat_content", 3)):
        res = m2_result.dimension_results[dim]
        assert res._is_multi_label is True
        # shape is (n, n_categories) with each column an independent P(present)
        assert res.weak_labels.shape[1] == n_categories
        # independence: at least one row must have two or more categories >= 0.5
        # (guaranteed by the synthetic fixture which mixes entity types).
        positives = res.weak_labels >= 0.5
        assert np.any(positives.sum(axis=1) >= 2), \
            f"{dim} should allow multi-label co-occurrence but none found"


def test_multi_label_diagnostics_preserve_independent_summary(m2_result):
    """The diagnostics summary must NOT report an exclusive argmax distribution
    for multi-label dimensions. Instead it reports per_category_statistics and
    co_occurrence (independent per-category results)."""
    for dim in ("entity_tags", "threat_content"):
        diag = m2_result.diagnostics[dim]
        # 'label_distribution' must be the multi-label summary dict, not a flat
        # per-class exclusive count.
        assert "per_category_statistics" in diag["label_distribution"]
        assert "co_occurrence" in diag["label_distribution"]
        # every category must carry mean/median/std/min/max/quantiles/above_threshold
        for cat, stats in diag["label_distribution"]["per_category_statistics"].items():
            for key in ("mean", "median", "std", "min", "max", "quantiles", "above_threshold"):
                assert key in stats, f"missing '{key}' for {dim}/{cat}"
        # co-occurrence keys
        for key in ("zero_positive_records", "exactly_one_positive_record",
                    "two_positive_records", "three_or_more_positive_records"):
            assert key in diag["label_distribution"]["co_occurrence"]


def test_single_label_dimensions_still_argmax_exclusive(m2_result):
    """Single-label multi-class dimensions retain a proper distribution that
    sums to the record count."""
    for dim in ("sensitivity", "intent", "disclosure_scope"):
        diag = m2_result.diagnostics[dim]
        dist = diag["label_distribution"]
        assert "positive_counts" not in dist  # not multi-label
        assert sum(dist.values()) == diag["n_records"]


def test_multi_label_lfs_include_absent_votes(small_cfg, m1_result):
    """Each multi-label category must provide BOTH positive (present) and
    negative (absent) LF votes, not just PRESENT-or-ABSTAIN. A binary LabelModel
    that only ever sees 'present' collapses onto the positive prior, so the LFs
    must include at least one evidence-based ABSENT (0) vote per category."""
    from m2_label_generation.lf_engine import build_lf_matrix
    from m2_label_generation.taxonomy import DimensionSpec

    for lfs_dict in (ENTITY_TAG_LFS, THREAT_CONTENT_LFS):
        for category, lfs in lfs_dict.items():
            binary_spec = DimensionSpec(name=f"test_{category}", labels=["absent", "present"],
                                         is_multi_label=False)
            result = build_lf_matrix(m1_result.records, lfs, binary_spec)
            # the matrix must contain at least one column that votes 0 (absent)
            # somewhere, guaranteeing negative evidence is available.
            has_absent = np.any(result.matrix == 0)
            assert has_absent, f"{category} has no ABSENT (0) LF votes"


def test_multi_label_records_actual_per_category_backend(m2_result):
    """Multi-label dimensions must record the ACTUAL backend used per category
    (snorkel_label_model or fallback_generative), not a hard-coded combined
    method string. This prevents falsely claiming all categories used Snorkel."""
    for dim in ("entity_tags", "threat_content"):
        res = m2_result.dimension_results[dim]
        assert res.category_backends is not None, f"{dim} missing category_backends"
        # every label must have a truthful backend value
        for cat in res.label_names:
            assert cat in res.category_backends
            assert res.category_backends[cat] in ("snorkel_label_model", "fallback_generative")
        # combined method must reflect reality
        distinct = set(res.category_backends.values())
        if len(distinct) > 1:
            assert res.generative_result.method == "mixed_multi_label"
        else:
            assert res.generative_result.method == next(iter(distinct))


def test_multi_label_diagnostics_report_category_backends(m2_result):
    """The diagnostics JSON must carry the per-category backend provenance."""
    for dim in ("entity_tags", "threat_content"):
        diag = m2_result.diagnostics[dim]
        cb = diag.get("category_backends")
        assert cb is not None and isinstance(cb, dict)
        assert all(cb[c] in ("snorkel_label_model", "fallback_generative") for c in cb)
        if len(set(cb.values())) > 1:
            assert diag["generative_model_method"] == "mixed_multi_label"