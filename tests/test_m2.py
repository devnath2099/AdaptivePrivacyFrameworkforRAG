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