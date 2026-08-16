"""Integration tests: M1 output must feed M2 input without manual modification."""
from __future__ import annotations

from m2_label_generation.pipeline import run_m2


def test_m1_output_feeds_m2_without_modification(small_cfg, m1_result):
    m2_result = run_m2(m1_result.records, small_cfg)
    assert len(m2_result.dimension_results) == 5
    for res in m2_result.dimension_results.values():
        assert res.weak_labels.shape[0] == len(m1_result.records)


def test_diagnostics_generated_for_every_dimension(small_cfg, m1_result):
    m2_result = run_m2(m1_result.records, small_cfg)
    assert set(m2_result.diagnostics.keys()) == set(m2_result.dimension_results.keys())
    for diag in m2_result.diagnostics.values():
        assert "lf_coverage" in diag
        assert "mean_entropy" in diag


def test_stage_snapshots_present_for_ui(small_cfg, m1_result):
    events = []
    m2_result = run_m2(m1_result.records, small_cfg, on_stage=lambda s, st, p: events.append((s, st)))
    stage_names = {e[0] for e in events}
    assert {"privacy_dimension_setup", "labeling_functions", "lf_matrix_construction",
            "generative_model_fitting", "posterior_inference"}.issubset(stage_names)
    assert all(status in ("running", "completed") for _, status in events)
