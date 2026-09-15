"""Offline M5 regression tests; no pretrained weights or datasets required."""
import json

import numpy as np
import pytest
import torch

from m5_uncertainty.mc_dropout import run_m5_inference
from m5_uncertainty.calibration import (
    compute_correct_vs_incorrect_uncertainty, compute_multilabel_ece,
)

DIMS = dict(sensitivity=3, intent=5, disclosure_scope=2,
            entity_tags=4, threat_content=3)


class ToyModel(torch.nn.Module):
    def __init__(self, passes):
        super().__init__()
        self.calls = 0
        self.passes = passes

    def forward(self, input_ids, attention_mask):
        offset = self.calls % self.passes
        self.calls += 1
        return {f"{dim}_logits": (input_ids[:, :1].float() + offset)
                * torch.arange(size).float() for dim, size in DIMS.items()}


@pytest.mark.parametrize("passes", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_record_statistics_and_serialization(passes, batch_size):
    dataset = [dict(record_id=f"r{i}", input_ids=torch.tensor([i]),
                    attention_mask=torch.ones(1)) for i in range(5)]
    results, summary = run_m5_inference(
        ToyModel(passes), dataset, mc_passes=passes,
        batch_size=batch_size, device="cpu")
    assert [r["record_id"] for r in results] == [f"r{i}" for i in range(5)]
    for i, record in enumerate(results):
        for dim, size in DIMS.items():
            logits = torch.stack([(i + t) * torch.arange(size).float()
                                  for t in range(passes)])
            probs = (logits.softmax(-1) if dim in
                     ("sensitivity", "intent", "disclosure_scope")
                     else logits.sigmoid())
            np.testing.assert_allclose(record[f"{dim}_mean"], probs.mean(0), atol=1e-7)
            assert record[f"{dim}_variance"] == pytest.approx(
                probs.var(0, unbiased=False).mean().item(), abs=1e-7)
    for dim in DIMS:
        assert summary[dim]["mean_variance"] == pytest.approx(
            np.mean([r[f"{dim}_variance"] for r in results]), abs=1e-7)
    json.dumps([results, summary], allow_nan=False)


@pytest.mark.parametrize("correct", [True, False])
def test_calibration_empty_groups_and_json(correct):
    preds = {dim: np.tile([0.9] + [0.1] * (size - 1), (3, 1))
             for dim, size in DIMS.items()}
    targets = {dim: p.copy() if correct else 1 - p for dim, p in preds.items()}
    variances = {dim: np.array([0.01, 0.02, 0.03]) for dim in DIMS}
    grouped = compute_correct_vs_incorrect_uncertainty(preds, variances, targets)
    empty_group = "incorrect" if correct else "correct"
    for dim in DIMS:
        assert all(v is None for v in grouped[dim][empty_group].values())
    ece, bins, per_category = compute_multilabel_ece(
        preds["entity_tags"], targets["entity_tags"])
    assert len(per_category) == 4
    assert ece == pytest.approx(0.1 if correct else 0.9)
    json.dumps([grouped, bins, per_category], allow_nan=False)


def test_empty_dataset_and_invalid_passes():
    with pytest.raises(ValueError, match="non-empty"):
        run_m5_inference(ToyModel(1), [], device="cpu")
    with pytest.raises(ValueError, match="mc_passes"):
        run_m5_inference(ToyModel(1), [], mc_passes=0, device="cpu")
