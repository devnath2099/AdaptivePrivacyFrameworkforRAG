import torch
import pytest
from review2.m4.calibration import fit_temperature
from review2.m4.uncertainty import sample_logits, uncertainty
from review2.m4.metrics import reliability_metrics
from review2.m2.alignment import encode
from review2.m2.model import build
from review2.m2.inference import predict


def test_temperature_calibration_only_and_detached():
    logits = torch.tensor([[10., 0.], [0., 10.], [10., 0.], [0., 10.]], requires_grad=True)
    targets = torch.tensor([0, 1, 1, 0])
    result = fit_temperature(logits, targets, 'calibration')
    assert result['temperature'] > 0 and result['nll_after'] <= result['nll_before']
    assert logits.grad is None
    with pytest.raises(ValueError):
        fit_temperature(logits, targets, 'test')


def test_mc_stochasticity_weights_and_mode_restoration(tokenizer, record):
    labels = ['O', 'B-EMAIL', 'I-EMAIL']
    model = build({'tiny': True, 'family': 'deberta'}, labels, tokenizer)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    data = encode([record], tokenizer, labels, 32)
    samples, _ = sample_logits(model, data, tokenizer, 3, 1)
    assert not torch.equal(samples[0][0], samples[0][1])
    assert not model.training
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    _, first, _ = predict(model, data, tokenizer, labels)
    _, second, _ = predict(model, data, tokenizer, labels)
    assert torch.equal(first[0], second[0])


def test_uncertainty_known_distribution():
    p = torch.tensor([[[1., 0.]], [[0., 1.]]])
    result = uncertainty(p)
    assert torch.allclose(result['entropy'], torch.ones(1))
    assert torch.allclose(result['mutual_information'], torch.ones(1))
    assert result['variation_ratio'].item() == .5


def test_perfect_calibration_metrics():
    p = torch.eye(2)
    metrics = reliability_metrics(p, torch.tensor([0, 1]))
    assert metrics['ece'] == metrics['nll'] == metrics['brier'] == metrics['aurc'] == 0
    assert metrics['error_auroc'] is None
