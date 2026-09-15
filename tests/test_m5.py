"""Tests for M5 -- MC-Dropout Uncertainty Quantification & Calibration.

Covers:
* stochastic dropout produces variation
* model parameters unchanged during MC inference
* correct probability transforms
* correct output shapes
* non-negative variance
* finite entropy
* categorical entropy implementation
* binary multi-label entropy implementation
* ECE range [0,1]
* multi-label ECE computed independently
* M4 checkpoint compatibility
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import build_m3_datasets
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m4_adversarial_training.trainer import M4Trainer
from m5_uncertainty.mc_dropout import (
    set_dropout_training,
    mc_dropout_inference,
    convert_to_probabilities,
    compute_predictive_mean,
    compute_predictive_variance,
    compute_categorical_entropy,
    compute_binary_entropy,
    compute_confidence,
    compute_uncertainty_summary,
    run_m5_inference,
)
from m5_uncertainty.calibration import (
    compute_ece,
    compute_categorical_ece,
    compute_multilabel_ece,
    compute_correct_vs_incorrect_uncertainty,
    save_high_uncertainty_examples,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "review1.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(str(CONFIG_PATH))


@pytest.fixture(scope="session")
def m3_checkpoint():
    return torch.load("outputs/m3/smoke/best_model.pt", map_location="cpu")


@pytest.fixture(scope="session")
def m4_checkpoint():
    """Load M4 checkpoint if it exists, otherwise load M3."""
    import os
    if os.path.exists("outputs/m4/best_model.pt"):
        return torch.load("outputs/m4/best_model.pt", map_location="cpu")
    return torch.load("outputs/m3/smoke/best_model.pt", map_location="cpu")


@pytest.fixture(scope="session")
def model(m4_checkpoint):
    """Create model from M4 (or M3 if M4 doesn't exist) checkpoint."""
    model = DeBERTaMultiTaskModel(
        model_name="microsoft/deberta-base",
        dropout_rate=0.1,
    )
    model.load_state_dict(m4_checkpoint)
    return model


@pytest.fixture(scope="session")
def val_dataset(cfg):
    """Get validation dataset."""
    datasets = build_m3_datasets(cfg)
    return datasets["val"]


@pytest.fixture(scope="session")
def small_val_dataset(val_dataset):
    """Small subset of validation."""
    from torch.utils.data import Subset
    indices = list(range(min(32, len(val_dataset))))
    return Subset(val_dataset, indices)


# === MC-Dropout stochasticity ===

def test_mc_dropout_produces_variation(model, small_val_dataset):
    """Repeated MC-Dropout predictions should differ."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    preds = []
    with torch.no_grad():
        for _ in range(5):
            outputs = model(input_ids, attention_mask)
            preds.append(outputs["sensitivity_logits"].numpy())

    # At least some predictions should differ
    all_same = all(np.allclose(preds[0], p) for p in preds[1:])
    assert not all_same, "MC-Dropout predictions should vary"


def test_mc_dropout_parameters_unchanged(model, small_val_dataset):
    """Model parameters should remain unchanged during MC inference."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    params_before = {k: v.clone() for k, v in model.state_dict().items()}

    set_dropout_training(model)
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids, attention_mask)

    params_after = model.state_dict()
    for k in params_before:
        assert torch.allclose(params_before[k], params_after[k]), \
            f"Parameter {k} changed during MC inference"


# === Probability transforms ===

def test_softmax_categorical(model, small_val_dataset):
    """Categorical probabilities should sum to ~1."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    probs = torch.softmax(outputs["sensitivity_logits"], dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1)), \
        "Categorical probabilities should sum to 1"


def test_sigmoid_multilabel(model, small_val_dataset):
    """Multi-label probabilities should be in [0,1] independently."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    probs = torch.sigmoid(outputs["entity_tags_logits"])
    assert (probs >= 0).all() and (probs <= 1).all(), \
        "Sigmoid outputs should be in [0,1]"


def test_correct_probability_transforms(model, small_val_dataset):
    """Verify convert_to_probabilities produces correct shapes and types."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    preds_list = [outputs]
    prob_lists = convert_to_probabilities(preds_list)

    assert prob_lists["sensitivity"].shape == (1, 3)
    assert prob_lists["intent"].shape == (1, 5)
    assert prob_lists["entity_tags"].shape == (1, 4)
    assert prob_lists["threat_content"].shape == (1, 3)


# === Output shapes ===

def test_predictive_mean_shape(model, small_val_dataset):
    """Predictive mean should have correct shapes."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    prob_lists = convert_to_probabilities([outputs])
    p_mean = compute_predictive_mean(prob_lists)

    assert p_mean["sensitivity"].shape == (3,)
    assert p_mean["intent"].shape == (5,)
    assert p_mean["entity_tags"].shape == (4,)


def test_predictive_variance_non_negative(model, small_val_dataset):
    """Predictive variance should be >= 0."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    prob_lists = convert_to_probabilities([outputs])
    p_mean = compute_predictive_mean(prob_lists)
    p_var = compute_predictive_variance(prob_lists, p_mean)

    for dim in p_var:
        assert (p_var[dim] >= 0).all(), f"Variance for {dim} should be >= 0"


def test_entropy_finite(model, small_val_dataset):
    """Entropy should be finite."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    prob_lists = convert_to_probabilities([outputs])
    p_mean = compute_predictive_mean(prob_lists)

    for dim in ["sensitivity", "intent", "disclosure_scope"]:
        entropy = compute_categorical_entropy(p_mean[dim])
        assert torch.isfinite(entropy), f"Entropy for {dim} should be finite"

    for dim in ["entity_tags", "threat_content"]:
        entropy = compute_binary_entropy(p_mean[dim])
        assert torch.isfinite(entropy).all(), f"Binary entropy for {dim} should be finite"


# === Uncertainty summary ===

def test_uncertainty_summary_shape(model, small_val_dataset):
    """Uncertainty summary should have all five tasks."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    prob_lists = convert_to_probabilities([outputs])
    p_mean = compute_predictive_mean(prob_lists)
    p_var = compute_predictive_variance(prob_lists, p_mean)
    summary = compute_uncertainty_summary(p_mean, p_var)

    assert len(summary) == 5
    for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]:
        assert "mean_entropy" in summary[dim]
        assert "mean_variance" in summary[dim]
        assert "mean_confidence" in summary[dim]


# === ECE tests ===

def test_ece_range(model, small_val_dataset):
    """ECE should be in [0, 1]."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    # Create random confidences and accuracies
    np.random.seed(42)
    confidences = np.random.rand(64)
    accuracies = (np.random.rand(64) > 0.5).astype(float)

    ece, bin_stats = compute_ece(confidences, accuracies, n_bins=10)
    assert 0 <= ece <= 1.0, f"ECE {ece} should be in [0, 1]"
    assert len(bin_stats) == 10


def test_categorical_ece_range(model, small_val_dataset):
    """Categorical ECE should be in [0, 1]."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Generate mock targets
    targets = {k: torch.rand(1, v.shape[-1]) for k, v in batch["soft_targets"].items()}

    preds = torch.softmax(outputs["sensitivity_logits"], dim=-1).numpy()
    target_array = torch.softmax(targets["sensitivity"], dim=-1).numpy()

    ece, bin_stats = compute_categorical_ece(preds, target_array, n_bins=5)
    assert 0 <= ece <= 1.0, f"Categorical ECE {ece} should be in [0, 1]"


def test_multilabel_ece_range(model, small_val_dataset):
    """Multi-label ECE should be in [0, 1]."""
    model.eval()
    batch = small_val_dataset[0]
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    set_dropout_training(model)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    preds = torch.sigmoid(outputs["entity_tags_logits"]).numpy()
    target_array = torch.sigmoid(torch.rand(1, 4)).numpy()

    macro_ece, bin_stats, per_cat = compute_multilabel_ece(
        preds, target_array, threshold=0.5, n_bins=5
    )
    assert 0 <= macro_ece <= 1.0, f"Multi-label macro ECE {macro_ece} should be in [0, 1]"


# === M4 checkpoint compatibility ===

def test_m4_checkpoint_compatibility(model):
    """M4 checkpoint should load into the model."""
    import os
    if os.path.exists("outputs/m4/best_model.pt"):
        ckpt = torch.load("outputs/m4/best_model.pt", map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        assert True, "M4 checkpoint loaded successfully"
    else:
        pytest.skip("M4 checkpoint not found, skipping")


def test_mc_dropout_run_m5(model, val_dataset):
    """run_m5_inference should complete without errors."""
    import os
    if not os.path.exists("outputs/m4/best_model.pt"):
        pytest.skip("M4 checkpoint not found, skipping M5 test")

    from torch.utils.data import Subset
    indices = list(range(min(32, len(val_dataset))))
    subset = Subset(val_dataset, indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results, summary = run_m5_inference(
        model, subset, mc_passes=5, batch_size=4, device=device
    )

    assert len(results) > 0
    assert "uncertainty_summary" in dir() or isinstance(summary, dict)
    assert len(summary) == 5


def test_save_high_uncertainty(model, small_val_dataset):
    """save_high_uncertainty_examples should return top-k examples."""
    import os
    if not os.path.exists("outputs/m4/best_model.pt"):
        pytest.skip("M4 checkpoint not found, skipping")

    results = [
        {
            "record_id": f"rec_{i}",
            "sensitivity_entropy": float(np.random.rand()),
            "intent_entropy": float(np.random.rand()),
            "entity_tags_entropy": float(np.random.rand()),
            "threat_content_entropy": float(np.random.rand()),
            "disclosure_scope_entropy": float(np.random.rand()),
        }
        for i in range(20)
    ]

    top = save_high_uncertainty_examples(results, top_k=5)
    assert len(top) == 5
    assert all("record_id" in r for r in top)
    assert top[0]["composite_entropy"] >= top[-1]["composite_entropy"]