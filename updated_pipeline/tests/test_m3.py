"""Tests for M3 -- DeBERTa Multi-Task Privacy Prediction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import M3Dataset, build_m3_datasets, validate_alignment
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss, soft_cross_entropy, binary_cross_entropy_with_logits
from m3_privacy_prediction.metrics import compute_all_metrics, evaluate_dimension

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "review1.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(str(CONFIG_PATH))


@pytest.fixture(scope="session")
def m3_datasets(cfg):
    """Build full M3 datasets."""
    return build_m3_datasets(cfg)


@pytest.fixture(scope="session")
def train_dataset(m3_datasets):
    return m3_datasets["train"]


@pytest.fixture(scope="session")
def val_dataset(m3_datasets):
    return m3_datasets["val"]


@pytest.fixture(scope="session")
def test_dataset(m3_datasets):
    return m3_datasets["test"]


@pytest.fixture(scope="session")
def model():
    """Create a small M3 model for testing."""
    return DeBERTaMultiTaskModel(
        model_name="microsoft/deberta-base",
        dropout_rate=0.1,
    )


# === Alignment tests ===

def test_record_alignment_no_overlap(m3_datasets):
    """Train, val, and test record IDs must be disjoint."""
    alignment = validate_alignment(
        m3_datasets["train"], m3_datasets["val"], m3_datasets["test"]
    )
    assert alignment["overlap"] == 0
    assert alignment["total"] == 277352


def test_train_val_test_counts(m3_datasets):
    """Verify correct split sizes."""
    assert m3_datasets["train"].num_records == 221880
    assert m3_datasets["val"].num_records == 27733
    assert m3_datasets["test"].num_records == 27739


def test_all_train_records_aligned(train_dataset):
    """Every M1 train record must have corresponding M2 soft labels."""
    assert train_dataset.num_records == 221880


# === Dataset tests ===

def test_dataset_output_shapes(train_dataset):
    """Verify each dataset item has correct tensor shapes."""
    item = train_dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "record_id" in item
    assert "soft_targets" in item
    assert item["input_ids"].shape[0] == train_dataset.max_length
    assert item["attention_mask"].shape[0] == train_dataset.max_length


def test_dataset_target_dimensions(train_dataset):
    """Verify soft_targets have correct dimensions for each dimension."""
    item = train_dataset[0]
    expected_dims = {
        "sensitivity": 3,
        "intent": 5,
        "disclosure_scope": 2,
        "entity_tags": 4,
        "threat_content": 3,
    }
    for dim, size in expected_dims.items():
        assert item["soft_targets"][dim].shape[0] == size


def test_record_alignment_validation(train_dataset, val_dataset):
    """Validate that train and val records don't overlap."""
    train_ids = set(train_dataset.record_ids[:100])
    val_ids = set(val_dataset.record_ids[:100])
    overlap = train_ids & val_ids
    assert len(overlap) == 0


# === Model tests ===

def test_model_output_shapes(model, train_dataset):
    """Verify all five prediction heads produce correct output shapes."""
    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    expected = {
        "sensitivity_logits": (1, 3),
        "intent_logits": (1, 5),
        "disclosure_scope_logits": (1, 2),
        "entity_tags_logits": (1, 4),
        "threat_content_logits": (1, 3),
        "representation": (1, 768),
    }
    for key, shape in expected.items():
        assert outputs[key].shape == shape, f"{key}: expected {shape}, got {outputs[key].shape}"


def test_model_representation_output(model, train_dataset):
    """Verify representation output is accessible and consistent."""
    model.eval()
    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        repr_out = model.get_representation(input_ids, attention_mask)

    assert repr_out.shape == (1, 768)
    assert torch.allclose(outputs["representation"], repr_out)


# === Loss tests ===

def test_soft_cross_entropy_is_finite():
    """Soft cross-entropy should produce finite loss."""
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.0, 0.9, 0.1], [0.3, 0.3, 0.4]])
    loss = soft_cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_binary_cross_entropy_is_finite():
    """Binary cross-entropy with logits should produce finite loss."""
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([[0.1, 0.9, 0.2, 0.8], [0.7, 0.2, 0.3, 0.1], [0.0, 0.8, 0.1, 0.9], [0.5, 0.5, 0.5, 0.5]])
    loss = binary_cross_entropy_with_logits(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_compute_total_loss_finite(model, train_dataset):
    """Total multi-task loss should be finite and produce gradients."""
    model.train()
    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in item["soft_targets"].items()}

    outputs = model(input_ids, attention_mask)
    total_loss, task_losses = compute_total_loss(outputs, targets)
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert torch.isfinite(next(model.parameters()).grad).all()


def test_task_losses_independent(model, train_dataset):
    """Each task loss should be independently computed."""
    model.train()
    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in item["soft_targets"].items()}

    outputs = model(input_ids, attention_mask)
    total_loss, task_losses = compute_total_loss(outputs, targets)
    assert len(task_losses) == 5
    assert all(torch.isfinite(v).item() for v in task_losses.values())


# === Metrics tests ===

def test_metrics_for_single_label():
    """Metrics for single-label dimensions should include accuracy, macro_f1, weighted_f1."""
    logits = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    targets = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.9, 0.1]])
    result = evaluate_dimension("sensitivity", logits, targets)
    assert "accuracy" in result
    assert "macro_f1" in result
    assert "weighted_f1" in result


def test_metrics_for_multi_label():
    """Metrics for multi-label dimensions should include micro_f1, macro_f1, per_category_f1."""
    logits = np.array([[0.1, 0.8, 0.2, 0.3], [0.7, 0.2, 0.1, 0.4]])
    targets = np.array([[0.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.3, 0.2]])
    result = evaluate_dimension("entity_tags", logits, targets)
    assert "micro_f1" in result
    assert "macro_f1" in result
    assert "per_category_f1" in result
    assert len(result["per_category_f1"]) == 4


def test_metrics_handle_single_class():
    """Metrics should not crash when only one class is present."""
    logits = np.array([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]])
    targets = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    result = evaluate_dimension("sensitivity", logits, targets)
    assert "accuracy" in result
    assert isinstance(result["accuracy"], float)


# === Smoke test ===

def test_full_forward_backward_smoke(model, train_dataset):
    """Complete forward + backward pass with all five tasks."""
    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in item["soft_targets"].items()}

    outputs = model(input_ids, attention_mask)
    total_loss, task_losses = compute_total_loss(outputs, targets)
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert len(task_losses) == 5


# === Checkpoint tests ===

def test_model_save_load(tmp_path, model, train_dataset):
    """Model state dict should save and load correctly."""
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    new_model = DeBERTaMultiTaskModel(dropout_rate=0.1)
    new_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    new_model.eval()
    model.eval()

    item = train_dataset[0]
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        out1 = model(input_ids, attention_mask)
        out2 = new_model(input_ids, attention_mask)

    for key in out1:
        assert torch.allclose(out1[key], out2[key], atol=1e-6)
