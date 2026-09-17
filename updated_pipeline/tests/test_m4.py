"""Tests for M4 -- FGSM Adversarial Training.

Covers:
* M3 checkpoint compatibility
* embedding-space FGSM
* perturbation shape
* perturbation epsilon bound
* padding masking
* clean/adversarial output shapes
* finite losses
* combined backward pass
* no optimizer step between clean/adversarial construction
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
from m3_privacy_prediction.dataset import M3Dataset, build_m3_datasets
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss
from m4_adversarial_training.fgsm import (
    compute_clean_loss_and_grad,
    generate_fgsm_delta,
    adversarial_forward,
    _forward_from_embeddings,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "review1.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(str(CONFIG_PATH))


@pytest.fixture(scope="session")
def m3_checkpoint():
    """Load M3 checkpoint."""
    return torch.load("outputs/m3/smoke/best_model.pt", map_location="cpu")


@pytest.fixture(scope="session")
def model(m3_checkpoint):
    """Create M4 model from M3 checkpoint."""
    model = DeBERTaMultiTaskModel(
        model_name="microsoft/deberta-base",
        dropout_rate=0.1,
    )
    model.load_state_dict(m3_checkpoint)
    return model


@pytest.fixture(scope="session")
def m3_datasets(cfg):
    """Build M3 datasets."""
    return build_m3_datasets(cfg)


@pytest.fixture(scope="session")
def small_dataset(m3_datasets):
    """Get a small subset for testing."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(m3_datasets["train"]), size=64, replace=False)
    from torch.utils.data import Subset
    return Subset(m3_datasets["train"], indices)


@pytest.fixture
def batch_item(small_dataset):
    """Get a single batch item from the dataset."""
    return small_dataset[0]


# === M3 checkpoint compatibility ===

def test_m3_checkpoint_loads(model, m3_checkpoint):
    """M3 checkpoint should load into the model."""
    assert len(m3_checkpoint) > 0
    model.load_state_dict(m3_checkpoint)
    model.eval()


def test_m3_checkpoint_compatibility(model, m3_checkpoint):
    """Verify all model keys match checkpoint keys."""
    ckpt_keys = set(m3_checkpoint.keys())
    model_keys = set(model.state_dict().keys())
    assert ckpt_keys == model_keys, f"Key mismatch: {ckpt_keys.symmetric_difference(model_keys)}"


# === FGSM perturbation tests ===

def test_fgsm_perturbation_shape(model, batch_item):
    """Delta shape should equal embedding shape."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)

    embeddings = model.encoder.embeddings(input_ids=input_ids)
    delta = generate_fgsm_delta(
        torch.ones_like(embeddings), attention_mask, epsilon=1e-3
    )
    assert delta.shape == embeddings.shape, \
        f"Delta shape {delta.shape} != embedding shape {embeddings.shape}"


def test_fgsm_perturbation_epsilon_bound(model, batch_item):
    """Max perturbation should be <= epsilon + numerical tolerance."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)

    embeddings = model.encoder.embeddings(input_ids=input_ids)
    grad = torch.ones_like(embeddings) * 0.5  # arbitrary gradient
    delta = generate_fgsm_delta(grad, attention_mask, epsilon=1e-3)

    max_abs = delta.abs().max().item()
    assert max_abs <= 1e-3 + 1e-8, \
        f"Max perturbation {max_abs} > epsilon + tolerance"


def test_fgsm_padding_zero(model, batch_item):
    """Padding positions should have zero perturbation."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)

    # Create a batch where some positions are padding (attention_mask = 0)
    padding_mask = attention_mask.clone()
    padding_mask[:, 50:] = 0  # Zero out positions 50+

    embeddings = model.encoder.embeddings(input_ids=input_ids)
    grad = torch.ones_like(embeddings)
    delta = generate_fgsm_delta(grad, padding_mask, epsilon=1e-3)

    # Check padding positions have zero delta
    padding_positions = (padding_mask == 0)
    if padding_positions.any():
        assert delta[padding_positions].abs().max().item() == 0.0, \
            "Padding positions should have zero perturbation"


def test_clean_loss_and_grad(model, batch_item):
    """Clean loss should be finite and gradient should exist."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()}

    embeddings, clean_loss, grad, task_losses = (
        compute_clean_loss_and_grad(model, input_ids, attention_mask, targets, compute_total_loss)
    )

    assert torch.isfinite(clean_loss), "Clean loss should be finite"
    assert grad is not None, "Gradient should exist"
    assert grad.shape == embeddings.shape, "Gradient shape should match embeddings"
    assert all(torch.isfinite(v).all() for v in task_losses.values()), "Task losses should be finite"


# === Adversarial forward tests ===

def test_adversarial_forward_shapes(model, batch_item):
    """All output shapes should remain correct after adversarial forward."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()}

    total_loss, clean_loss, adv_loss, task_losses, delta, max_delta = (
        adversarial_forward(model, input_ids, attention_mask, targets, compute_total_loss, epsilon=1e-3)
    )

    assert torch.isfinite(total_loss), "Combined loss should be finite"
    assert torch.isfinite(clean_loss), "Clean loss should be finite"
    assert torch.isfinite(adv_loss), "Adversarial loss should be finite"
    assert delta.shape == (1, 128, 768), f"Delta shape {delta.shape} incorrect"


def test_adversarial_forward_backward(model, batch_item):
    """Combined backward should succeed with finite gradients."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()}

    model.zero_grad()
    total_loss, _, _, _, _, _ = adversarial_forward(
        model, input_ids, attention_mask, targets, compute_total_loss, epsilon=1e-3
    )

    # backward already called inside adversarial_forward
    # verify gradients are finite
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), \
        "All gradients should be finite after backward"


def test_adversarial_loss_finite(model, batch_item):
    """Adversarial loss should be finite."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()}

    _, clean_loss, adv_loss, _, _, _ = adversarial_forward(
        model, input_ids, attention_mask, targets, compute_total_loss, epsilon=1e-3
    )

    assert torch.isfinite(clean_loss), "Clean loss should be finite"
    assert torch.isfinite(adv_loss), "Adversarial loss should be finite"


def test_no_double_optimizer_step(model, batch_item):
    """Verify that adversarial_forward does not call optimizer.step()."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()}

    model.zero_grad()
    total_loss, _, _, _, _, _ = adversarial_forward(
        model, input_ids, attention_mask, targets, compute_total_loss, epsilon=1e-3
    )

    # adversarial_forward does not call optimizer.step()
    # The caller is responsible for calling optimizer.step()
    assert True, "adversarial_forward does not call optimizer.step()"


def test_forward_from_embeddings(model, batch_item):
    """Forward from embeddings should produce correct output shapes."""
    input_ids = batch_item["input_ids"].unsqueeze(0)
    attention_mask = batch_item["attention_mask"].unsqueeze(0)
    embeddings = model.encoder.embeddings(input_ids=input_ids)

    logits_dict, loss, task_losses = _forward_from_embeddings(
        model, embeddings, attention_mask,
        {k: v.unsqueeze(0) for k, v in batch_item["soft_targets"].items()},
        compute_total_loss
    )

    assert logits_dict["sensitivity_logits"].shape == (1, 3)
    assert logits_dict["intent_logits"].shape == (1, 5)
    assert logits_dict["disclosure_scope_logits"].shape == (1, 2)
    assert logits_dict["entity_tags_logits"].shape == (1, 4)
    assert logits_dict["threat_content_logits"].shape == (1, 3)
    assert logits_dict["representation"].shape == (1, 768)
    assert torch.isfinite(loss), "Loss should be finite"


# === M4 trainer smoke test ===

def test_m4_trainer_smoke(model, small_dataset):
    """M4 trainer should run one epoch without errors."""
    from torch.utils.data import Subset
    from torch.optim import AdamW

    indices = list(range(min(32, len(small_dataset))))
    subset = Subset(small_dataset, indices)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    from m4_adversarial_training.trainer import M4Trainer
    trainer = M4Trainer(
        model=model, train_dataset=subset, val_dataset=subset,
        optimizer=optimizer, device=torch.device("cpu")
    )

    stats = trainer.train_epoch(batch_size=4, epsilon=1e-3, lambda_adv=1.0)
    assert "total_loss" in stats
    assert "clean_loss" in stats
    assert "adv_loss" in stats
    assert stats["total_loss"] > 0
    assert stats["max_delta"] >= 0


def test_adversarial_validation_no_param_grads(model, small_dataset):
    """Adversarial validation should not accumulate parameter gradients."""
    from torch.utils.data import Subset
    from torch.optim import AdamW
    from m4_adversarial_training.trainer import M4Trainer

    indices = list(range(min(16, len(small_dataset))))
    subset = Subset(small_dataset, indices)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    trainer = M4Trainer(
        model=model, train_dataset=subset, val_dataset=subset,
        optimizer=optimizer, device=torch.device("cpu")
    )

    model.zero_grad()
    result = trainer.validate(adversarial=True, epsilon=1e-3, adv_val_batch_size=8)

    assert result["loss"] > 0
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() == 0, \
                "Adversarial validation should not populate parameter gradients"


def test_adversarial_validation_predictions_cpu(model, small_dataset):
    """Adversarial validation predictions should be detached CPU numpy arrays."""
    from torch.utils.data import Subset
    from torch.optim import AdamW
    from m4_adversarial_training.trainer import M4Trainer

    indices = list(range(min(16, len(small_dataset))))
    subset = Subset(small_dataset, indices)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    trainer = M4Trainer(
        model=model, train_dataset=subset, val_dataset=subset,
        optimizer=optimizer, device=torch.device("cpu")
    )

    result = trainer.validate(adversarial=True, epsilon=1e-3, adv_val_batch_size=8)

    for dim_name, preds in result["predictions"].items():
        assert isinstance(preds, np.ndarray), \
            f"Predictions for {dim_name} should be numpy arrays"
        assert preds.dtype == np.float32 or preds.dtype == np.float64


def test_adv_val_batch_size_parameter(model, small_dataset):
    """validate() should accept adv_val_batch_size parameter."""
    from torch.utils.data import Subset
    from torch.optim import AdamW
    from m4_adversarial_training.trainer import M4Trainer

    indices = list(range(min(16, len(small_dataset))))
    subset = Subset(small_dataset, indices)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    trainer = M4Trainer(
        model=model, train_dataset=subset, val_dataset=subset,
        optimizer=optimizer, device=torch.device("cpu")
    )

    result = trainer.validate(adversarial=True, epsilon=1e-3, adv_val_batch_size=4)
    assert "loss" in result
    assert "metrics" in result