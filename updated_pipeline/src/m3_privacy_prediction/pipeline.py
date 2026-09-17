"""M3 pipeline: orchestrates training from M2 outputs.

Loads M1 splits, aligns M2 soft labels, builds the DeBERTa model,
trains with multi-task loss, validates, and saves checkpoints.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from m1_data_integration.config import load_config
from m3_privacy_prediction.dataset import build_m3_datasets, validate_alignment
from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.trainer import M3Trainer


def run_m3(cfg, dev_mode=False, dev_train_size=100, dev_val_size=50):
    """Run the M3 training pipeline.

    Parameters
    ----------
    cfg : ReviewConfig
        Project configuration.
    dev_mode : bool
        If True, use small subsets for smoke testing.
    dev_train_size : int
        Number of training records in dev mode.
    dev_val_size : int
        Number of validation records in dev mode.

    Returns
    -------
    result : dict
        Training results including history, best metrics, and alignment info.
    """
    # Build datasets
    datasets = build_m3_datasets(cfg)

    # Validate alignment on full datasets before Subset wrapping
    alignment = validate_alignment(datasets["train"], datasets["val"], datasets["test"])
    print(f"Alignment: {alignment}")

    if dev_mode:
        from torch.utils.data import Subset
        datasets["train"] = Subset(datasets["train"], range(dev_train_size))
        datasets["val"] = Subset(datasets["val"], range(dev_val_size))

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    m3_cfg = cfg.raw.get("m3", {})
    model_name = m3_cfg.get("model_name", "microsoft/deberta-base")
    label_dims = {
        "sensitivity": 3,
        "intent": 5,
        "disclosure_scope": 2,
        "entity_tags": 4,
        "threat_content": 3,
    }
    model = DeBERTaMultiTaskModel(
        model_name=model_name,
        label_dims=label_dims,
        dropout_rate=m3_cfg.get("dropout_rate", 0.1),
    )

    lr = float(m3_cfg.get("learning_rate", 1e-5))
    weight_decay = float(m3_cfg.get("weight_decay", 0.01))
    dropout_rate = float(m3_cfg.get("dropout_rate", 0.1))
    task_weights = m3_cfg.get("task_weights", {
        "sensitivity": 1.0, "intent": 1.0, "disclosure_scope": 1.0,
        "entity_tags": 1.0, "threat_content": 1.0,
    })
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    if hasattr(train_dataset, "dataset"):
        train_dataset = train_dataset.dataset
    if hasattr(val_dataset, "dataset"):
        val_dataset = val_dataset.dataset

    trainer = M3Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        device=device,
    )

    epochs = m3_cfg.get("epochs", 1)
    batch_size = m3_cfg.get("batch_size", 32)

    if dev_mode:
        epochs = 1
        batch_size = 4

    result = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        save_dir=cfg.resolve_output("m3_dir"),
    )

    return {
        "alignment": alignment,
        "train_records": alignment["train"],
        "val_records": alignment["val"],
        "test_records": alignment["test"],
        "history": result["history"],
        "best_val_loss": result["best_val_loss"],
        "best_metrics": result["best_metrics"],
    }


def main():
    cfg = load_config("configs/review1.yaml")
    result = run_m3(cfg, dev_mode=True)
    print("\n=== M3 Training Complete ===")
    print(f"Train records: {result['train_records']}")
    print(f"Val records: {result['val_records']}")
    print(f"Test records: {result['test_records']}")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    for dim, metrics in result["best_metrics"].items():
        print(f"  {dim}: {metrics}")


if __name__ == "__main__":
    main()