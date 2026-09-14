"""M3 trainer: single-task multi-task training loop with validation and checkpointing.

Trains the DeBERTaMultiTaskModel using M2 soft labels as targets.
Selects the best checkpoint by validation loss.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss
from m3_privacy_prediction.metrics import compute_all_metrics


class M3Trainer:
    """Trains the M3 DeBERTa multi-task model.

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    train_dataset : M3Dataset
    val_dataset : M3Dataset
    optimizer : torch.optim.Optimizer
    device : torch.device
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        optimizer,
        device,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.device = device
        self.history = []

    def train_epoch(self, batch_size=32):
        """Train one epoch. Returns dict with total_loss and per-task losses."""
        self.model.train()
        dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        epoch_losses = []
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["soft_targets"].items()}

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            total_loss, task_losses = compute_total_loss(outputs, targets)
            total_loss.backward()
            self.optimizer.step()

            epoch_losses.append(
                {
                    "total_loss": total_loss.item(),
                    **{f"{k}_loss": v.item() for k, v in task_losses.items()},
                }
            )

        avg = {k: sum(e[k] for e in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
        return avg

    def validate(self):
        """Validate the model. Returns dict with loss and metrics."""
        self.model.eval()
        dataloader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        all_preds = {}
        all_targets = {}
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch["soft_targets"].items()}

                outputs = self.model(input_ids, attention_mask)
                loss, task_losses = compute_total_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

                for dim_name in targets:
                    logits_key = f"{dim_name}_logits"
                    if dim_name not in all_preds:
                        all_preds[dim_name] = []
                        all_targets[dim_name] = []
                    all_preds[dim_name].append(outputs[logits_key].cpu().numpy())
                    all_targets[dim_name].append(targets[dim_name].cpu().numpy())

        # Concatenate all batches
        for dim_name in all_preds:
            all_preds[dim_name] = np.concatenate(all_preds[dim_name], axis=0)
            all_targets[dim_name] = np.concatenate(all_targets[dim_name], axis=0)

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_all_metrics(all_preds, all_targets)

        return {"loss": avg_loss, "metrics": metrics, "predictions": all_preds}

    def train(
        self,
        epochs=1,
        batch_size=32,
        save_dir="outputs/m3",
        save_best_every_epoch=False,
    ):
        """Train the model for the specified number of epochs.

        Saves the best checkpoint by validation loss.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            train_stats = self.train_epoch(batch_size=batch_size)
            val_result = self.validate()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_stats["total_loss"],
                "val_loss": val_result["loss"],
                "train_task_losses": train_stats,
                "val_metrics": val_result["metrics"],
            }
            self.history.append(epoch_record)

            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_metrics = val_result["metrics"]

        # Save best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, save_dir / "best_model.pt")

        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        # Save validation metrics
        with open(save_dir / "validation_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2, default=str)

        # Save config metadata
        config = {
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        with open(save_dir / "m3_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
        }

    def save_checkpoint(self, path):
        """Save model state dict to path."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Load model state dict from path."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
