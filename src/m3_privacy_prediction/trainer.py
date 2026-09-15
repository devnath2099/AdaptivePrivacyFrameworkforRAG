"""M3 trainer: single-task multi-task training loop with validation and checkpointing.

Trains the DeBERTaMultiTaskModel using M2 soft labels as targets.
Selects the best checkpoint by validation loss.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
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
        epochs=5,
        batch_size=32,
        save_dir="outputs/m3/smoke",
        learning_rate=1e-5,
    ):
        """Train for specified epochs with full reporting.

        Saves best checkpoint to save_dir based on validation loss.
        Reports per-epoch metrics, duration, and GPU memory.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_state = None
        best_metrics = None
        tracemalloc.start()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_stats = self.train_epoch(batch_size=batch_size)
            val_result = self.validate()

            epoch_duration = time.time() - epoch_start

            # Peak GPU memory
            if torch.cuda.is_available():
                peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
            else:
                peak_gpu_mb = 0.0

            # CPU memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            peak_cpu_mb = peak / 1024**2

            # Get learning rate from optimizer
            lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(train_stats["total_loss"], 4),
                "train_task_losses": {k: round(v, 4) for k, v in train_stats.items()},
                "val_loss": round(val_result["loss"], 4),
                "val_metrics": self._serialize_metrics(val_result["metrics"]),
                "learning_rate": lr,
                "epoch_duration_seconds": round(epoch_duration, 2),
                "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
                "peak_cpu_memory_mb": round(peak_cpu_mb, 2),
            }
            self.history.append(epoch_record)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train loss: {epoch_record['train_loss']:.4f}")
            for k, v in train_stats.items():
                if k != "total_loss":
                    print(f"  Train {k}: {v:.4f}")
            print(f"  Val loss: {epoch_record['val_loss']:.4f}")
            print(f"  Duration: {epoch_duration:.1f}s")
            print(f"  Peak GPU memory: {peak_gpu_mb:.1f} MB")
            print(f"{'='*60}")

            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_metrics = val_result["metrics"]
                torch.save(best_state, save_dir / "best_model.pt")
                print(f"  → New best model saved (val_loss={best_val_loss:.4f})")

        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        # Save best validation metrics
        if best_metrics is not None:
            with open(save_dir / "validation_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2, default=str)

        # Save config metadata
        config = {
            "best_val_loss": round(best_val_loss, 4),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "label_dims": {"sensitivity": 3, "intent": 5, "disclosure_scope": 2,
                           "entity_tags": 4, "threat_content": 3},
        }
        with open(save_dir / "m3_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Save best model state dict
        if best_state is not None:
            torch.save(best_state, save_dir / "best_model.pt")

        print(f"\nBest val loss: {best_val_loss:.4f}")
        tracemalloc.stop()

        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
        }

    @staticmethod
    def _serialize_metrics(metrics):
        result = {}
        for dim, m in metrics.items():
            if isinstance(m, dict):
                result[dim] = {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                              for k, v in m.items()}
        return result

    def save_checkpoint(self, path):
        """Save model state dict to path."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Load model state dict from path."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
