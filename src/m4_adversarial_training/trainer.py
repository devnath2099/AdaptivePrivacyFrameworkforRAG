"""M4 trainer: adversarial fine-tuning of M3 model using FGSM.

Trains the DeBERTaMultiTaskModel using FGSM adversarial examples
in the token-embedding space. Loads from M3 checkpoint and
saves best robust checkpoint.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from m3_privacy_prediction.losses import compute_total_loss
from m3_privacy_prediction.metrics import compute_all_metrics
from m4_adversarial_training.fgsm import adversarial_forward


class M4Trainer:
    """Adversarial fine-tunes the M3 model using FGSM.

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    train_dataset : M3Dataset
    val_dataset : M3Dataset
    optimizer : torch.optim.Optimizer
    device : torch.device
    """

    def __init__(self, model, train_dataset, val_dataset, optimizer, device):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.device = device
        self.history = []

    def _make_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = {k: v.to(self.device) for k, v in batch["soft_targets"].items()}
        return input_ids, attention_mask, targets

    def train_epoch(self, batch_size=32, epsilon=1e-3, lambda_adv=1.0):
        """Train one epoch with FGSM adversarial examples."""
        self.model.train()
        dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        epoch_stats = []
        for batch in tqdm(dataloader, desc="Adversarial Training"):
            input_ids, attention_mask, targets = self._make_batch(batch)

            total_loss, clean_loss_val, adv_loss_val, task_losses, delta, max_delta = (
                adversarial_forward(
                    self.model, input_ids, attention_mask, targets,
                    compute_total_loss, epsilon=epsilon, lambda_adv=lambda_adv
                )
            )

            self.optimizer.step()

            epoch_stats.append({
                "total_loss": total_loss.item(),
                "clean_loss": clean_loss_val,
                "adv_loss": adv_loss_val,
                "max_delta": max_delta,
                **{f"{k}_loss": v.item() for k, v in task_losses.items()},
            })

        avg = {k: sum(e[k] for e in epoch_stats) / len(epoch_stats)
               for k in epoch_stats[0]}
        return avg

    def validate(self, adversarial=False, epsilon=1e-3):
        """Validate the model on clean or adversarial inputs.

        Parameters
        ----------
        adversarial : bool — if True, apply FGSM to validation inputs
        epsilon : float — FGSM epsilon if adversarial

        Returns
        -------
        dict with loss, metrics, predictions
        """
        self.model.eval()
        dataloader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        all_preds = {}
        all_targets = {}
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validation (adversarial)" if adversarial else "Validation"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["soft_targets"].items()}

            if adversarial:
                total_loss_batch, _, _, _, _, _ = adversarial_forward(
                    self.model, input_ids, attention_mask, targets,
                    compute_total_loss, epsilon=epsilon
                )
                outputs = self.model(input_ids, attention_mask)
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    total_loss_batch, _ = compute_total_loss(outputs, targets)

                total_loss += total_loss_batch.item()
                num_batches += 1

                for dim_name in targets:
                    logits_key = f"{dim_name}_logits"
                    if dim_name not in all_preds:
                        all_preds[dim_name] = []
                        all_targets[dim_name] = []
                    all_preds[dim_name].append(outputs[logits_key].cpu().numpy())
                    all_targets[dim_name].append(targets[dim_name].cpu().numpy())

        for dim_name in all_preds:
            all_preds[dim_name] = np.concatenate(all_preds[dim_name], axis=0)
            all_targets[dim_name] = np.concatenate(all_targets[dim_name], axis=0)

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_all_metrics(all_preds, all_targets)
        return {"loss": avg_loss, "metrics": metrics, "predictions": all_preds}

    def train(self, epochs=3, batch_size=32, save_dir="outputs/m4",
              learning_rate=1e-5, epsilon=1e-3, lambda_adv=1.0):
        """Train M4 with FGSM adversarial fine-tuning.

        Parameters
        ----------
        epochs : int
        batch_size : int
        save_dir : str
        learning_rate : float
        epsilon : float — FGSM perturbation magnitude
        lambda_adv : float — weight for adversarial loss
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_state = None
        best_metrics = None
        tracemalloc.start()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_stats = self.train_epoch(batch_size=batch_size, epsilon=epsilon, lambda_adv=lambda_adv)
            val_result = self.validate(adversarial=False)
            adv_val_result = self.validate(adversarial=True, epsilon=epsilon)

            epoch_duration = time.time() - epoch_start

            if torch.cuda.is_available():
                peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
            else:
                peak_gpu_mb = 0.0

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            peak_cpu_mb = peak / 1024**2

            lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(train_stats["total_loss"], 4),
                "clean_loss": round(train_stats["clean_loss"], 4),
                "adv_loss": round(train_stats["adv_loss"], 4),
                "max_delta": round(train_stats["max_delta"], 6),
                "val_loss": round(val_result["loss"], 4),
                "adv_val_loss": round(adv_val_result["loss"], 4),
                "learning_rate": lr,
                "epoch_duration_seconds": round(epoch_duration, 2),
                "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
                "peak_cpu_memory_mb": round(peak_cpu_mb, 2),
            }
            self.history.append(epoch_record)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train total loss: {epoch_record['train_loss']:.4f}")
            print(f"  Clean train loss: {epoch_record['clean_loss']:.4f}")
            print(f"  Adv train loss: {epoch_record['adv_loss']:.4f}")
            print(f"  Max delta: {epoch_record['max_delta']:.6f}")
            print(f"  Val loss: {epoch_record['val_loss']:.4f}")
            print(f"  Adv Val loss: {epoch_record['adv_val_loss']:.4f}")
            print(f"  Duration: {epoch_duration:.1f}s")
            print(f"  Peak GPU memory: {peak_gpu_mb:.1f} MB")
            print(f"{'='*60}")

            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_metrics = val_result["metrics"]
                torch.save(best_state, save_dir / "best_model.pt")
                print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

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
            "epsilon": epsilon,
            "lambda_adv": lambda_adv,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "label_dims": {"sensitivity": 3, "intent": 5, "disclosure_scope": 2,
                           "entity_tags": 4, "threat_content": 3},
        }
        with open(save_dir / "m4_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        tracemalloc.stop()

        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
        }

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)