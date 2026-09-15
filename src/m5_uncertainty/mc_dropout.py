"""M5 -- MC-Dropout-Based Uncertainty Quantification & Calibration.

Performs stochastic inference with dropout active on the M4 model,
computes predictive mean/variance/entropy/confidence, and measures
Expected Calibration Error (ECE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_dropout_training(model):
    """Set all Dropout modules to training mode while keeping the rest in eval mode.

    This is critical for MC-Dropout: model.eval() disables dropout,
    but we need dropout to remain stochastic during inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_inference(model, input_ids, attention_mask, mc_passes=20):
    """Run MC-Dropout stochastic inference for a single batch.

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    input_ids : torch.Tensor [B, seq_len]
    attention_mask : torch.Tensor [B, seq_len]
    mc_passes : int

    Returns
    -------
    predictions : list of dicts
    """
    set_dropout_training(model)
    predictions = []
    with torch.no_grad():
        for _ in range(mc_passes):
            outputs = model(input_ids, attention_mask)
            predictions.append(outputs)
    return predictions


def convert_to_probabilities(predictions):
    """Convert logits to probabilities for each task.

    Categorical tasks (sensitivity, intent, disclosure_scope): softmax
    Multi-label tasks (entity_tags, threat_content): sigmoid
    """
    prob_lists = {}
    for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]:
        logits_key = f"{dim}_logits"
        if dim in ["sensitivity", "intent", "disclosure_scope"]:
            probs = [torch.softmax(p[logits_key], dim=-1) for p in predictions]
        else:
            probs = [torch.sigmoid(p[logits_key]) for p in predictions]
        prob_lists[dim] = probs
    return prob_lists


def compute_predictive_mean(prob_lists):
    """Compute predictive mean: p_mean = (1/T) * sum_t p_t."""
    return {dim: torch.stack(probs).mean(dim=0) for dim, probs in prob_lists.items()}


def compute_predictive_variance(prob_lists, p_mean):
    """Compute predictive variance: (1/T) * sum_t (p_t - p_mean)^2."""
    return {dim: torch.stack([(p - p_mean[dim]) ** 2 for p in probs]).mean(dim=0)
            for dim, probs in prob_lists.items()}


def compute_categorical_entropy(p_mean):
    """Compute predictive entropy for categorical tasks."""
    eps = 1e-8
    return -torch.sum(p_mean * torch.log(p_mean + eps), dim=-1)


def compute_binary_entropy(p):
    """Compute binary entropy for each label in multi-label tasks."""
    eps = 1e-8
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)


def compute_confidence(p_mean, dim):
    """Compute confidence: max(p_mean) for categorical, mean(max(p,1-p)) for multi-label."""
    if dim in ["sensitivity", "intent", "disclosure_scope"]:
        return p_mean.max(dim=-1).values
    else:
        return torch.max(p_mean, 1 - p_mean).mean(dim=-1)


def compute_uncertainty_summary(p_mean, p_variance):
    """Compute aggregate uncertainty metrics for all tasks."""
    summary = {}
    for dim in p_mean:
        if dim in ["sensitivity", "intent", "disclosure_scope"]:
            entropy = compute_categorical_entropy(p_mean[dim])
        else:
            entropy = compute_binary_entropy(p_mean[dim])

        summary[dim] = {
            "mean_entropy": float(entropy.mean().item()),
            "mean_variance": float(p_variance[dim].mean().item()),
            "mean_confidence": float(compute_confidence(p_mean[dim], dim).mean().item()),
        }
    return summary


def run_m5_inference(model, dataset, mc_passes=20, batch_size=32, device=None):
    """Run MC-Dropout inference over the full validation dataset.

    Returns
    -------
    results : list of dicts, one per record
    uncertainty_summary : dict with aggregate uncertainty metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    set_dropout_training(model)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_prob_lists = {dim: [] for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]}
    all_record_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MC-Dropout Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            record_ids = batch["record_id"]
            all_record_ids.extend(record_ids)

            predictions = []
            for _ in range(mc_passes):
                outputs = model(input_ids, attention_mask)
                predictions.append(outputs)

            prob_lists = convert_to_probabilities(predictions)
            for dim in prob_lists:
                all_prob_lists[dim].append(prob_lists[dim])

    p_mean = {}
    p_variance = {}
    for dim in all_prob_lists:
        all_probs = torch.cat(all_prob_lists[dim], dim=0)
        p_mean[dim] = all_probs.mean(dim=0)
        p_variance[dim] = ((all_probs - p_mean[dim]) ** 2).mean(dim=0)

    uncertainty_summary = compute_uncertainty_summary(p_mean, p_variance)

    results = []
    for i, rid in enumerate(all_record_ids):
        record_result = {"record_id": rid}
        for dim in p_mean:
            record_result[f"{dim}_mean"] = p_mean[dim][i].tolist()
            record_result[f"{dim}_variance"] = p_variance[dim][i].item()
            if dim in ["sensitivity", "intent", "disclosure_scope"]:
                record_result[f"{dim}_entropy"] = float(compute_categorical_entropy(p_mean[dim][i].unsqueeze(0)).item())
            else:
                record_result[f"{dim}_entropy"] = float(compute_binary_entropy(p_mean[dim][i]).mean().item())
            record_result[f"{dim}_confidence"] = float(compute_confidence(p_mean[dim][i].unsqueeze(0), dim).item())
        results.append(record_result)

    return results, uncertainty_summary