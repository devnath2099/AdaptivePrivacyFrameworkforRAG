"""M5 -- MC-Dropout-Based Uncertainty Quantification.

Stochastic inference with dropout active on the M4 model,
computes predictive mean/variance/entropy/confidence.

Temperature calibration is in temperature_scaling.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_dropout_training(model):
    """Set all Dropout modules to training mode while keeping the rest in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_inference(model, input_ids, attention_mask, mc_passes=20):
    """Run MC-Dropout stochastic inference for a single batch."""
    if mc_passes < 1:
        raise ValueError("mc_passes must be at least 1")
    set_dropout_training(model)
    predictions = []
    with torch.no_grad():
        for _ in range(mc_passes):
            outputs = model(input_ids, attention_mask)
            predictions.append(outputs)
    return predictions


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


def run_m5_inference(model, dataset, mc_passes=20, batch_size=32, device=None,
                     temperature=1.0):
    """Run MC-Dropout inference over the full validation dataset.

    Parameters
    ----------
    temperature : float or dict
        If float: shared temperature for all heads.
        If dict: per-head temps for categorical + '__shared__' key for multi-label.
    """
    if mc_passes < 1:
        raise ValueError("mc_passes must be at least 1")
    if len(dataset) == 0:
        raise ValueError("M5 inference requires a non-empty dataset")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    set_dropout_training(model)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_means = {dim: [] for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]}
    all_variances = {dim: [] for dim in all_means}
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

            prob_lists = _probabilities_from_logits(predictions, temperature)
            batch_mean = compute_predictive_mean(prob_lists)
            batch_variance = compute_predictive_variance(prob_lists, batch_mean)
            for dim in prob_lists:
                all_means[dim].append(batch_mean[dim].cpu())
                all_variances[dim].append(batch_variance[dim].cpu())
            del predictions, outputs, prob_lists, batch_mean, batch_variance

    p_mean = {}
    p_variance = {}
    for dim in all_means:
        p_mean[dim] = torch.cat(all_means[dim], dim=0)
        p_variance[dim] = torch.cat(all_variances[dim], dim=0)

    uncertainty_summary = compute_uncertainty_summary(p_mean, p_variance)

    results = []
    for i, rid in enumerate(all_record_ids):
        record_result = {"record_id": rid}
        for dim in p_mean:
            record_result[f"{dim}_mean"] = p_mean[dim][i].tolist()
            record_result[f"{dim}_variance"] = p_variance[dim][i].mean().item()
            if dim in ["sensitivity", "intent", "disclosure_scope"]:
                record_result[f"{dim}_entropy"] = float(compute_categorical_entropy(p_mean[dim][i].unsqueeze(0)).item())
            else:
                record_result[f"{dim}_entropy"] = float(compute_binary_entropy(p_mean[dim][i]).mean().item())
            record_result[f"{dim}_confidence"] = float(compute_confidence(p_mean[dim][i].unsqueeze(0), dim).item())
        results.append(record_result)

    return results, uncertainty_summary


def _probabilities_from_logits(predictions, temperature):
    """Convert logits to probabilities with optional temperature scaling.

    Parameters
    ----------
    predictions : list of dicts
    temperature : float or dict

    Returns
    -------
    prob_lists : dict
    """
    prob_lists = {}
    categorical_dims = ["sensitivity", "intent", "disclosure_scope"]
    multi_label_dims = ["entity_tags", "threat_content"]

    if isinstance(temperature, dict):
        ml_temp = temperature.get("__shared__", 1.0)
        for dim in categorical_dims:
            T = temperature.get(dim, 1.0)
            T = max(T, 1e-6)
            logits_list = [p[f"{dim}_logits"] / T for p in predictions]
            prob_lists[dim] = [torch.softmax(l, dim=-1) for l in logits_list]
        for dim in multi_label_dims:
            T = max(ml_temp, 1e-6)
            logits_list = [p[f"{dim}_logits"] / T for p in predictions]
            prob_lists[dim] = [torch.sigmoid(l) for l in logits_list]
    else:
        T = max(temperature, 1e-6)
        for dim in categorical_dims:
            logits_list = [p[f"{dim}_logits"] / T for p in predictions]
            prob_lists[dim] = [torch.softmax(l, dim=-1) for l in logits_list]
        for dim in multi_label_dims:
            logits_list = [p[f"{dim}_logits"] / T for p in predictions]
            prob_lists[dim] = [torch.sigmoid(l) for l in logits_list]

    return prob_lists