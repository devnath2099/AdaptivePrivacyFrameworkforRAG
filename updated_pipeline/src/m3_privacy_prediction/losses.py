"""M3 multi-task loss functions.

For mutually-exclusive dimensions (sensitivity, intent, disclosure_scope),
use soft cross-entropy against probabilistic targets:
    L = -sum_k y_k * log(p_k)

For independent multi-label dimensions (entity_tags, threat_content),
use binary cross-entropy with logits against each independent target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Dimensions that are mutually exclusive (single-label classification)
MUTUALLY_EXCLUSIVE = {"sensitivity", "intent", "disclosure_scope"}

# Dimensions that are independent binary (multi-label)
INDEPENDENT_BINARY = {"entity_tags", "threat_content"}


def soft_cross_entropy(logits, targets):
    """Soft cross-entropy for mutually-exclusive dimensions.

    logits: [B, num_classes]
    targets: [B, num_classes] — probabilistic soft labels (sum to ~1)
    Returns scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss


def binary_cross_entropy_with_logits(logits, targets):
    """Binary cross-entropy with logits for independent multi-label.

    logits: [B, num_labels]
    targets: [B, num_labels] — probabilistic soft labels in [0, 1]
    Returns scalar loss averaged over all labels and samples.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def compute_task_loss(logits, targets, dimension):
    """Compute loss for a single task dimension.

    Parameters
    ----------
    logits : torch.Tensor
        Model logits for this dimension.
    targets : torch.Tensor
        Probabilistic soft targets from M2.
    dimension : str
        Dimension name (determines loss type).

    Returns
    -------
    torch.Tensor
        Scalar loss for this task.
    """
    if dimension in MUTUALLY_EXCLUSIVE:
        return soft_cross_entropy(logits, targets)
    elif dimension in INDEPENDENT_BINARY:
        return binary_cross_entropy_with_logits(logits, targets)
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def compute_total_loss(outputs, targets, task_weights=None):
    """Compute weighted total multi-task loss.

    Parameters
    ----------
    outputs : dict
        Model output dict with keys like 'sensitivity_logits', etc.
    targets : dict
        Dict mapping dimension name to target tensor.
    task_weights : dict, optional
        Weight per dimension. Defaults to 1.0 for all.

    Returns
    -------
    total_loss : torch.Tensor
    task_losses : dict
        Individual loss per dimension.
    """
    if task_weights is None:
        task_weights = {
            "sensitivity": 1.0,
            "intent": 1.0,
            "disclosure_scope": 1.0,
            "entity_tags": 1.0,
            "threat_content": 1.0,
        }

    task_losses = {}
    for dim_name in targets:
        logits_key = f"{dim_name}_logits"
        if logits_key in outputs and dim_name in targets:
            task_losses[dim_name] = compute_task_loss(
                outputs[logits_key], targets[dim_name], dim_name
            ) * task_weights[dim_name]

    total_loss = sum(task_losses.values()) if task_losses else torch.tensor(0.0)
    return total_loss, task_losses
