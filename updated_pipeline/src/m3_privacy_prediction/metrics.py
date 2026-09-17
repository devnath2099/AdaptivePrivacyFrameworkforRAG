"""M3 metrics for multi-task privacy prediction evaluation.

For mutually-exclusive dimensions: accuracy, macro F1, weighted F1.
For independent multi-label dimensions: micro F1, macro F1, per-category F1.
ROC-AUC and PR-AUC are reported when valid.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_single_label(dim_name, logits, targets, threshold=None):
    """Evaluate a mutually-exclusive single-label dimension.

    logits: np.ndarray [N, num_classes]
    targets: np.ndarray [N, num_classes] (probabilistic soft labels)
    Returns dict with accuracy, macro_f1, weighted_f1, roc_auc, pr_auc.
    """
    preds = logits.argmax(axis=1)
    hard_targets = targets.argmax(axis=1)
    num_classes = logits.shape[1]

    result = {
        "accuracy": float(accuracy_score(hard_targets, preds)),
        "macro_f1": float(f1_score(hard_targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(hard_targets, preds, average="weighted", zero_division=0)),
    }

    # ROC-AUC (one-vs-rest) — safe if only one class present
    try:
        if num_classes == 2:
            result["roc_auc"] = float(roc_auc_score(hard_targets, logits[:, 1]))
            result["pr_auc"] = float(
                average_precision_score(hard_targets, logits[:, 1])
            )
        else:
            result["roc_auc"] = float(
                roc_auc_score(hard_targets, logits, multi_class="ovr", average="macro")
            )
            result["pr_auc"] = float(
                average_precision_score(hard_targets, logits, average="macro")
            )
    except ValueError:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def evaluate_multi_label(dim_name, logits, targets, threshold=0.5):
    """Evaluate an independent multi-label dimension.

    logits: np.ndarray [N, num_labels]
    targets: np.ndarray [N, num_labels] (probabilistic soft labels)
    threshold: float — binarize logits at this value.
    Returns dict with micro_f1, macro_f1, per_category_f1.
    """
    preds = (logits >= threshold).astype(int)
    hard_targets = (targets >= threshold).astype(int)
    num_labels = logits.shape[1]

    try:
        micro_f1 = float(f1_score(hard_targets, preds, average="micro", zero_division=0))
        macro_f1 = float(f1_score(hard_targets, preds, average="macro", zero_division=0))
    except ValueError:
        micro_f1 = 0.0
        macro_f1 = 0.0

    per_category = {}
    for i in range(num_labels):
        try:
            f1 = float(f1_score(
                hard_targets[:, i], preds[:, i], average="binary", zero_division=0
            ))
        except ValueError:
            f1 = 0.0
        per_category[f"category_{i}"] = f1

    result = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_category_f1": per_category,
        "threshold": threshold,
    }

    return result


def evaluate_dimension(dim_name, logits, targets, threshold=0.5):
    """Evaluate a single dimension, dispatching to the correct evaluator.

    logits: np.ndarray [N, num_classes]
    targets: np.ndarray [N, num_classes] (probabilistic soft labels)
    threshold: float — used for multi-label binarization.
    Returns dict with metrics.
    """
    MUTUALLY_EXCLUSIVE = {"sensitivity", "intent", "disclosure_scope"}
    INDEPENDENT_BINARY = {"entity_tags", "threat_content"}

    if dim_name in MUTUALLY_EXCLUSIVE:
        return evaluate_single_label(dim_name, logits, targets)
    elif dim_name in INDEPENDENT_BINARY:
        return evaluate_multi_label(dim_name, logits, targets, threshold)
    else:
        raise ValueError(f"Unknown dimension: {dim_name}")


def compute_all_metrics(predictions, targets, threshold=0.5):
    """Compute metrics for all five dimensions.

    predictions: dict {dim_name: np.ndarray [N, num_classes]}
    targets: dict {dim_name: np.ndarray [N, num_classes]}
    threshold: float — for multi-label binarization.
    Returns dict {dim_name: metric_dict}.
    """
    results = {}
    for dim_name in targets:
        results[dim_name] = evaluate_dimension(
            dim_name, predictions[dim_name], targets[dim_name], threshold
        )
    return results
