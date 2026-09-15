"""M5 -- Expected Calibration Error (ECE) and calibration metrics.

Computes ECE for categorical and multi-label tasks using
frozen M2 soft targets as reference for evaluation only.
"""
from __future__ import annotations

import torch
import numpy as np


def compute_ece(confidences, accuracies, n_bins=10):
    """Compute Expected Calibration Error using uniform bins.

    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Parameters
    ----------
    confidences : np.ndarray [N]
    accuracies : np.ndarray [N] (0 or 1)
    n_bins : int

    Returns
    -------
    ece : float
    bin_stats : list of dicts with count, avg_confidence, accuracy, gap
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    n = len(confidences)

    bin_size = 1.0 / n_bins
    bin_stats = []
    ece = 0.0

    for b in range(n_bins):
        low = b * bin_size
        high = (b + 1) * bin_size

        # Last bin includes the upper edge
        if b == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        bin_count = int(mask.sum())
        if bin_count == 0:
            bin_stats.append({"count": 0, "avg_confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (bin_count / n) * gap

        bin_stats.append({
            "count": bin_count,
            "avg_confidence": float(bin_conf),
            "accuracy": float(bin_acc),
            "gap": float(gap),
        })

    return float(ece), bin_stats


def compute_categorical_ece(predictions, targets, n_bins=10):
    """Compute ECE for categorical tasks (sensitivity, intent, disclosure_scope).

    predicted class = argmax(predictive_mean)
    confidence = max(predictive_mean)
    reference = argmax(frozen M2 soft target)
    """
    all_confidences = []
    all_accuracies = []

    for i in range(len(predictions)):
        pred_mean = predictions[i]  # [num_classes]
        target = targets[i]  # [num_classes]

        pred_class = int(np.argmax(pred_mean))
        confidence = float(pred_mean[pred_class])
        ref_class = int(np.argmax(target))
        correct = 1 if pred_class == ref_class else 0

        all_confidences.append(confidence)
        all_accuracies.append(correct)

    ece, bin_stats = compute_ece(np.array(all_confidences), np.array(all_accuracies), n_bins)
    return ece, bin_stats


def compute_multilabel_ece(predictions, targets, threshold=0.5, n_bins=10):
    """Compute per-category ECE for multi-label tasks (entity_tags, threat_content).

    For each category independently:
    predicted positive if p >= 0.5
    confidence = p if predicted positive, 1-p if predicted negative
    reference = 1 if target >= threshold else 0
    """
    n_categories = predictions.shape[1]
    all_ece = {}
    all_bin_stats = {}
    macro_ece = 0.0

    for cat in range(n_categories):
        all_confidences = []
        all_accuracies = []

        for i in range(len(predictions)):
            p = float(predictions[i, cat])
            target = float(targets[i, cat])

            pred_positive = 1 if p >= threshold else 0
            ref_positive = 1 if target >= threshold else 0

            confidence = p if pred_positive else (1 - p)
            correct = 1 if pred_positive == ref_positive else 0

            all_confidences.append(confidence)
            all_accuracies.append(correct)

        cat_ece, cat_bin_stats = compute_ece(
            np.array(all_confidences), np.array(all_accuracies), n_bins
        )
        cat_name = f"category_{cat}"
        all_ece[cat_name] = cat_ece
        all_bin_stats[cat_name] = cat_bin_stats
        macro_ece += cat_ece

    macro_ece /= n_categories
    all_ece["macro_ece"] = macro_ece
    all_ece["per_category_ece"] = all_ece
    all_bin_stats["macro"] = all_bin_stats

    return macro_ece, all_bin_stats, all_ece


def compute_correct_vs_incorrect_uncertainty(p_mean, p_variance, targets, threshold=0.5):
    """Separate uncertainty metrics for correct vs incorrect predictions.

    For categorical tasks: compare argmax(p_mean) vs argmax(target).
    For multi-label tasks: compare each category independently.
    """
    results = {}

    # Categorical tasks
    for dim in ["sensitivity", "intent", "disclosure_scope"]:
        preds = p_mean[dim]  # [N, num_classes]
        targets_t = targets[dim]  # [N, num_classes]
        pred_classes = np.argmax(preds, axis=1)
        ref_classes = np.argmax(targets_t, axis=1)
        correct = pred_classes == ref_classes

        results[dim] = {
            "correct": {
                "mean_entropy": float(-np.sum(preds[correct] * np.log(preds[correct] + 1e-8), axis=1).mean()),
                "mean_variance": float(p_variance[dim][correct].mean()),
                "mean_confidence": float(preds[correct].max(axis=1).mean()),
            },
            "incorrect": {
                "mean_entropy": float(-np.sum(preds[~correct] * np.log(preds[~correct] + 1e-8), axis=1).mean()),
                "mean_variance": float(p_variance[dim][~correct].mean()),
                "mean_confidence": float(preds[~correct].max(axis=1).mean()),
            },
        }

    # Multi-label tasks
    for dim in ["entity_tags", "threat_content"]:
        preds = p_mean[dim]  # [N, num_labels]
        targets_t = targets[dim]  # [N, num_labels]
        pred_positive = (preds >= threshold).astype(int)
        ref_positive = (targets_t >= threshold).astype(int)

        # Label-level correctness
        correct = (pred_positive == ref_positive).all(axis=1)

        # Compute per-label entropy and variance
        entropy = -preds * np.log(preds + 1e-8) - (1 - preds) * np.log(1 - preds + 1e-8)
        mean_entropy_per_sample = entropy.mean(axis=1)
        mean_var_per_sample = p_variance[dim]
        confidence = np.maximum(preds, 1 - preds).mean(axis=1)

        results[dim] = {
            "correct": {
                "mean_entropy": float(mean_entropy_per_sample[correct].mean()),
                "mean_variance": float(mean_var_per_sample[correct].mean()),
                "mean_confidence": float(confidence[correct].mean()),
            },
            "incorrect": {
                "mean_entropy": float(mean_entropy_per_sample[~correct].mean()),
                "mean_variance": float(mean_var_per_sample[~correct].mean()),
                "mean_confidence": float(confidence[~correct].mean()),
            },
        }

    return results


def save_high_uncertainty_examples(results, top_k=10):
    """Save the top-k highest-uncertainty examples."""
    # Sort by mean entropy across all tasks
    if not results:
        return []

    # Compute composite uncertainty score
    for r in results:
        entropies = []
        for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]:
            entropies.append(r.get(f"{dim}_entropy", 0))
        r["composite_entropy"] = sum(entropies) / len(entropies)

    sorted_results = sorted(results, key=lambda r: r["composite_entropy"], reverse=True)
    return sorted_results[:top_k]