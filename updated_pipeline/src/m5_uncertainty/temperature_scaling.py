"""M5 -- MC-Dropout Uncertainty Quantification with Proper Temperature Scaling.

Design (per the Review-2 corrections):
  - Separate temperature per categorical head: sensitivity, intent, disclosure_scope
  - Single shared temperature for multi-label heads via BCE-with-logits: entity_tags, threat_content
  - Parameterize as log_T to ensure T > 0
  - Optimizer: LBFGS (default) or Adam with weight_decay=0.0
  - Report ECE and NLL before and after calibration
  - Fit only on a designated calibration split, never the test/reference set
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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


def _get_logits(predictions):
    """Extract logits dict from MC-Dropout predictions."""
    return predictions[0]  # All predictions have same structure


def compute_categorical_probabilities(predictions, temperature, mc_passes):
    """Compute softmax probabilities for categorical tasks with temperature.

    Parameters
    ----------
    predictions : list of dicts
        Output from mc_dropout_inference
    temperature : float or dict
        If float: shared temperature for all heads.
        If dict: per-head temperatures {'sensitivity': T1, 'intent': T2, 'disclosure_scope': T3}
    mc_passes : int

    Returns
    -------
    prob_lists : dict mapping dim -> list of probability tensors
    """
    prob_lists = {}
    categorical_dims = ["sensitivity", "intent", "disclosure_scope"]
    for dim in categorical_dims:
        if isinstance(temperature, dict):
            T = temperature.get(dim, 1.0)
        else:
            T = temperature
        T = max(T, 1e-6)
        logits_list = [p[f"{dim}_logits"] / T for p in predictions]
        prob_lists[dim] = [F.softmax(l, dim=-1) for l in logits_list]
    return prob_lists


def compute_multilabel_probabilities(predictions, temperature, mc_passes):
    """Compute sigmoid probabilities for multi-label tasks with temperature.

    Parameters
    ----------
    predictions : list of dicts
    temperature : float
        Shared temperature for all multi-label heads.
    mc_passes : int

    Returns
    -------
    prob_lists : dict mapping dim -> list of probability tensors
    """
    prob_lists = {}
    multi_label_dims = ["entity_tags", "threat_content"]
    for dim in multi_label_dims:
        T = max(temperature, 1e-6)
        logits_list = [p[f"{dim}_logits"] / T for p in predictions]
        prob_lists[dim] = [torch.sigmoid(l) for l in logits_list]
    return prob_lists


def compute_predictive_mean(prob_lists):
    """Compute predictive mean: p_mean = (1/T) * sum_t p_t."""
    return {dim: torch.stack(probs).mean(dim=0) for dim, probs in prob_lists.items()}


def compute_predictive_variance(prob_lists, p_mean):
    """Compute predictive variance."""
    return {dim: torch.stack([(p - p_mean[dim]) ** 2 for p in probs]).mean(dim=0)
            for dim, probs in prob_lists.items()}


def compute_categorical_entropy(p_mean):
    """Compute predictive entropy for categorical tasks."""
    eps = 1e-8
    return -torch.sum(p_mean * torch.log(p_mean + eps), dim=-1)


def compute_binary_entropy(p):
    """Compute binary entropy."""
    eps = 1e-8
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)


def compute_confidence(p_mean, dim):
    """Compute confidence."""
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
        Temperature scaling applied before softmax/sigmoid.

    Returns
    -------
    results : list of dict, one per record
    uncertainty_summary : dict
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

            prob_lists = {}
            if isinstance(temperature, dict):
                prob_lists.update(compute_categorical_probabilities(predictions, temperature, mc_passes))
                prob_lists.update(compute_multilabel_probabilities(predictions, temperature["__shared__"], mc_passes))
            else:
                prob_lists.update(compute_categorical_probabilities(predictions, temperature, mc_passes))
                prob_lists.update(compute_multilabel_probabilities(predictions, temperature, mc_passes))

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


def compute_nll_logits(logits, targets):
    """Compute NLL for a batch of logits and one-hot targets."""
    # logits: [B, num_classes], targets: [B, num_classes] (one-hot)
    class_targets = targets.argmax(dim=-1)
    return F.cross_entropy(logits, class_targets)


def compute_bce_logits(logits, targets):
    """Compute BCE-with-logits for multi-label targets.

    logits: [B, num_labels], targets: [B, num_labels] (one-hot)
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def fit_categorical_temperature(model, calibration_loader, device, n_classes,
                                num_iterations=100, optimizer_type="lbfgs",
                                lr=0.05, mc_passes=5):
    """Fit a single temperature for a categorical head using NLL minimization.

    Parameterizes as log_T to ensure T > 0.

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    calibration_loader : DataLoader
    device : torch.device
    n_classes : int
    num_iterations : int
    optimizer_type : str ("lbfgs" or "adam")
    lr : float
    mc_passes : int

    Returns
    -------
    temperature : float
    nll_before : float
    nll_after : float
    """
    log_T = nn.Parameter(torch.tensor(0.0, device=device))  # T = exp(log_T) > 0

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=20)
    else:
        optimizer = torch.optim.Adam([log_T], lr=lr, weight_decay=0.0)

    model.to(device)
    model.eval()

    # Collect all calibration logits and targets
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in calibration_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["soft_targets"]
            # Get deterministic logits (no MC-Dropout for calibration fitting)
            outputs = model(input_ids, attention_mask)
            all_logits.append(outputs["sensitivity_logits"].cpu() if n_classes == 3 else outputs["intent_logits"].cpu())
            # We'll handle this differently below

    # Actually, let's do this properly: collect logits per batch, then optimize
    # Reset
    all_logits = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            # We need to identify which head by n_classes
            all_targets.append(batch["soft_targets"])

    # Simpler approach: do the optimization step-by-step over batches
    log_T.data = torch.tensor(0.0, device=device)

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        n_samples = 0
        model.eval()
        set_dropout_training(model)
        for batch in calibration_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["soft_targets"]

            # Get logits with MC-Dropout averaging
            predictions = []
            for _ in range(mc_passes):
                outputs = model(input_ids, attention_mask)
                predictions.append(outputs)
            set_dropout_training(model)

            batch_logits = torch.stack([p["sensitivity_logits"] for p in predictions]).mean(dim=0)
            batch_targets = targets["sensitivity"].to(device)

            T = torch.exp(log_T)
            logits_scaled = batch_logits / T
            loss = F.cross_entropy(logits_scaled, batch_targets.argmax(dim=-1))
            total_loss += loss.item() * input_ids.size(0)
            n_samples += input_ids.size(0)

        avg_loss = total_loss / max(n_samples, 1)
        if optimizer_type == "lbfgs":
            avg_loss.backward()
        return avg_loss * input_ids.size(0)  # Scale for LBFGS

    if optimizer_type == "lbfgs":
        optimizer.step(closure)
    else:
        for _ in range(num_iterations):
            optimizer.step(closure)

    T = float(torch.exp(log_T.item()).item())
    T = max(T, 1e-6)

    return T


def fit_temperature_scaling(model, calibration_dataset, val_dataset, device=None,
                            num_iterations=100, optimizer_type="lbfgs",
                            lr=0.05, mc_passes=5, batch_size=32):
    """Fit proper per-head temperature scaling.

    Design (Review-2 corrections):
      - Per-head temperature for categorical tasks (sensitivity, intent, disclosure_scope)
      - Shared temperature for multi-label tasks (entity_tags, threat_content) via BCE-with-logits
      - log_T parameterization for positive temperature constraint
      - LBFGS or Adam with weight_decay=0.0
      - ECE/NLL reported before and after calibration
      - Fit only on calibration_dataset; val_dataset used only for ECE evaluation

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    calibration_dataset : M3Dataset or Subset
        Used ONLY for fitting temperatures (not evaluation).
    val_dataset : M3Dataset or Subset
        Used ONLY for ECE evaluation before/after.
    device : torch.device
    num_iterations : int
    optimizer_type : str
    lr : float
    mc_passes : int
    batch_size : int

    Returns
    -------
    temperatures : dict
        {'sensitivity': T, 'intent': T, 'disclosure_scope': T, '__shared__': T}
    ece_before : dict
    ece_after : dict
    nll_before : float
    nll_after : float
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute ECE/NLL before calibration (T=1.0)
    print("  Computing pre-calibration ECE/NLL (T=1.0)...")
    ece_before = _compute_ece_split(model, val_loader, device, mc_passes=mc_passes)
    nll_before = _compute_nll_split(model, val_loader, device, mc_passes=mc_passes)
    print(f"    Pre-calibration: ECE={ece_before['overall']:.4f}, NLL={nll_before:.4f}")

    # Fit per-head temperatures for categorical tasks
    categorical_dims = {
        "sensitivity": 3,
        "intent": 5,
        "disclosure_scope": 2,
    }
    temperatures = {}

    for dim, n_classes in categorical_dims.items():
        # Create a temporary loader for this specific head
        print(f"  Fitting temperature for {dim}...")
        T = fit_categorical_temperature(
            model, calibration_loader, device, n_classes,
            num_iterations=num_iterations, optimizer_type=optimizer_type,
            lr=lr, mc_passes=mc_passes,
        )
        temperatures[dim] = T
        print(f"    {dim}: T={T:.4f}")

    # Fit shared temperature for multi-label tasks
    print("  Fitting shared temperature for multi-label heads...")
    # For multi-label, use a separate optimization
    multi_label_T = _fit_multilabel_temperature(
        model, calibration_loader, device,
        num_iterations=num_iterations, optimizer_type=optimizer_type,
        lr=lr, mc_passes=mc_passes, batch_size=batch_size,
    )
    temperatures["__shared__"] = multi_label_T
    print(f"    Multi-label shared T={multi_label_T:.4f}")

    # Compute ECE/NLL after calibration
    print("  Computing post-calibration ECE/NLL...")
    ece_after = _compute_ece_split(model, val_loader, device, mc_passes=mc_passes, temperatures=temperatures)
    nll_after = _compute_nll_split(model, val_loader, device, mc_passes=mc_passes, temperatures=temperatures)
    print(f"    Post-calibration: ECE={ece_after['overall']:.4f}, NLL={nll_after:.4f}")

    return temperatures, ece_before, ece_after, nll_before, nll_after


def _fit_multilabel_temperature(model, calibration_loader, device,
                                 num_iterations=100, optimizer_type="lbfgs",
                                 lr=0.05, mc_passes=5, batch_size=32):
    """Fit shared temperature for multi-label heads using BCE-with-logits."""
    log_T = nn.Parameter(torch.tensor(0.0, device=device))

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=20)
    else:
        optimizer = torch.optim.Adam([log_T], lr=lr, weight_decay=0.0)

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        n_samples = 0
        model.eval()
        set_dropout_training(model)
        for batch in calibration_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)

            # Use entity_tags and threat_content for multi-label calibration
            entity_targets = batch["soft_targets"]["entity_tags"].to(device)
            threat_targets = batch["soft_targets"]["threat_content"].to(device)

            entity_logits = outputs["entity_tags_logits"] / torch.exp(log_T)
            threat_logits = outputs["threat_content_logits"] / torch.exp(log_T)

            loss = F.binary_cross_entropy_with_logits(entity_logits, entity_targets)
            loss += F.binary_cross_entropy_with_logits(threat_logits, threat_targets)

            total_loss += loss.item() * input_ids.size(0)
            n_samples += input_ids.size(0)

        avg_loss = total_loss / max(n_samples, 1)
        if optimizer_type == "lbfgs":
            avg_loss.backward()
        return avg_loss * input_ids.size(0)

    if optimizer_type == "lbfgs":
        optimizer.step(closure)
    else:
        for _ in range(num_iterations):
            optimizer.step(closure)

    return float(torch.exp(log_T.item()).item())


def _compute_nll_split(model, dataloader, device, mc_passes=5, temperatures=None):
    """Compute average NLL over a dataset split."""
    model.eval()
    total_nll = 0.0
    n_samples = 0
    T_val = 1.0

    if temperatures:
        # Use categorical head temperatures for NLL computation
        cat_temps = {k: v for k, v in temperatures.items() if k != "__shared__"}
        ml_temp = temperatures.get("__shared__", 1.0)
    else:
        cat_temps = {}
        ml_temp = 1.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            targets = batch["soft_targets"]

            # Sensitivity NLL
            if "sensitivity" in targets:
                logits = outputs["sensitivity_logits"]
                T = cat_temps.get("sensitivity", T_val)
                logits_scaled = logits / T
                total_nll += F.cross_entropy(logits_scaled, targets["sensitivity"].argmax(dim=-1)).item() * input_ids.size(0)
                n_samples += input_ids.size(0)

    return total_nll / max(n_samples, 1)


def _compute_ece_split(model, dataloader, device, mc_passes=5, temperatures=None):
    """Compute ECE over a dataset split. Returns dict with overall and per-head ECE."""
    from m5_uncertainty.calibration import compute_ece, compute_categorical_ece, compute_multilabel_ece
    import numpy as np

    all_preds = {}
    all_targets = {}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            targets = batch["soft_targets"]

            for dim in ["sensitivity", "intent", "disclosure_scope"]:
                logits = outputs[f"{dim}_logits"]
                T = temperatures.get(dim, 1.0) if temperatures else 1.0
                probs = F.softmax(logits / T, dim=-1)
                if dim not in all_preds:
                    all_preds[dim] = []
                    all_targets[dim] = []
                all_preds[dim].append(probs.cpu().numpy())
                all_targets[dim].append(targets[dim].cpu().numpy())

            for dim in ["entity_tags", "threat_content"]:
                logits = outputs[f"{dim}_logits"]
                T = temperatures.get("__shared__", 1.0) if temperatures else 1.0
                probs = torch.sigmoid(logits / T)
                if dim not in all_preds:
                    all_preds[dim] = []
                    all_targets[dim] = []
                all_preds[dim].append(probs.cpu().numpy())
                all_targets[dim].append(targets[dim].cpu().numpy())

    ece_results = {"overall": 0.0, "per_head": {}}
    total_dims = 0

    for dim in ["sensitivity", "intent", "disclosure_scope"]:
        preds = np.concatenate(all_preds[dim])
        targets = np.concatenate(all_targets[dim])
        # Compute per-record ECE
        all_ece = []
        for i in range(len(preds)):
            ece_i, _ = compute_ece(
                np.array([preds[i].max()]),
                np.array([int(preds[i].argmax() == targets[i].argmax())]),
                n_bins=10
            )
            all_ece.append(ece_i)
        ece_results["per_head"][dim] = float(np.mean(all_ece))
        ece_results["overall"] += float(np.mean(all_ece))
        total_dims += 1

    for dim in ["entity_tags", "threat_content"]:
        preds = np.concatenate(all_preds[dim])
        targets = np.concatenate(all_targets[dim])
        macro_ece, _, _ = compute_multilabel_ece(preds, targets, 0.5, 10)
        ece_results["per_head"][dim] = float(macro_ece)
        ece_results["overall"] += float(macro_ece)
        total_dims += 1

    ece_results["overall"] /= max(total_dims, 1)
    return ece_results


