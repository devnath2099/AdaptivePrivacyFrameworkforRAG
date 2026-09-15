"""FGSM adversarial perturbation in the DeBERTa embedding space.

For each batch:
1. Get token embeddings from encoder (requires_grad)
2. Forward through encoder + heads to compute clean loss
3. Use torch.autograd.grad to get gradient w.r.t. embeddings
4. Generate delta = epsilon * sign(grad) * attention_mask
5. E_adv = E + delta.detach()
6. Forward from E_adv to compute adversarial loss
7. L_total = L_clean + lambda_adv * L_adv
8. Single backward on L_total
"""
from __future__ import annotations

import torch


def _forward_from_embeddings(model, embeddings, attention_mask, targets, criterion):
    """Run model forward from embedding tensor (bypassing token input).

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    embeddings : torch.Tensor [B, seq_len, hidden_size]
    attention_mask : torch.Tensor [B, seq_len]
    targets : dict of {dim: tensor}
    criterion : callable, compute_total_loss

    Returns
    -------
    logits_dict : dict
    loss : torch.Tensor (scalar)
    task_losses : dict
    """
    encoder_outputs = model.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
    pooled = encoder_outputs.last_hidden_state[:, 0]  # CLS token
    h = model.dropout(pooled)

    logits_dict = {"representation": h}
    for dim_name, head in model.heads.items():
        logits_dict[f"{dim_name}_logits"] = head(h)

    loss, task_losses = criterion(logits_dict, targets)
    return logits_dict, loss, task_losses


def compute_clean_loss_and_grad(model, input_ids, attention_mask, targets, criterion):
    """Compute clean loss and gradient of loss w.r.t. embeddings.

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    input_ids : torch.Tensor [B, seq_len]
    attention_mask : torch.Tensor [B, seq_len]
    targets : dict of {dim: tensor}
    criterion : callable, compute_total_loss

    Returns
    -------
    embeddings : torch.Tensor [B, seq_len, hidden_size]
    clean_loss : torch.Tensor (scalar)
    grad_wrt_embeddings : torch.Tensor [B, seq_len, hidden_size]
    task_losses : dict
    """
    model.zero_grad()

    embeddings = model.encoder.embeddings(input_ids=input_ids)

    _, clean_loss, task_losses = _forward_from_embeddings(
        model, embeddings, attention_mask, targets, criterion
    )

    grad_wrt_embeddings = torch.autograd.grad(
        clean_loss, embeddings, create_graph=False, retain_graph=True
    )[0]

    return embeddings, clean_loss, grad_wrt_embeddings, task_losses


def generate_fgsm_delta(embedding_grad, attention_mask, epsilon=1e-3):
    """Generate FGSM perturbation.

    delta = epsilon * sign(grad) * attention_mask

    Parameters
    ----------
    embedding_grad : torch.Tensor [B, seq_len, hidden_size]
    attention_mask : torch.Tensor [B, seq_len]
    epsilon : float

    Returns
    -------
    delta : torch.Tensor [B, seq_len, hidden_size]
    """
    sign_grad = embedding_grad.sign()
    mask = attention_mask.unsqueeze(-1).float()
    delta = epsilon * sign_grad * mask
    return delta


def adversarial_forward(
    model, input_ids, attention_mask, targets, criterion, epsilon=1e-3, lambda_adv=1.0
):
    """Full M4 adversarial forward pass with combined loss.

    Sequence:
      input_ids -> embeddings E -> clean forward -> L_clean -> grad_E L_clean
      -> delta = epsilon * sign(grad) * mask -> E_adv = E + delta.detach()
      -> adversarial forward -> L_adv -> L_total = L_clean + lambda_adv * L_adv
      -> ONE backward on L_total -> optimizer step

    Parameters
    ----------
    model : DeBERTaMultiTaskModel
    input_ids : torch.Tensor [B, seq_len]
    attention_mask : torch.Tensor [B, seq_len]
    targets : dict of {dim: tensor}
    criterion : callable, compute_total_loss
    epsilon : float — FGSM perturbation magnitude
    lambda_adv : float — weight for adversarial loss

    Returns
    -------
    total_loss : torch.Tensor
    clean_loss_val : float
    adv_loss_val : float
    task_losses : dict
    delta : torch.Tensor [B, seq_len, hidden_size]
    max_abs_delta : float
    """
    model.zero_grad()

    # 1. Get embeddings and compute clean loss + gradient
    embeddings, clean_loss, grad_wrt_embeddings, task_losses = (
        compute_clean_loss_and_grad(model, input_ids, attention_mask, targets, criterion)
    )
    clean_loss_val = clean_loss.item()

    # 2. Generate FGSM perturbation
    delta = generate_fgsm_delta(grad_wrt_embeddings, attention_mask, epsilon)
    max_abs_delta = float(delta.abs().max().item())

    # 3. Adversarial embeddings (detach perturbation so it's not updated)
    embeddings_adv = embeddings + delta.detach()

    # 4. Adversarial forward
    _, adv_loss, _ = _forward_from_embeddings(
        model, embeddings_adv, attention_mask, targets, criterion
    )
    adv_loss_val = adv_loss.item()

    # 5. Combined loss
    total_loss = clean_loss + lambda_adv * adv_loss

    # 6. Single backward for combined loss
    total_loss.backward()

    return total_loss, clean_loss, adv_loss, task_losses, delta, max_abs_delta


def compute_adv_loss_from_embeddings(model, embeddings, attention_mask, targets, criterion):
    """Compute adversarial loss given pre-computed embeddings."""
    _, adv_loss, _ = _forward_from_embeddings(
        model, embeddings, attention_mask, targets, criterion
    )
    return adv_loss