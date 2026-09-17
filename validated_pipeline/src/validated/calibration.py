"""Positive temperatures fitted to fixed MC samples on calibration only."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def probability(samples, kind, temperature=1.0):
    scaled = samples / temperature
    return (scaled.softmax(-1) if kind == 'categorical' else scaled.sigmoid()).mean(0)


def mc_nll(samples, targets, mask, kind, temperature=1.0):
    # Log-mixture categorical likelihood; binary mixture is Bernoulli with mean probability.
    if kind == 'categorical':
        logp = torch.logsumexp(F.log_softmax(samples / temperature, -1), dim=0) - math.log(samples.shape[0])
        raw = -(targets * logp).sum(-1)
    else:
        # Each independent label uses BCE-with-logits after converting MC mean to equivalent logits.
        p = probability(samples, kind, temperature).clamp(1e-7, 1 - 1e-7)
        raw = F.binary_cross_entropy_with_logits(torch.logit(p), targets, reduction='none')
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def fit_temperature(samples, targets, mask, kind, split, iterations=100):
    if split != 'calibration':
        raise ValueError('Temperatures may only be fitted on calibration split')
    if mask.sum() == 0:
        return {'temperature': 1.0, 'status': 'unfitted_no_observed_labels'}
    samples, targets, mask = samples.detach(), targets.detach(), mask.detach()
    raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=samples.dtype))
    optimizer = torch.optim.LBFGS([raw], max_iter=int(iterations), line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        objective = mc_nll(samples, targets, mask, kind, F.softplus(raw) + 1e-6)
        objective.backward()
        return objective
    optimizer.step(closure)
    t = float((F.softplus(raw) + 1e-6).detach())
    if not math.isfinite(t):
        raise ValueError('Nonfinite temperature')
    return {'temperature': t, 'status': 'fitted_calibration_weak_targets'}


def ece(probs, targets, mask, kind, bins=15):
    if kind == 'categorical':
        confidence, pred = probs.max(-1)
        # Expected correctness under a soft weak target; explicitly not gold calibration.
        correctness = targets.gather(1, pred[:, None]).squeeze(1)
    else:
        confidence = torch.maximum(probs, 1 - probs)
        correctness = torch.where(probs >= .5, targets, 1 - targets)
    valid = mask.bool()
    confidence, correctness = confidence[valid].flatten(), correctness[valid].flatten()
    if not len(confidence):
        return None
    total = 0.0
    for i in range(bins):
        keep = (confidence >= i / bins) & ((confidence < (i + 1) / bins) if i < bins - 1 else (confidence <= 1))
        if keep.any():
            total += float(keep.float().mean() * (confidence[keep].mean() - correctness[keep].mean()).abs())
    return total


def uncertainty(samples, kind, temperature=1.0):
    p = (samples / temperature).softmax(-1) if kind == 'categorical' else (samples / temperature).sigmoid()
    mean = p.mean(0)
    if kind == 'categorical':
        entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
        normalized_entropy = entropy / math.log(mean.shape[-1])
        confidence = mean.max(-1).values
    else:
        q = mean.clamp(1e-7, 1 - 1e-7)
        entropy = -(q * q.log() + (1 - q) * (1 - q).log()).mean(-1)
        normalized_entropy = entropy / math.log(2)
        confidence = torch.maximum(mean, 1 - mean).mean(-1)
    return dict(mean=mean, predictive_variance=p.var(0, unbiased=False), entropy=entropy,
                normalized_entropy=normalized_entropy, confidence=confidence)


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout) or module.__class__.__name__ == 'StableDropout':
            module.train()
