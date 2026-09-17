"""Positive temperature fitted to detached calibration logits, never detector weights."""
import math
import torch
from torch.nn import functional as F


def fit_temperature(logits, targets, split, iterations=60):
    if split != 'calibration':
        raise ValueError('Temperature fitting requires calibration split')
    logits, targets = logits.detach(), targets.detach()
    valid = targets != -100
    logits, targets = logits[valid], targets[valid]
    if not len(targets):
        raise ValueError('No calibration labels')
    raw = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.)), device=logits.device))
    optimizer = torch.optim.LBFGS([raw], max_iter=iterations, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / (F.softplus(raw) + 1e-6), targets)
        loss.backward()
        return loss
    before = float(F.cross_entropy(logits, targets))
    optimizer.step(closure)
    temperature = float((F.softplus(raw) + 1e-6).detach())
    after = float(F.cross_entropy(logits / temperature, targets))
    if not math.isfinite(temperature) or after > before + 1e-6:
        raise ValueError('Temperature optimization failed')
    return {'temperature': temperature, 'nll_before': before, 'nll_after': after,
            'fit_split': split, 'fit_tokens': len(targets), 'detector_weights_updated': False}
