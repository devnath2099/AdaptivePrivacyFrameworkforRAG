import torch
from torch.nn import functional as F


def class_weights(examples, count):
    counts = torch.zeros(count)
    for example in examples:
        for index in example['labels']:
            if index >= 0:
                counts[index] += 1
    # Inverse square-root frequency, normalized across observed training labels only.
    weights = counts.clamp_min(1).rsqrt()
    weights /= weights[counts > 0].mean()
    return weights


def token_loss(logits, labels, weights=None):
    valid = labels != -100
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[valid], labels[valid], weight=weights)
