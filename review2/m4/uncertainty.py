import math
import time
import torch
from review2.m2.alignment import collate


def enable_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or module.__class__.__name__ == 'StableDropout':
            module.train()


def sample_logits(model, examples, tokenizer, passes, batch_size):
    if passes < 2:
        raise ValueError('MC uncertainty requires at least two passes')
    states = {module: module.training for module in model.modules()}
    device = next(model.parameters()).device
    samples, elapsed = [], []
    start = time.perf_counter()
    try:
        enable_dropout(model)
        with torch.no_grad():
            for _ in range(passes):
                current = []
                for index in range(0, len(examples), batch_size):
                    part = examples[index:index + batch_size]
                    batch = collate(part, tokenizer.pad_token_id)
                    z = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits.cpu()
                    current.extend(t[:len(e['input_ids'])] for e, t in zip(part, z))
                samples.append(current)
                elapsed.append(time.perf_counter() - start)
    finally:
        for module, state in states.items():
            module.training = state
    return [torch.stack([p[i] for p in samples]) for i in range(len(examples))], elapsed


def uncertainty(probabilities):
    mean = probabilities.mean(0)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
    expected = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean(0)
    return {'mean': mean, 'entropy': entropy / math.log(mean.shape[-1]),
            'mutual_information': (entropy - expected).clamp_min(0) / math.log(mean.shape[-1]),
            'variance': probabilities.var(0, unbiased=False).mean(-1),
            'variation_ratio': 1 - torch.nn.functional.one_hot(probabilities.argmax(-1), mean.shape[-1]).float().mean(0).max(-1).values}
