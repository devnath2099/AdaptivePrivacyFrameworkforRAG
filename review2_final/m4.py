"""Calibration of deterministic and MC predictive distributions; no risk scoring."""
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from .common import seed_all
from .metrics import calibration_metrics, entity_metrics
from .neural import predict_logits, decode


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def scale(p, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return softmax(np.log(np.clip(p, 1e-12, 1))/temperature)


def fit_temperature(probabilities, gold, budget_check=None):
    """Fit post-hoc scalar temperature on calibration split only.

    For MC this scales log(mean predictive probability), not mean logits.
    Optimize in chunks to avoid retaining a large autograd graph.
    """
    print(f"Fitting temperature on {len(gold):,} calibration tokens...", flush=True)
    logp = torch.from_numpy(np.log(np.clip(probabilities, 1e-12, 1))).double()
    y = torch.tensor(gold, dtype=torch.long)
    if not len(y):
        raise ValueError("Empty calibration data")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=.2, max_iter=40, line_search_fn="strong_wolfe")
    def closure():
        if budget_check:
            budget_check()
        opt.zero_grad()
        total = 0.
        for start in range(0, len(y), 32768):
            temp = log_t.clamp(-4.6, 4.6).exp()
            loss = torch.nn.functional.cross_entropy(logp[start:start+32768]/temp, y[start:start+32768], reduction="sum")/len(y)
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total, dtype=torch.float64)
    opt.step(closure)
    temperature = float(log_t.detach().clamp(-4.6, 4.6).exp())
    # A failed optimizer must never make calibration worse than the identity map.
    def nll(t):
        p = scale(probabilities, t)
        return -np.log(np.clip(p[np.arange(len(y)), gold], 1e-12, 1)).mean()
    chosen = temperature if nll(temperature) <= nll(1.) else 1.
    print(f"Temperature fitting complete: T={chosen:.5f}", flush=True)
    return chosen


def gold_array(records, labels):
    index = {v: i for i, v in enumerate(labels)}
    return np.array([index[t] for r in records for t in r["labels"]], dtype=np.int64)


def sample_predictive(model, batches, records, labels, device, passes, seed, budget_check=None):
    """Generator retains running sums, not all MC samples. Nested pass budgets."""
    passes = sorted(set(passes))
    if not passes or passes[0] < 1:
        raise ValueError("Positive pass budgets required")
    total, total_entropy = None, None
    elapsed = 0.
    for i in tqdm(range(1, max(passes)+1), desc="MC dropout passes", unit="pass", mininterval=1.0):
        start = time.perf_counter()
        seed_all(seed+i)
        kwargs = {'budget_check': budget_check} if budget_check else {}
        logits = predict_logits(model, batches, records, len(labels), device, stochastic=True,
                                description=f"MC pass {i}/{max(passes)}", **kwargs)
        p = softmax(np.concatenate(logits))
        h = entropy(p)
        if total is None:
            total, total_entropy = p.astype(np.float64), h.astype(np.float64)
        else:
            total += p
            total_entropy += h
        elapsed += time.perf_counter()-start
        if i in passes:
            mean = (total/i).astype(np.float32)
            expected_entropy = (total_entropy/i).astype(np.float32)
            yield i, mean, {"predictive_entropy": entropy(mean),
                           "expected_entropy": expected_entropy,
                           "mutual_information": np.maximum(0., entropy(mean)-expected_entropy),
                           "seconds": elapsed}


def report(records, p, labels, temperature, bins, uncertainty=None):
    gold = gold_array(records, labels)
    calibrated = scale(p, temperature)
    raw = calibration_metrics(p, gold, bins)
    result = {"temperature": temperature, "raw_token_metrics": raw,
              "calibrated_token_metrics": calibration_metrics(calibrated, gold, bins),
              "pii_gold_token_metrics": calibration_metrics(calibrated[gold != labels.index("O")], gold[gold != labels.index("O")], bins),
              "metric_scope": "Native tokens (first subword); not entity-level calibration or privacy-risk accuracy."}
    if uncertainty is not None:
        result["entropy_ranked_metrics"] = calibration_metrics(calibrated, gold, bins, uncertainty)
    cuts = np.cumsum([len(r["tokens"]) for r in records])[:-1]
    result["entities"] = entity_metrics(records, decode(np.split(calibrated, cuts), labels), sorted({t[2:] for t in labels if t != "O"}))
    return result
