"""Prediction-blind, grouped multilabel subset plans for the bounded experiment."""
from collections import Counter, defaultdict
import math
import random
import numpy as np
from .m1 import features, statistics


def ordered_groups(records, seed, rare_document_threshold=10):
    """Nested stratified order, not source order or shortest-document selection.

    Anchor EVERY group containing a class with <= threshold supporting documents,
    then cover remaining classes. Greedily minimize multilabel distribution error
    thereafter, using a seeded tie break. No predictions or correctness enter.
    """
    groups = defaultdict(list)
    for r in records:
        groups[r['group']].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    kinds = sorted({e['type'] for r in records for e in r['entities']})
    tags = kinds + ['__positive__', '__negative__']
    supports = Counter(k for r in records for k in {e['type'] for e in r['entities']})
    matrix = np.array([[sum(k in features([r]) for r in groups[g]) for k in tags] for g in keys], dtype=float)
    sizes = np.array([len(groups[g]) for g in keys], dtype=float)
    rates = matrix.sum(0)/max(1, len(records))
    rare = {k for k in kinds if supports[k] <= rare_document_threshold}
    chosen = [i for i, key in enumerate(keys) if features(groups[key]) & rare]
    covered = set().union(*(features(groups[keys[i]]) for i in chosen)) if chosen else set()
    for k in sorted(kinds, key=lambda t: (supports[t], t)):
        if k not in covered:
            i = next(i for i in range(len(keys)) if k in features(groups[keys[i]]))
            if i not in chosen:
                chosen.append(i)
                covered |= features(groups[keys[i]])
    anchor_count = len(chosen)
    available = np.ones(len(keys), dtype=bool)
    available[chosen] = False
    actual = matrix[chosen].sum(0) if chosen else np.zeros(len(tags))
    n = sizes[chosen].sum() if chosen else 0.
    weights = 1/np.maximum(rates, 1/max(1, len(records)))
    while available.any():
        candidates = np.flatnonzero(available)
        residual = actual[None, :] + matrix[candidates] - (n+sizes[candidates,None])*rates
        scores = (residual**2*weights).sum(1)
        i = candidates[int(np.argmin(scores))]
        chosen.append(int(i))
        available[i] = False
        actual += matrix[i]
        n += sizes[i]
    return [groups[keys[i]] for i in chosen], {
        'anchor_groups': anchor_count, 'anchor_documents': int(sum(sizes[i] for i in chosen[:anchor_count])),
        'rare_classes_all_support_retained': sorted(rare), 'category_document_support': dict(supports),
        'seed': seed, 'strategy': 'rare-class anchors + greedy document-level multilabel balancing; prediction-blind'}


def choose_training(groups, anchors, windows_by_id, seconds_available, step_cost, batch_size):
    """Largest prefix of the precomputed stratified group order that fits estimate."""
    selected, windows = [], 0
    for group in groups:
        added = sum(windows_by_id[r['record_id']] for r in group)
        if math.ceil((windows+added)/batch_size)*step_cost > seconds_available:
            break
        selected.extend(group)
        windows += added
    if len(selected) < anchors:
        raise RuntimeError('Time budget cannot fit mandatory rare-class coverage. No training started; use faster compute or more time.')
    return selected, windows


def demo_plans(records, seed, fractions=(1., .75, .5, .25, .125)):
    groups, info = ordered_groups(records, seed)
    result, seen = [], set()
    for fraction in fractions:
        target = max(info['anchor_documents'], math.ceil(len(records)*fraction))
        selected = []
        for group in groups:
            if len(selected) >= target:
                break
            selected.extend(group)
        ids = tuple(sorted(r['record_id'] for r in selected))
        if ids not in seen:
            seen.add(ids)
            result.append({'fraction_requested': fraction, 'record_ids': list(ids),
                           'statistics': statistics(selected)})
    return {'selection': info, 'candidates': result,
            'warning': 'Nested convenience demo subsets of the SAME reference test, not independent test sets. Rare-class anchors can skew prevalence.'}


def metric_differences(reference, candidate, tolerance=.05):
    differences = {}
    for name in ('precision', 'recall', 'f1'):
        a, b = reference['micro'][name], candidate['micro'][name]
        differences[f'micro_{name}'] = None if a is None or b is None else b-a
    a, b = reference['macro_f1_supported_classes'], candidate['macro_f1_supported_classes']
    differences['macro_f1'] = None if a is None or b is None else b-a
    for k, a in reference['per_class'].items():
        b = candidate['per_class'][k]
        for name in ('precision', 'recall', 'f1'):
            differences[f'{k}_{name}'] = None if a[name] is None or b[name] is None else b[name]-a[name]
    return {'signed_differences': differences, 'absolute_tolerance': tolerance,
            'all_metrics_within_tolerance': all(v is not None and abs(v) <= tolerance for v in differences.values()),
            'interpretation': 'Descriptive agreement only, not statistical equivalence. Never used to choose membership or retrain.'}
