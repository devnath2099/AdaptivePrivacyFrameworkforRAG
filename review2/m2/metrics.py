"""Strict typed entity matching is primary; no token-accuracy substitution."""
from collections import Counter


def prf(tp, fp, fn):
    return {'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'tp': tp, 'fp': fp, 'fn': fn}


def entity_metrics(rows, predictions, train_counts=None):
    if len(rows) != len(predictions):
        raise ValueError('Prediction/record count mismatch')
    counts, by_type, by_source = Counter(), {}, {}
    pii_tp = pii_fn = nonpii_ok = nonpii_n = 0
    for row, pred in zip(rows, predictions):
        gold = {(s['start'], s['end'], s['type']) for s in row['canonical_spans']}
        found = {(s['start'], s['end'], s['type']) for s in pred}
        delta = Counter(tp=len(gold & found), fp=len(found - gold), fn=len(gold - found))
        counts.update(delta)
        by_source.setdefault(row['source_id'], Counter()).update(delta)
        for kind in {s[2] for s in gold | found}:
            g, p = {s for s in gold if s[2] == kind}, {s for s in found if s[2] == kind}
            by_type.setdefault(kind, Counter()).update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
        g, p = {(s[0], s[1]) for s in gold}, {(s[0], s[1]) for s in found}
        pii_tp += len(g & p)
        pii_fn += len(g - p)
        if not gold:
            nonpii_n += 1
            nonpii_ok += not found
    per_type = {k: prf(**v) for k, v in by_type.items()}
    rare = [k for k in per_type if train_counts is not None and train_counts.get(k, 0) <= 10]
    rare_tp = sum(by_type[k]['tp'] for k in rare)
    rare_fn = sum(by_type[k]['fn'] for k in rare)
    return {**prf(counts['tp'], counts['fp'], counts['fn']), 'micro_f1': prf(counts['tp'], counts['fp'], counts['fn'])['f1'],
            'macro_f1': sum(x['f1'] for x in per_type.values()) / max(1, len(per_type)),
            'per_type': per_type, 'per_source': {k: prf(**v) for k, v in by_source.items()},
            'rare_type_recall': rare_tp / (rare_tp + rare_fn) if rare_tp + rare_fn else None,
            'rare_definition': 'at most 10 train mentions',
            'pii_untyped_span_recall': pii_tp / (pii_tp + pii_fn) if pii_tp + pii_fn else None,
            'nonpii_record_recall': nonpii_ok / nonpii_n if nonpii_n else None}
