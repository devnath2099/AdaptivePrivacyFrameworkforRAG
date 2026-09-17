import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def reliability_metrics(probabilities, targets, bins=15, error_score=None):
    valid = targets != -100
    p, y = probabilities[valid], targets[valid]
    if not len(y):
        raise ValueError('No evaluable labels')
    confidence, predicted = p.max(-1)
    correct = predicted == y
    errors = (~correct).numpy().astype(int)
    score = (1 - confidence).numpy() if error_score is None else error_score[valid].numpy()
    reliability, ece = [], 0.
    for i in range(bins):
        keep = (confidence >= i / bins) & ((confidence < (i + 1) / bins) if i < bins - 1 else (confidence <= 1))
        if keep.any():
            c, a = float(confidence[keep].mean()), float(correct[keep].float().mean())
            ece += float(keep.float().mean()) * abs(c - a)
            reliability.append({'bin': i, 'count': int(keep.sum()), 'confidence': c, 'accuracy': a})
    order = np.argsort(score, kind='stable')
    risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    # Store a bounded curve, computing AURC on all retained tokens.
    indices = np.unique(np.linspace(0, len(errors) - 1, min(100, len(errors)), dtype=int))
    return {'unit': 'BIO token; includes O; PII-only metrics reported separately',
            'ece': ece, 'nll': float(-p[torch.arange(len(y)), y].clamp_min(1e-12).log().mean()),
            'brier': float(((p - torch.nn.functional.one_hot(y, p.shape[-1]))**2).sum(-1).mean()),
            'error_auroc': float(roc_auc_score(errors, score)) if len(np.unique(errors)) == 2 else None,
            'error_average_precision': float(average_precision_score(errors, score)) if errors.sum() else None,
            'aurc': float(risk.mean()), 'reliability_bins': reliability,
            'risk_coverage': [{'coverage': float((i + 1) / len(errors)), 'risk': float(risk[i])} for i in indices],
            'tokens': len(y)}


def plot_metrics(comparisons, directory):
    # Standalone vector report avoids a plotting-library dependency for headless runs.
    from html import escape
    colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#d97706', '#0891b2', '#be185d']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="540" viewBox="0 0 1040 540">',
           '<rect width="1040" height="540" fill="white"/>', '<g font-family="sans-serif" font-size="13">']
    for panel, title, xlabel, ylabel in [(0, 'Validation reliability', 'Confidence', 'Accuracy'),
                                         (1, 'Validation risk–coverage', 'Coverage', 'Error rate')]:
        x0 = 60 + panel * 500
        svg.append(f'<text x="{x0}" y="25">{title}</text><path d="M{x0} 50 V400 H{x0+400}" fill="none" stroke="black"/>')
        svg.append(f'<text x="{x0+160}" y="440">{xlabel}</text><text x="{x0}" y="45">{ylabel}</text>')
        for tick in (0, .25, .5, .75, 1):
            svg.append(f'<text x="{x0+400*tick}" y="420">{tick:g}</text><text x="{x0-35}" y="{400-350*tick}">{tick:g}</text>')
        if panel == 0:
            svg.append(f'<path d="M{x0} 400 L{x0+400} 50" stroke="gray" stroke-dasharray="5,5"/>')
        for i, (name, result) in enumerate(comparisons.items()):
            points = result['metrics']['reliability_bins' if panel == 0 else 'risk_coverage']
            coordinates = ' '.join(f"{x0+400*p['confidence' if panel == 0 else 'coverage']:.2f},{400-350*p['accuracy' if panel == 0 else 'risk']:.2f}" for p in points)
            svg.append(f'<polyline points="{coordinates}" stroke="{colors[i % len(colors)]}" fill="none" stroke-width="2"/>')
    for i, name in enumerate(comparisons):
        svg.append(f'<text x="{60+(i%4)*245}" y="{480+(i//4)*25}" fill="{colors[i % len(colors)]}">{escape(name)}</text>')
    svg.append('</g></svg>')
    (directory / 'reliability.svg').write_text('\n'.join(svg), encoding='utf-8')
