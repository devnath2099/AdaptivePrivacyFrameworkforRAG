"""Configurable design assumptions, not external risk ground truth.

All entity predictions from one detector are one dependent evidence group.
They are pooled by a convex mixture before DS combination. Confidence and
uncertainty are not combined as separate mass sources.
"""
from .dempster_shafer import FRAME, validate_mass


def evidence(reliability, config):
    groups, trace, values = {}, [], []
    for entity in reliability['entities']:
        kind = entity['type']
        tier = config['type_tiers'].get(kind, config['unknown_type_tier'])
        confidence = float(entity['calibrated_confidence'])
        if tier not in range(4) or not 0 <= confidence <= 1:
            raise ValueError('Invalid risk tier/confidence')
        source = entity.get('evidence_group', 'neural_detector')
        weight = float(config.get('type_weights', {}).get(kind, 1.))
        if weight <= 0:
            raise ValueError('Evidence weights must be positive')
        mass = {1 << tier: confidence, FRAME: 1 - confidence}
        groups.setdefault(source, []).append((weight, mass))
        values.append({'tier': tier, 'confidence': confidence, 'weight': weight})
        trace.append({'type': kind, 'tier': tier, 'confidence': confidence, 'group': source,
                      'mapping': 'explicit' if kind in config['type_tiers'] else 'unknown_type_default',
                      'basis': config['basis']})
    pooled = []
    for source, items in groups.items():
        total = sum(w for w, _ in items)
        mass = {}
        for weight, item in items:
            for key, value in item.items():
                mass[key] = mass.get(key, 0.) + weight * value / total
        validate_mass(mass)
        pooled.append({'group': source, 'mass': mass})
    if not pooled:
        # No detection is not proof of absence; calibrated O confidence discounts Low.
        confidence = float(reliability['confidence'])
        if not 0 <= confidence <= 1:
            raise ValueError('Invalid absence confidence')
        pooled = [{'group': 'neural_detector', 'mass': {1: confidence, FRAME: 1 - confidence}}]
        values = [{'tier': 0, 'confidence': confidence, 'weight': 1.}]
    return pooled, trace, values
