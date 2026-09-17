from .evidence import evidence
from .dempster_shafer import combine, summarize, FRAME, TIERS


def aggregate(reliability, config, method='ds'):
    if method not in ('max', 'weighted', 'ds'):
        raise ValueError('Unknown risk aggregator')
    groups, trace, values = evidence(reliability, config)
    conflicts, fallback = [], False
    if method == 'ds':
        mass = groups[0]['mass']
        independent = set(config.get('independent_groups', []))
        for group in groups[1:]:
            if group['group'] not in independent or groups[0]['group'] not in independent:
                raise ValueError('DS fusion of separate sources needs explicit independence justification')
            mass, conflict, failed = combine(mass, group['mass'], config['conflict_limit'])
            conflicts.append(conflict)
            fallback |= failed
            if failed:
                break
        result = summarize(mass)
        result['masses'] = {str(k): v for k, v in mass.items()}
    else:
        # Discounted tier score; uncommitted confidence is reported separately.
        scores = [v['tier'] / 3 * v['confidence'] for v in values]
        if method == 'max':
            score = max(scores)
        else:
            score = sum(s * v['weight'] for s, v in zip(scores, values)) / sum(v['weight'] for v in values)
        uncertainty = sum((1 - v['confidence']) * v['weight'] for v in values) / sum(v['weight'] for v in values)
        tier = min(3, int(score * 3 + .5))
        result = {'score': score, 'tier': TIERS[tier], 'ignorance': uncertainty,
                  'probabilities': None, 'belief': None, 'plausibility': None, 'masses': None}
    return {'record_id': reliability['record_id'], 'split': reliability['split'], 'method': method,
            **result, 'uncertainty': reliability['uncertainty'], 'entity_types': sorted({s['type'] for s in reliability['entities']}),
            'entity_count': len(reliability['entities']), 'type_diversity': len({s['type'] for s in reliability['entities']}),
            'conflicts': conflicts, 'conflict_fallback': fallback, 'evidence_trace': trace,
            'assumption_basis': config['basis'], 'independent_group_count': len(groups)}
