from .constraints import requirement, evaluate_constraints


def fallback(risk, required, candidates, reason):
    return {'record_id': risk['record_id'], 'split': risk['split'], 'policy': None,
            'action': 'block', 'feasible': False, 'fallback': True, 'reason': reason,
            'required_protection': required, 'candidates': candidates, 'enforcement_executed': False}


def select(risk, policies, profiles, config, fixed=None):
    required = requirement(risk, config['uncertainty_strength'])
    if fixed is not None and fixed not in policies:
        raise ValueError('Unknown fixed-policy baseline')
    candidates = []
    for name, policy in policies.items():
        if fixed is not None and name != fixed:
            continue
        if name not in profiles:
            candidates.append({'policy_id': name, 'feasible': False, 'reason': 'unprofiled'})
            continue
        checks = evaluate_constraints(policy, profiles[name], risk, required, config['utility_min'], config['max_latency_ms'])
        candidates.append({'policy_id': name, **checks, 'latency_ms': profiles[name]['latency_ms']})
    feasible = [r for r in candidates if r['feasible']]
    if not feasible:
        return fallback(risk, required, candidates, 'no_policy_satisfies_all_constraints')
    selected = min(feasible, key=lambda r: (r['latency_ms'], r['policy_id']))
    name = selected['policy_id']
    return {'record_id': risk['record_id'], 'split': risk['split'], 'policy': policies[name],
            'profile': profiles[name], 'action': 'apply_policy', 'feasible': True, 'fallback': False,
            'required_protection': required, 'candidates': candidates, 'enforcement_executed': False,
            'reason': 'minimum profiled latency among feasible policies' if fixed is None else 'fixed baseline with constraint checks'}
