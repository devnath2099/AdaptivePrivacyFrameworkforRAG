from .selection import select
from .constraints import requirement, evaluate_constraints


def compare(risks, policies, profiles, config):
    results = {}
    for mode in ['no_privacy', *policies, 'adaptive']:
        decisions, violations, cost = [], 0, 0.
        for risk in risks:
            if mode == 'no_privacy':
                violated = requirement(risk, config['uncertainty_strength']) > 0
                decisions.append({'record_id': risk['record_id'], 'mode': mode, 'constraint_violation': violated})
                violations += violated
            elif mode == 'adaptive':
                decision = select(risk, policies, profiles, config)
                decisions.append(decision)
                cost += decision.get('profile', {}).get('latency_ms', 0.)
            else:
                # Evaluate fixed policies as actually fixed, recording unmet constraints rather than adapting them.
                checks = evaluate_constraints(policies[mode], profiles[mode], risk, requirement(risk, config['uncertainty_strength']),
                                              config['utility_min'], config['max_latency_ms'])
                violations += not checks['feasible']
                cost += profiles[mode]['latency_ms']
                decisions.append({'record_id': risk['record_id'], 'policy_id': mode, **checks})
        results[mode] = {'decisions': decisions, 'constraint_violations': violations,
                         'fallbacks': sum(d.get('fallback', False) for d in decisions),
                         'profiled_latency_total_ms': cost, 'scope': 'policy profile simulation; no enforcement/RAG measured'}
    return results


def sensitivity(risks, policies, profiles, config):
    report = []
    for strength in config['uncertainty_grid']:
        for utility in config['utility_grid']:
            cfg = {**config, 'uncertainty_strength': strength, 'utility_min': utility}
            decisions = [select(r, policies, profiles, cfg) for r in risks]
            report.append({'uncertainty_strength': strength, 'utility_min': utility,
                           'fallbacks': sum(d['fallback'] for d in decisions),
                           'selected_ids': [d['policy']['id'] if d['policy'] else 'block' for d in decisions]})
    return report
