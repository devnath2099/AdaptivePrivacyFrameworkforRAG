import math


def requirement(risk, uncertainty_strength):
    if not 0 <= uncertainty_strength <= 1:
        raise ValueError('Uncertainty strength must be in [0,1]')
    values = [risk['score'], risk['uncertainty'], risk['ignorance']]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('Risk contract requires finite normalized values')
    uncertainty = max(risk['uncertainty'], risk['ignorance'])
    # Configurable conservative interpolation to maximum protection, not learned truth.
    return risk['score'] + (1 - risk['score']) * uncertainty_strength * uncertainty


def evaluate_constraints(policy, profile, risk, required, utility_min, max_latency_ms):
    if not 0 <= utility_min <= 1 or not math.isfinite(max_latency_ms) or max_latency_ms < 0:
        raise ValueError('Invalid utility/latency constraints')
    categories = set(policy['protected_types'])
    covered = '*' in categories or set(risk['entity_types']) <= categories
    checks = {'privacy': profile['privacy'] >= required, 'utility': profile['utility'] >= utility_min,
              'latency': profile['latency_ms'] <= max_latency_ms, 'categories': covered,
              'coverage': policy['coverage'] >= required}
    return {'feasible': all(checks.values()), 'checks': checks,
            'privacy_margin': profile['privacy'] - required, 'utility_margin': profile['utility'] - utility_min}
