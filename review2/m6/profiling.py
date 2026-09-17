"""Profile supplied validation measurements or explicitly marked controlled observations.

The callback interface allows later M7/M8 measurements without implementing
enforcement or RAG here. No epsilon-to-privacy guarantee is fabricated.
"""
import math
from collections import defaultdict


def collect_profiles(policies, cases, evaluator):
    observations = []
    for case in cases:
        if case['split'] != 'validation':
            raise ValueError('Policy profiling requires validation cases')
        for policy in policies:
            observed = evaluator(policy, case)
            observations.append({**observed, 'policy_id': policy['id'], 'case_id': case['record_id'], 'split': 'validation'})
    return observations


def profile(observations, policies, allow_controlled=False):
    groups = defaultdict(list)
    for row in observations:
        if row['split'] != 'validation' or row['policy_id'] not in policies:
            raise ValueError('Profiles require known policies and validation-only measurements')
        if row['provenance'] not in ('measured', 'controlled_fixture'):
            raise ValueError('Missing measurement provenance')
        if row['provenance'] == 'controlled_fixture' and not allow_controlled:
            raise ValueError('Controlled profiles are forbidden for measured runs')
        if not row.get('experiment_id'):
            raise ValueError('Profile needs experiment provenance')
        if any(not math.isfinite(row[k]) for k in ('privacy', 'utility', 'latency_ms')):
            raise ValueError('Nonfinite profile value')
        if not 0 <= row['privacy'] <= 1 or not 0 <= row['utility'] <= 1 or row['latency_ms'] < 0:
            raise ValueError('Invalid profile value')
        groups[row['policy_id']].append(row)
    result = {}
    for name, rows in groups.items():
        result[name] = {'privacy': min(r['privacy'] for r in rows), 'utility': min(r['utility'] for r in rows),
                        'latency_ms': max(r['latency_ms'] for r in rows), 'observations': len(rows),
                        'provenance': sorted({r['provenance'] for r in rows}),
                        'experiments': sorted({r['experiment_id'] for r in rows}),
                        'estimator': 'observed minimum privacy/utility and maximum latency; not a statistical guarantee'}
    return result
