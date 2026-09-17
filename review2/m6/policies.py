import itertools
import math


def validate_policy(policy):
    required = {'id', 'epsilon', 'coverage', 'protected_types', 'mechanisms', 'fallback'}
    if not required <= policy.keys():
        raise ValueError('Incomplete policy schema')
    if not math.isfinite(policy['epsilon']) or policy['epsilon'] <= 0:
        raise ValueError('epsilon must be positive and finite')
    if not 0 <= policy['coverage'] <= 1 or not isinstance(policy['protected_types'], list):
        raise ValueError('Invalid coverage/categories')
    if set(policy['mechanisms']) != {'word', 'number', 'phrase'} or set(policy['mechanisms'].values()) - {'WPM', 'NPM', 'PPM'}:
        raise ValueError('Unsupported M7 mechanism contract')
    if policy['fallback'] not in ('block', 'manual_review'):
        raise ValueError('Policy needs an explicit safe fallback')


def generate_candidates(space):
    policies = []
    for epsilon, coverage, categories in itertools.product(space['epsilons'], space['coverages'], space['category_sets']):
        policy = {'id': f'eps{epsilon:g}_cov{coverage:g}_types{len(policies)}', 'epsilon': epsilon,
                  'coverage': coverage, 'protected_types': categories, 'mechanisms': {'word': 'WPM', 'number': 'NPM', 'phrase': 'PPM'},
                  'fallback': space['fallback']}
        validate_policy(policy)
        policies.append(policy)
    return policies


def repository(policies):
    output = {}
    for policy in policies:
        validate_policy(policy)
        if policy['id'] in output:
            raise ValueError('Duplicate policy id')
        output[policy['id']] = policy
    if not output:
        raise ValueError('Empty policy repository')
    return output
