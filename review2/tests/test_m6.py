import pytest
from review2.common import read_json, ROOT
from review2.m6.policies import repository, generate_candidates, validate_policy
from review2.m6.profiling import profile, collect_profiles
from review2.m6.constraints import requirement
from review2.m6.selection import select


def setup(config):
    policies = repository(config['m6']['policies'])
    profiles = profile(read_json(ROOT / 'configs/controlled_profiles.json'), policies, True)
    return policies, profiles


def risk(score=.4, u=0.):
    return {'record_id': 'x', 'split': 'validation', 'score': score, 'uncertainty': u, 'ignorance': 0., 'entity_types': ['EMAIL']}


def test_candidate_generator_and_schema():
    candidates = generate_candidates({'epsilons': [.5, 2.], 'coverages': [.5, 1.], 'category_sets': [['*']], 'fallback': 'block'})
    assert len(repository(candidates)) == 4
    candidates[0]['epsilon'] = 0
    with pytest.raises(ValueError):
        validate_policy(candidates[0])


def test_optimum_satisfies_all_constraints(config):
    policies, profiles = setup(config)
    result = select(risk(), policies, profiles, config['m6'])
    assert result['policy']['id'] == 'medium'
    assert result['feasible'] and not result['enforcement_executed']
    assert all(next(r for r in result['candidates'] if r['policy_id'] == 'medium')['checks'].values())


def test_infeasible_fallback_and_fixed_baseline(config):
    policies, profiles = setup(config)
    assert select(risk(.99), policies, profiles, config['m6'])['action'] == 'block'
    assert select(risk(.4), policies, profiles, config['m6'], fixed='weak')['fallback']
    assert select(risk(.2), policies, profiles, config['m6'], fixed='strong')['policy']['id'] == 'strong'


def test_uncertainty_monotone_and_categories(config):
    policies, profiles = setup(config)
    assert requirement(risk(.4, 1), .5) > requirement(risk(.4, 0), .5)
    for p in policies.values():
        p['protected_types'] = ['PHONE']
    assert select(risk(), policies, profiles, config['m6'])['fallback']


def test_profiles_never_test_or_silent_simulation(config):
    policies, _ = setup(config)
    observations = read_json(ROOT / 'configs/controlled_profiles.json')
    with pytest.raises(ValueError):
        profile(observations, policies, False)
    observations[0]['split'] = 'test'
    with pytest.raises(ValueError):
        profile(observations, policies, True)


def test_profile_callback(config):
    policies, _ = setup(config)
    observed = collect_profiles(list(policies.values()), [{'record_id': 'a', 'split': 'validation'}],
                                lambda p, r: {'privacy': .5, 'utility': .8, 'latency_ms': 1., 'provenance': 'measured', 'experiment_id': 'measurement'})
    assert len(profile(observed, policies)) == 3
