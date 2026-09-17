import copy
import pytest
from review2.m5.dempster_shafer import validate_mass, combine, summarize
from review2.m5.aggregation import aggregate


def test_mass_validation_and_belief():
    with pytest.raises(ValueError):
        validate_mass({0: 1.})
    with pytest.raises(ValueError):
        validate_mass({1: .3})
    summary = summarize({1: .4, 3: .2, 15: .4})
    assert summary['belief']['Low'] == .4
    assert summary['plausibility']['Low'] == 1.
    assert sum(summary['probabilities']) == pytest.approx(1)


def test_combination_known_answer_and_conflict():
    mass, conflict, failed = combine({1: .6, 15: .4}, {1: .5, 15: .5})
    assert mass[1] == pytest.approx(.8) and conflict == 0 and not failed
    mass, conflict, failed = combine({1: 1.}, {8: 1.})
    assert mass == {15: 1.} and conflict == 1 and failed


def test_baselines_and_dependent_evidence(config, reliability):
    cfg = config['m5']
    maximum = aggregate(reliability, cfg, 'max')
    assert maximum['score'] == pytest.approx(.6)
    assert aggregate(reliability, cfg, 'weighted')['score'] == pytest.approx(.6)
    result = aggregate(reliability, cfg, 'ds')
    duplicate = copy.deepcopy(reliability)
    duplicate['entities'] *= 5
    assert aggregate(duplicate, cfg, 'ds')['score'] == pytest.approx(result['score'])
    assert result['independent_group_count'] == 1


def test_independence_must_be_declared(config, reliability):
    reliability['entities'].append({'type': 'NAME', 'calibrated_confidence': .7, 'evidence_group': 'second'})
    with pytest.raises(ValueError):
        aggregate(reliability, config['m5'])


def test_no_entities_preserves_ignorance(config, reliability):
    reliability.update(entities=[], confidence=0.)
    assert aggregate(reliability, config['m5'])['ignorance'] == 1.
