import copy
import json
import numpy as np
import pytest
import torch
import yaml

from validated.common import path, digest, require_match
from validated.corpus import record, tatqa_rows, split_exact, check_isolation
from validated.evidence import extract
from validated.labels import votes, majority, synthesize, unanimous_fallback
from validated.model import loss, checkpoint_contract, validate_checkpoint
from validated.adversarial import perturb, adversarial_loss
from validated.calibration import fit_temperature, mc_nll, uncertainty, ece
from validated.benchmark import generate, augment, verify_separation


def query(text, context=''):
    r = record(text, context, 'fixture', 'irrelevant', digest(text), digest(text), 'fixture')
    r['evidence'] = extract(r)
    return r


def test_context_only_pii_cannot_change_evidence_or_labels():
    a = query('What is diabetes?')
    b = query('What is diabetes?', 'Patient Alice in London; password: secret42; alice@example.org; account number: 123456789')
    assert a['evidence'] == b['evidence']
    assert votes(a) == votes(b)
    assert 2 not in votes(a)['sensitivity'].values()


@pytest.mark.parametrize('text,task,label', [
    ('What does my HbA1c result mean?', 'privacy_relevant_intent', 2),
    ('Why was my account debited?', 'privacy_relevant_intent', 3),
    ('What was the company revenue in 2024?', 'privacy_relevant_intent', 0),
    ('What is diabetes?', 'sensitivity', 0),
    ('Show my private records.', 'sensitivity', 2),
])
def test_semantic_examples(text, task, label):
    assert label in votes(query(text))[task].values()


def test_financial_rules_require_personal_financial_artifact():
    assert 3 in votes(query('Why was my account debited?'))['privacy_relevant_intent'].values()
    assert 3 not in votes(query('The blood bank said I am anemic.'))['privacy_relevant_intent'].values()
    assert 3 not in votes(query('My physical balance is poor after surgery.'))['privacy_relevant_intent'].values()


def test_medium_requires_limited_personal_topic_not_generic_medical_term():
    assert 1 in votes(query('I have a medical question about preventive wellness.'))['sensitivity'].values()
    assert 1 not in votes(query('What is diabetes?'))['sensitivity'].values()


def test_unanimous_fallback_only_preserves_direct_suppressed_support():
    p = np.array([[.8, .1, .1], [.8, .1, .1], [.8, .1, .1]])
    matrix = np.array([[2, 2], [2, -1], [-1, -1]])
    result, provenance = unanimous_fallback(p.copy(), matrix, {2}, True)
    assert np.array_equal(result[0], [0, 0, 1])
    assert np.array_equal(result[1], [0, 0, 1])
    assert np.array_equal(result[2], p[2])
    assert provenance.tolist() == ['unanimous_direct_lf_fallback', 'unanimous_direct_lf_fallback', 'snorkel']


def test_diagnostics_report_per_class_support_and_aggregation(cfg):
    rows = [query('What is diabetes?'), query('I have a medical question about preventive wellness.'),
            query('Why was my account debited?'), query('blorp')] * 4
    result, _ = synthesize({s: rows for s in ['train', 'calibration', 'validation']}, cfg)
    detail = result['train']['diagnostics']['sensitivity']['all']
    assert set(detail['class_support']) == {'low', 'medium', 'high'}
    assert 'lf_firing_count' in detail
    assert 'aggregation_provenance' in detail


def test_source_domain_never_changes_labels():
    r = query('What was the company revenue in 2024?')
    a = votes(r)
    r.update(domain='medical', source_dataset='healthcaremagic')
    assert votes(r) == a


def test_unknown_abstains_and_entities_alone_not_threat():
    r = query('Alice London')
    r['evidence']['entities'] = [{'label': 'PERSON'}, {'label': 'GPE'}]
    assert all(x == -1 for d in votes(r).values() for x in d.values())


@pytest.mark.parametrize('text,target', [
    ('Identify the anonymous person in record X.', 're_identification'),
    ('Infer the confidential diagnosis of person X.', 'attribute_inference'),
    ('Is person X in the hospital cohort database?', 'membership_inference'),
])
def test_threat_requires_request(text, target):
    assert 1 in votes(query(text))[f'threat_content/{target}'].values()


def test_tatqa_question_only():
    docs = [{'table': {'uid': 'doc1', 'table': [['secret@example.org']]}, 'paragraphs': [],
             'questions': [{'uid': 'q1', 'question': 'What was revenue?'}]}]
    r = list(tatqa_rows(docs, 'train'))[0]
    assert r['query_text'] == 'What was revenue?'
    assert 'secret@example.org' in r['context_text']
    assert extract(r)['patterns']['email'] == []
    assert r['group_id'] == 'doc1'


def test_manifest_disables_fallback():
    m = yaml.safe_load(path('configs/sources.v1.yaml').read_text())
    assert m['allow_synthetic_fallback'] is False
    assert set(m['sources']) == {'healthcaremagic', 'natural_questions', 'tat_qa'}
    assert sum(s['cap'] for s in m['sources'].values()) == 50000
    assert all(len(s['revision']) == 40 for s in m['sources'].values())


def natural_fixture():
    return [record(f'{source} question {i}', '', source, source, i, i // 2, 'train')
            for source in ['a', 'b'] for i in range(60)]


def test_deterministic_group_split_and_duplicate_leakage():
    rows = natural_fixture()
    duplicate = dict(rows[0], record_id='dup', source_record_id='dup', group_id='newgroup')
    rows += [duplicate]
    a = split_exact(rows, {'a': 40, 'b': 40}, 42)
    b = split_exact(list(reversed(rows)), {'a': 40, 'b': 40}, 42)
    assert a == b
    assert [len(a[s]) for s in ['train','calibration','validation','test']] == [56,8,8,8]
    check_isolation(a)
    group_splits = {}
    for s, rs in a.items():
        for r in rs:
            key = (r['source_dataset'], r['group_id'])
            assert group_splits.setdefault(key, s) == s


def test_deficit_is_fatal():
    with pytest.raises(ValueError, match='deficit'):
        split_exact(natural_fixture(), {'a': 100, 'b': 100})


def test_synthetic_train_only(cfg):
    splits = split_exact(natural_fixture(), {'a': 40, 'b': 40})
    c = dict(enabled=True, count=2, max_synthetic_training_examples=1500, max_ratio_to_natural_train=.05)
    result = augment(splits, c, cfg['active_tasks'])
    assert len(result['train']) == len(splits['train']) + 2
    assert result['test'] == splits['test']
    result['calibration'].append(generate('train', 1)[0])
    with pytest.raises(ValueError, match='Synthetic'):
        check_isolation(result)


def test_dynamic_shapes_freezing_and_masks(model, batch, cfg):
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    assert {k: list(v.shape) for k,v in output.items()} == {'sensitivity':[4,3], 'privacy_relevant_intent':[4,5], 'threat_content':[4,3]}
    assert not any(p.requires_grad for p in model.encoder.parameters())
    blank = {k: torch.zeros_like(v) for k,v in batch['observed'].items()}
    assert loss(output, batch['targets'], blank, model.tasks, cfg['task_weights']).item() == 0


def test_checkpoint_taxonomy_compatibility(model, cfg):
    contract = checkpoint_contract(cfg, 'split', 'source', None, 'm3')
    c = {'contract':contract, 'model':model.state_dict()}
    validate_checkpoint(c, contract)
    changed = copy.deepcopy(contract)
    changed['active_tasks']['sensitivity']['classes'].reverse()
    with pytest.raises(ValueError, match='Incompatible'):
        validate_checkpoint(c, changed)


def test_fgsm_all_heads_padding_and_eval_gradients(model, batch, cfg):
    model.eval()
    with torch.no_grad():
        adv, delta = perturb(model, batch, cfg)
    assert not adv.requires_grad and not delta.requires_grad
    assert delta.abs().max() <= float(cfg['m4']['epsilon']) + 1e-9
    assert torch.all(delta[:, 5:] == 0)
    adversarial_loss(model, batch, cfg).backward()
    assert all(h.weight.grad is not None and h.weight.grad.abs().sum() > 0 for h in model.heads.values())
    assert all(p.grad is None for p in model.encoder.parameters())


@pytest.mark.parametrize('kind,k', [('categorical',3),('multilabel',3)])
def test_temperature_positive_and_improves_nll(kind,k):
    torch.manual_seed(10)
    samples = torch.randn(3,40,k) * 6
    targets = torch.softmax(torch.randn(40,k),-1) if kind == 'categorical' else torch.rand(40,k)
    mask = torch.ones(40) if kind == 'categorical' else torch.ones(40,k)
    result = fit_temperature(samples, targets, mask, kind, 'calibration', 30)
    assert result['temperature'] > 0
    assert mc_nll(samples,targets,mask,kind,result['temperature']) <= mc_nll(samples,targets,mask,kind) + 1e-5
    with pytest.raises(ValueError, match='calibration'):
        fit_temperature(samples,targets,mask,kind,'validation')


def test_uncertainty_and_population_ece():
    samples = torch.zeros(20,2,3)
    u = uncertainty(samples,'categorical')
    assert torch.allclose(u['normalized_entropy'],torch.ones(2))
    assert u['predictive_variance'].sum() == 0
    p = torch.tensor([[.8,.2],[.8,.2]])
    y = torch.tensor([[1.,0.],[0.,1.]])
    assert ece(p,y,torch.ones(2),'categorical',10) == pytest.approx(.3)


def test_benchmark_separation_and_verified_labels():
    verify_separation()
    tr, te = generate('train',80), generate('heldout',80)
    assert not {r['query_text'] for r in tr} & {r['query_text'] for r in te}
    for r in tr + te:
        assert all(k in r for k in ['scenario_id','rule_id','template_id','lexical_set_id','label_provenance','future_expected_policy'])
        assert r['label_source'] == 'synthetic_verified'
        assert sum(r['targets']['sensitivity']) == 1


def test_resume_lineage_requires_exact_match():
    a = dict(source='a', config='b', split='c', parent='d')
    require_match(a, dict(a))
    for key in a:
        with pytest.raises(ValueError, match='Incompatible'):
            require_match(a, dict(a, **{key:'changed'}))


def test_majority_retains_all_abstain_and_snorkel_train_fit(cfg):
    m = np.array([[-1,-1,-1], [1,1,0]])
    assert np.allclose(majority(m,2),[[.5,.5],[1/3,2/3]])
    texts = ['What is diabetes?', 'Why was my account debited?', 'What does my HbA1c result mean?',
             'Identify the anonymous person in record X.', 'Infer the confidential salary of person X.',
             'Is person X in the hospital dataset?', 'financial advice', 'blorp'] * 5
    splits = {s:[query(t) for t in texts] for s in ['train','calibration','validation']}
    result,_ = synthesize(splits,cfg)
    unknown = result['train']['records'][7]
    assert unknown['observed']['sensitivity'] == 0
    assert unknown['observed']['threat_content'] == [0,0,0]
    with pytest.raises(ValueError, match='test'):
        synthesize(dict(splits,test=splits['validation']),cfg)
