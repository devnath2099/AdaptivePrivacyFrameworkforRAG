import torch
import pytest
from review2.m3.fgsm import perturb, adversarial_loss
from review2.m3.attacks import ATTACKS, attack_record
from review2.m1.validation import validate_record
from review2.m2.alignment import encode, collate
from review2.m2.model import build


def test_fgsm_bound_zero_and_loss(tokenizer, record):
    labels = ['O', 'B-EMAIL', 'I-EMAIL']
    model = build({'tiny': True, 'family': 'deberta'}, labels, tokenizer)
    batch = collate(encode([record], tokenizer, labels, 32), tokenizer.pad_token_id)
    _, delta = perturb(model, batch, .02)
    assert delta.abs().max() <= .020001 and delta.abs().max() > 0
    assert not delta[batch['labels'] == -100].any()
    _, zero = perturb(model, batch, 0)
    assert not zero.any()
    with pytest.raises(ValueError):
        perturb(model, batch, -1)
    loss = adversarial_loss(model, batch, .02, .5)
    loss.backward()
    assert model.classifier.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('name', ATTACKS)
def test_attack_preserves_span_type_and_parent(record, name):
    original = record['text']
    attacked = attack_record(record, name)
    validate_record(attacked)
    assert attacked['parent_record_id'] == record['record_id']
    assert attacked['split'] == record['split']
    assert [s['type'] for s in attacked['canonical_spans']] == ['EMAIL']
    assert record['text'] == original
    assert attacked == attack_record(record, name)


def test_multiple_span_offset_repair(record):
    record['text'] += ' call 1234567890'
    record['canonical_spans'].append({'start': 29, 'end': 39, 'type': 'PHONE_NUMBER'})
    result = attack_record(record, 'email_obfuscation')
    validate_record(result)
    assert result['text'][result['canonical_spans'][1]['start']:result['canonical_spans'][1]['end']] == '1234567890'
