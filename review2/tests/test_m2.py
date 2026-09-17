import torch
import pytest
from review2.m2.alignment import align, decode, encode, collate
from review2.m2.loss import token_loss, class_weights
from review2.m2.metrics import entity_metrics
from review2.m2.model import build
from review2.m2.training import train, load_checkpoint

LABELS = ['O', 'B-EMAIL', 'I-EMAIL']


def test_bio_and_special_mask():
    offsets = [(0, 0), (0, 2), (2, 5), (6, 9), (0, 0)]
    labels = align(offsets, [{'start': 0, 'end': 5, 'type': 'EMAIL'}], dict(zip(LABELS, range(3))))
    assert labels == [-100, 1, 2, 0, -100]
    spans = decode([0, 1, 2, 0, 0], offsets, LABELS)
    assert (spans[0]['start'], spans[0]['end']) == (0, 5)


def test_boundary_token_mask():
    assert align([(0, 5)], [{'start': 1, 'end': 4, 'type': 'EMAIL'}], dict(zip(LABELS, range(3)))) == [-100]


def test_weighted_loss_and_padding_gradient():
    z = torch.tensor([[[2., 0., 0.], [0., 2., 0.], [100., -100., 0.]]], requires_grad=True)
    y = torch.tensor([[1, 1, -100]])
    w = torch.tensor([1., 3., 1.])
    actual = token_loss(z, y, w)
    expected = torch.nn.functional.cross_entropy(z[0, :2], y[0, :2], weight=w)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert not z.grad[0, 2].any()
    assert token_loss(z, torch.full_like(y, -100)) == 0


def test_train_only_class_weights():
    weights = class_weights([{'labels': [0, 0, 0, 0, 1, -100]}], 3)
    assert weights[1] > weights[0] and torch.isfinite(weights).all()


def test_strict_entity_metrics(record):
    perfect = entity_metrics([record], [record['canonical_spans']])
    assert perfect['f1'] == 1
    wrong = entity_metrics([record], [[{'start': 6, 'end': 22, 'type': 'EMAIL'}]])
    assert wrong['f1'] == 0 and wrong['fp'] == wrong['fn'] == 1


def test_train_checkpoint_resume_and_split_guard(tokenizer, record, tmp_path):
    spec = {'tiny': True, 'family': 'deberta'}
    model = build(spec, LABELS, tokenizer)
    training = encode([record], tokenizer, LABELS, 32)
    validation = encode([dict(record, split='validation')], tokenizer, LABELS, 32)
    cfg = {'seed': 42, 'loss': 'weighted', 'epochs': 1, 'patience': 2, 'batch_size': 1,
           'learning_rate': .001, 'weight_decay': 0., 'grad_clip': 1., 'checkpoint_every': 1}
    contract = {'stage': 'test', 'labels': LABELS}
    train(model, tokenizer, LABELS, training, validation, cfg, tmp_path / 'training', contract)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    train(model, tokenizer, LABELS, training, validation, cfg, tmp_path / 'training', contract)
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / 'training/best.pt', {'different': True})
    with pytest.raises(ValueError):
        train(model, tokenizer, LABELS, validation, validation, cfg, tmp_path / 'bad', contract)


def test_interrupted_batch_resume_matches_uninterrupted(tokenizer, record, tmp_path, monkeypatch):
    from review2.m2 import training as module
    from review2.common import seed_all
    cfg = {'seed': 42, 'loss': 'ce', 'epochs': 1, 'patience': 2, 'batch_size': 1,
           'learning_rate': .001, 'weight_decay': 0., 'grad_clip': 1., 'checkpoint_every': 1}
    data = encode([record] * 3, tokenizer, LABELS, 32)
    validation = encode([dict(record, split='validation')], tokenizer, LABELS, 32)
    seed_all(42)
    baseline = build({'tiny': True, 'family': 'deberta'}, LABELS, tokenizer)
    train(baseline, tokenizer, LABELS, data, validation, cfg, tmp_path / 'baseline', {'test': 'resume'})
    seed_all(42)
    resumed = build({'tiny': True, 'family': 'deberta'}, LABELS, tokenizer)
    save = module.atomic_save
    def interrupted(value, path):
        save(value, path)
        if path.name == 'last.pt' and value['offset'] == 1:
            raise RuntimeError('simulated interruption after atomic save')
    monkeypatch.setattr(module, 'atomic_save', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(resumed, tokenizer, LABELS, data, validation, cfg, tmp_path / 'resumed', {'test': 'resume'})
    monkeypatch.setattr(module, 'atomic_save', save)
    train(resumed, tokenizer, LABELS, data, validation, cfg, tmp_path / 'resumed', {'test': 'resume'})
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in baseline.state_dict().items())
