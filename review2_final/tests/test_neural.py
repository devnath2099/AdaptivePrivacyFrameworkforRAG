"""Tiny random networks are unit fixtures only, never research evidence."""
import numpy as np
import torch
from transformers import DebertaConfig, DebertaForTokenClassification
from review2_final.neural import fgsm_embeddings, dropout_on, predict_logits, loader, encode


def tiny():
    return DebertaForTokenClassification(DebertaConfig(vocab_size=30, hidden_size=24, num_hidden_layers=1,
          num_attention_heads=4, intermediate_size=32, max_position_embeddings=64, num_labels=3,
          hidden_dropout_prob=.2, attention_probs_dropout_prob=.2))


def test_fgsm_bound_padding_and_gradients():
    model = tiny().eval()
    x = {"input_ids": torch.tensor([[1, 3, 4, 2, 0]]), "attention_mask": torch.tensor([[1,1,1,1,0]]),
         "labels": torch.tensor([[-100, 1, 2, -100, -100]])}
    base = model.get_input_embeddings()(x["input_ids"]).detach()
    adv = fgsm_embeddings(model, x, .01)
    difference = adv-base
    assert difference.abs().max() <= .010001
    assert torch.equal(adv[:, [0,3,4]], base[:, [0,3,4]])
    assert difference[:, 1:3].abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_mc_dropout_is_stochastic_and_eval_restorable():
    model = tiny()
    assert dropout_on(model) > 0
    x = torch.tensor([[1,3,4,2]])
    with torch.no_grad():
        a, b = model(x).logits, model(x).logits
    assert not torch.equal(a, b)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x).logits, model(x).logits)


def test_reassembly_scores_native_tokens_once():
    model = tiny()
    windows = [{"input_ids": [1,3,4,2], "labels": [-100,1,2,-100], "words": [-1,0,1,-1], "doc": 0},
               {"input_ids": [1,5,2], "labels": [-100,0,-100], "words": [-1,2,-1], "doc": 0}]
    class Tokenizer:
        pad_token_id = 0
    batches = loader(windows, Tokenizer(), 2)
    logits = predict_logits(model, batches, [{"tokens": ["a","b","c"]}], 3, "cpu")
    assert logits[0].shape == (3,3)
    assert np.isfinite(logits[0]).all()
