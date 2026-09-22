"""Offline mechanics checks with the actual cached tokenizer; no research metrics."""
import numpy as np
import pytest
from transformers import AutoTokenizer
from review2_final.neural import encode, MODEL
from review2_final.m4 import sample_predictive


def test_long_document_keeps_all_native_tokens_and_bio_boundaries():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, add_prefix_space=True, local_files_only=True)
    except OSError:
        pytest.skip("Public DeBERTa tokenizer not cached; run smoke after downloading it")
    words = ["student"]*700
    labels = ["O"]*700
    labels[61:64] = ["B-NAME", "I-NAME", "I-NAME"]
    windows = encode([{"record_id": "fixture", "tokens": words, "labels": labels}], tokenizer,
                     ["O", "B-NAME", "I-NAME"], max_length=64)
    assert len(windows) > 1
    words_seen = [w for row in windows for w in row["words"] if w >= 0]
    assert words_seen == list(range(700))
    assert all(len(row["input_ids"]) <= 64 for row in windows)
    gold = [tag for row in windows for tag in row["labels"] if tag != -100]
    assert gold[61:64] == [1,2,2]


def test_mc_averages_probabilities_not_logits(monkeypatch):
    import review2_final.m4 as m4
    samples = iter([np.array([[4.,0.]]), np.array([[0.,1.]])])
    monkeypatch.setattr(m4, "predict_logits", lambda *a, **k: [next(samples)])
    result = list(sample_predictive(None, None, [], ["O", "B-X"], "cpu", [2], 1))
    p = result[0][1]
    expected = (m4.softmax(np.array([[4.,0.]]))+m4.softmax(np.array([[0.,1.]])))/2
    assert np.allclose(p, expected)
    assert result[0][2]["mutual_information"][0] > 0


def test_oversized_native_token_retains_label_head_tail_and_audit():
    import copy
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, add_prefix_space=True, local_files_only=True)
    except OSError:
        pytest.skip("Public DeBERTa tokenizer not cached")
    row = {"record_id": "oversized", "tokens": ["before", "ab/"*1000, "after"],
           "labels": ["O", "B-X", "O"]}
    original = copy.deepcopy(row)
    windows = encode([row], tokenizer, ["O", "B-X", "I-X"], 64)
    assert row == original
    assert [w for r in windows for w in r["words"] if w >= 0] == [0,1,2]
    assert [t for r in windows for t in r["labels"] if t != -100] == [0,1,0]
    assert all(len(r["input_ids"]) <= 64 for r in windows)
    tokens = tokenizer(row["tokens"], is_split_into_words=True, add_special_tokens=False)
    pieces = [t for t, w in zip(tokens["input_ids"], tokens.word_ids()) if w == 1]
    long_window, = [r for r in windows if r["token_truncations"]]
    assert long_window["input_ids"][1:-1] == pieces[:31]+pieces[-31:]
    assert long_window["token_truncations"][0]["discarded_subwords"] == len(pieces)-62
