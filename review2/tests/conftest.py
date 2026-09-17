import copy
import json
from pathlib import Path
import pytest
import torch
from transformers import BertTokenizerFast
from review2.common import ROOT, read_json


@pytest.fixture(autouse=True)
def deterministic_tests():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / 'vocab.txt'
    path.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'email', 'alice', 'bob', '@', 'example', '.', 'com',
                               'call', '123', 'hello', 'public', 'text', 'person', 'at', 'dot', 'id']), encoding='utf-8')
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


@pytest.fixture
def config():
    return copy.deepcopy(read_json(ROOT / 'configs/smoke.json'))


@pytest.fixture
def record():
    return {'record_id': 'one', 'source_id': 'fixture', 'source_record_id': 'one', 'group_id': 'one', 'component_id': 'one',
            'text': 'email alice@example.com', 'canonical_spans': [{'start': 6, 'end': 23, 'type': 'EMAIL'}],
            'source_annotations': [], 'split': 'train', 'metadata': {}}


@pytest.fixture
def reliability():
    return {'record_id': 'one', 'split': 'validation', 'confidence': .9, 'uncertainty': .1,
            'entities': [{'type': 'EMAIL', 'calibrated_confidence': .9, 'uncertainty': .1}]}
