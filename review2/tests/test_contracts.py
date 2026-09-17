import copy
import pytest
from review2.common import write_json, read_json, digest, finish_stage, load_stage
from review2.m1.loaders import acquire
from review2.pipeline import execute


def test_artifact_parent_and_file_hash(tmp_path):
    directory = tmp_path / 'm1'
    write_json(directory / 'data.json', {'x': 1})
    finish_stage(directory, {'seed': 42}, 'upstream')
    assert load_stage(tmp_path, 'm1', 'upstream')['parent_hash'] == 'upstream'
    with pytest.raises(ValueError):
        load_stage(tmp_path, 'm1', 'wrong')
    write_json(directory / 'data.json', {'x': 2})
    with pytest.raises(ValueError):
        load_stage(tmp_path, 'm1')


def test_application_cache_cannot_be_pii_gold(tmp_path):
    rows = [{'source_record_id': 'one', 'source_dataset': 'medical', 'source_mode': 'natural',
             'context_text': 'application context', 'query_text': 'question'}]
    path = tmp_path / 'legacy.json'
    write_json(path, {'records': rows, 'content_hash': digest(rows), 'spec': {'revision': 'version'}})
    spec = {'kind': 'validated_cache', 'path': str(path), 'role': 'medical_application'}
    loaded, info = acquire(spec, tmp_path / 'cache')
    assert loaded[0]['text'] == 'application context' and not info['pii_supervision']
    with pytest.raises(ValueError):
        acquire({**spec, 'role': 'pii'}, tmp_path / 'cache')


def test_frozen_run_cannot_be_retuned(tmp_path, config):
    write_json(tmp_path / 'final_test/frozen_selection.json', {})
    with pytest.raises(ValueError, match='frozen'):
        execute(config, tmp_path)
