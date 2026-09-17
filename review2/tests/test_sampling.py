import io
import json
import pytest
from review2.common import record_subset, evaluation_cohort, read_json, write_jsonl, file_hash
from review2.m1.loaders import acquire
from review2.m1.run import run


def test_group_sampling_is_order_independent_and_complete(tmp_path):
    rows = [{'record_id': str(i), 'component_id': str(i // 2), 'source_id': 'fixture'} for i in range(20)]
    chosen = evaluation_cohort(rows, 5, 42, tmp_path / 'cohort.json')
    assert len(chosen) == 4
    assert chosen == record_subset(list(reversed(rows)), 5, 42)
    for row in chosen:
        assert sum(r['component_id'] == row['component_id'] for r in chosen) == 2
    assert read_json(tmp_path / 'cohort.json')['selected'] == 4
    with pytest.raises(ValueError):
        record_subset(rows, 1, 42)


@pytest.mark.parametrize('limit', [0, -1, True, 2.5])
def test_invalid_caps(limit):
    with pytest.raises(ValueError):
        record_subset([], limit, 42)


def test_reservoir_scans_release_and_reproduces(monkeypatch, tmp_path):
    data = ''.join(json.dumps({'id': i}) + '\n' for i in range(100)).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(data))
    spec = {'kind': 'huggingface_jsonl', 'revision': 'a'*40, 'dataset_id': 'fixture',
            'files': {'train': 'train.jsonl'}, 'sampling': 'reservoir', 'limits_per_split': {'train': 10}, 'seed': 42}
    rows, info = acquire(spec, tmp_path / 'first')
    assert len(rows) == 10 and info['records_scanned_per_split'] == {'train': 100}
    assert max(r['id'] for r in rows) > 9
    assert acquire(spec, tmp_path / 'second')[0] == rows
    assert acquire(spec, tmp_path / 'first')[0] == rows
    spec['sampling'] = 'prefix'
    assert [r['id'] for r in acquire(spec, tmp_path / 'prefix')[0]] == list(range(10))


def test_exact_valid_targets_and_deficit(config, tmp_path):
    source = tmp_path / 'rows.jsonl'
    rows = [{'id': f'{split}-{i}', 'split': split, 'text': f'{split} example {i}', 'spans': []}
            for split in ('train', 'validation', 'calibration', 'test') for i in range(8)]
    write_jsonl(source, rows)
    config['m1'].update(sources=[{'kind': 'local', 'path': str(source), 'sha256': file_hash(source),
                                'version': '1', 'source_id': 'fixture'}], calibration_fraction=0,
                        official_split_targets={s: 4 for s in ('train', 'validation', 'calibration', 'test')})
    run(config, tmp_path / 'success')
    assert read_json(tmp_path / 'success/m1/sampling_manifest.json')['selected'] == 16
    config['m1']['official_split_targets']['train'] = 10
    with pytest.raises(ValueError, match='cannot meet valid-record target'):
        run(config, tmp_path / 'deficit')
    assert read_json(tmp_path / 'deficit/m1/target_deficit.json')['available'] == 8
