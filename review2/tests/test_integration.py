"""All six stages on test-only fixtures, plus artifact compatibility and frozen test."""
import copy
from pathlib import Path
import pytest
from review2.common import write_jsonl, file_hash, read_json, read_jsonl, load_stage, ROOT
from review2.pipeline import execute


@pytest.mark.parametrize('bounded', [False, True])
def test_six_stage_pipeline(config, tokenizer, tmp_path, bounded):
    tokenizer.save_pretrained(tmp_path / 'tokenizer')
    source = tmp_path / 'gold_fixture.jsonl'
    rows = []
    for split in ('train', 'validation', 'calibration', 'test'):
        for i in range(6):
            text = f'{split} {i} email alice@example.com'
            start = text.index('alice')
            rows.append({'id': f'{split}-{i}', 'text': text, 'split': split,
                         'spans': [{'start': start, 'end': len(text), 'type': 'EMAIL'}]})
    write_jsonl(source, rows)
    config['m1']['sources'] = [{'kind': 'local', 'path': str(source), 'sha256': file_hash(source),
                                'source_id': 'unit_test_fixture_not_research', 'version': '1'}]
    config['m1']['calibration_fraction'] = 0.
    for spec in config['m2']['models']:
        spec.update(tokenizer=str(tmp_path / 'tokenizer'))
    config['m2']['lengths'] = [32]
    config['m2']['losses'] = ['ce']
    config['m3'].update(epsilons=[.001], lambdas=[.5], attacks=['email_obfuscation'])
    config['m4']['passes'] = [2]
    if bounded:
        config['evaluation'] = {'m2_validation': 4, 'm3_validation': 4,
                                'm4_calibration': 4, 'm4_validation': 4, 'test': 4}
    config['m6']['observations'] = str(ROOT / 'configs/controlled_profiles.json')
    output = tmp_path / 'run'
    execute(config, output, evaluate_test=True)
    for stage in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6'):
        assert load_stage(output, stage)['files']
    assert read_json(output / 'final_test/manifest.json')['selection_frozen_before_test']
    assert len(read_jsonl(output / 'm6/decisions.jsonl')) == (4 if bounded else 6)
    assert read_json(output / 'final_test/test_cohort.json')['selected'] == (4 if bounded else 6)
    assert all(not r['enforcement_executed'] for r in read_jsonl(output / 'm6/decisions.jsonl'))
    frozen = read_json(output / 'final_test/frozen_selection.json')
    execute(config, output, stages=(), evaluate_test=True)
    assert read_json(output / 'final_test/frozen_selection.json') == frozen
    assert read_json(output / 'status.json')['completed_stages'] == ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
    with (output / 'm1/train.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError):
        load_stage(output, 'm1')
