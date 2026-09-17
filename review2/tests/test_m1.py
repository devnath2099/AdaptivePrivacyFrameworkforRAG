import copy
import pytest
from review2.m1.normalization import canonical, token_offsets, bio_spans
from review2.m1.validation import validate_record, check_isolation
from review2.m1.splitting import deduplicate, split_records


def test_bio_source_alignment():
    row = canonical({'tokens': ['Alice', 'Jones', 'here'], 'labels': ['B-NAME', 'I-NAME', 'O'], 'text': 'Alice  Jones here'}, 'external', '1', 'train')
    assert row['canonical_spans'] == [{'start': 0, 'end': 12, 'type': 'NAME'}]
    validate_record(row)


def test_no_guessed_alignment():
    with pytest.raises(ValueError):
        token_offsets('Alice SECRET Bob', ['Alice', 'Bob'])
    with pytest.raises(ValueError):
        bio_spans(['I-NAME'], [(0, 5)])


def test_explicit_wordpiece_and_bio_repair():
    assert token_offsets('Mayoress, here', ['Mayor', '##ess', ',', 'here'], True) == [(0, 5), (5, 8), (8, 9), (10, 14)]
    assert bio_spans(['I-NAME'], [(0, 5)], True) == [{'start': 0, 'end': 5, 'type': 'NAME'}]


def test_overlap_and_value_validation(record):
    row = copy.deepcopy(record)
    row['canonical_spans'][0]['value'] = 'wrong'
    with pytest.raises(ValueError):
        validate_record(row)
    row = copy.deepcopy(record)
    row['canonical_spans'].append({'start': 8, 'end': 10, 'type': 'NAME'})
    with pytest.raises(ValueError):
        validate_record(row)


def test_duplicate_and_group_transitivity(record):
    a = copy.deepcopy(record)
    b = dict(a, record_id='two', group_id='two')
    c = dict(a, record_id='three', group_id='two', text='different text', canonical_spans=[])
    rows, discarded = deduplicate([a, b, c])
    assert len(rows) == 2 and len(discarded) == 1
    assert len({r['component_id'] for r in rows}) == 1


def test_official_conflict_quarantined(record):
    other = dict(record, record_id='test_copy', split='test')
    rows, discarded = deduplicate([record, other])
    assert rows == [] and len(discarded) == 2


def test_conflicting_gold_rejected(record):
    other = dict(record, record_id='two', canonical_spans=[])
    with pytest.raises(ValueError):
        deduplicate([record, other])


def test_split_reproducibility_and_leakage(record):
    rows = [dict(record, record_id=str(i), group_id=str(i), component_id=str(i), text=f'text {i}', split=None, canonical_spans=[]) for i in range(100)]
    ratios = {'train': .7, 'validation': .1, 'calibration': .1, 'test': .1}
    first = split_records(rows, 42, ratios)
    assert first == split_records(list(reversed(rows)), 42, ratios)
    assert {r['split'] for r in first} == set(ratios)
    check_isolation(first)
    with pytest.raises(ValueError):
        check_isolation(first + [dict(first[0], split='test' if first[0]['split'] != 'test' else 'train')])
