"""Fail closed on invalid, overlapping, or contradictory annotations."""
from review2.common import SPLITS


def validate_record(record):
    if not isinstance(record['text'], str) or not record['text'].strip():
        raise ValueError('Empty or non-string text')
    previous = 0
    for span in record['canonical_spans']:
        start, end = span['start'], span['end']
        if not 0 <= start < end <= len(record['text']) or start < previous:
            raise ValueError('Invalid/overlapping span; flat BIO cannot represent overlap')
        if not isinstance(span['type'], str) or not span['type']:
            raise ValueError('Missing entity type')
        if 'value' in span and record['text'][start:end] != span['value']:
            raise ValueError('Annotation value does not match text')
        previous = end


def check_isolation(rows):
    seen = {}
    for row in rows:
        if row['split'] not in SPLITS:
            raise ValueError('Unknown split')
        for key in [('id', row['record_id']), ('text', ' '.join(row['text'].casefold().split())),
                    ('group', row['source_id'], row['group_id']), ('component', row['component_id'])]:
            if key in seen and seen[key] != row['split']:
                raise ValueError('Cross-split leakage')
            seen[key] = row['split']
