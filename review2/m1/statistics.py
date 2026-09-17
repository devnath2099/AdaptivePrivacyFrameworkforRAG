from collections import Counter
from review2.common import SPLITS


def statistics(rows):
    report = {}
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        report[split] = {'records': len(selected), 'sources': dict(Counter(r['source_id'] for r in selected)),
                         'entities': dict(Counter(s['type'] for r in selected for s in r['canonical_spans'])),
                         'characters': sum(len(r['text']) for r in selected),
                         'negative_records': sum(not r['canonical_spans'] for r in selected)}
    return report
