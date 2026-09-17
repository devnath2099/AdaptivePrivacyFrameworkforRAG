"""Reuse verified existing medical/control caches as M1 application artifacts only."""
from pathlib import Path
from review2.common import ROOT, write_json, write_jsonl, finish_stage
from review2.m1.loaders import acquire


def main():
    destination = ROOT / 'outputs/application_corpora'
    sources, records = [], []
    for source, role in [('healthcaremagic', 'medical_application'), ('natural_questions', 'public_control')]:
        files = sorted((ROOT.parent / 'validated_pipeline/cache/sources').glob(source + '-*.json'))
        if len(files) != 1:
            raise ValueError(f'Expected one verified cache for {source}; found {len(files)}')
        spec = {'kind': 'validated_cache', 'path': str(files[0]), 'source_id': source, 'role': role, 'limit': 100}
        rows, info = acquire(spec, destination / 'cache')
        sources.append(info)
        records.extend({**r, 'role': role, 'pii_gold': False} for r in rows)
    write_jsonl(destination / 'rag_corpus.jsonl', records)
    write_json(destination / 'sources.json', sources)
    finish_stage(destination, {'seed': 42, 'purpose': 'M1 application corpus reuse', 'limit_per_source': 100},
                 role='application_only', source_count=len(sources), records=len(records))
    print(f'Saved {len(records)} application/control records; no PII gold labels assigned')


if __name__ == '__main__':
    main()
