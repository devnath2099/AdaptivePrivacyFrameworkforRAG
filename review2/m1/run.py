from pathlib import Path
from review2.common import SPLITS, write_json, write_jsonl, finish_stage, digest
from .loaders import acquire
from .normalization import canonical
from .validation import validate_record, check_isolation
from .splitting import deduplicate, split_records
from .statistics import statistics


def run(config, run_dir):
    directory = Path(run_dir) / 'm1'
    rows, sources, rag, rejected = [], [], [], []
    for spec in config['m1']['sources']:
        loaded, info = acquire(spec, Path(run_dir).parent / 'source_cache')
        sources.append(info)
        for index, raw in enumerate(loaded):
            source = raw.get('source', spec['source_id'])
            source_id = raw.get('_source_record_id', raw.get('id', str(index)))
            if spec.get('role', 'pii') != 'pii':
                rag.append({'record_id': digest([source, source_id])[:24], 'source_id': source,
                            'text': raw.get('text', raw.get('context', '')), 'role': spec['role'], 'pii_gold': False})
                continue
            try:
                row = canonical(raw, source, source_id, raw.get('_official_split', raw.get('split')),
                                config['m1'].get('type_map'), config['m1'].get('wordpiece_alignment', False),
                                config['m1'].get('repair_orphan_i', False))
                validate_record(row)
                rows.append(row)
            except (ValueError, KeyError, TypeError) as error:
                rejected.append({'source_id': source, 'source_record_id': source_id, 'reason': str(error)})
    write_json(directory / 'annotation_integrity.json', {'accepted': len(rows), 'rejected': rejected,
               'orphan_i_repairs': sum(r['metadata']['orphan_i_repairs'] for r in rows),
               'wordpiece_aligned_records': sum(r['metadata']['wordpiece_alignment'] for r in rows)})
    if rejected and config['m1'].get('invalid_annotation', 'error') == 'error':
        raise ValueError('Invalid source annotations; inspect annotation_integrity.json')
    unique, discarded = deduplicate(rows)
    rows = split_records(unique, config['seed'], config['m1']['ratios'], config['m1']['calibration_fraction'])
    check_isolation(rows)
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        if not selected:
            raise ValueError(f'Empty {split}; acquire more data or revise split configuration')
        write_jsonl(directory / f'{split}.jsonl', selected)
    kinds = sorted({s['type'] for r in rows for s in r['canonical_spans']})
    release_labels = {label for info in sources for label in info.get('label_mapping', {}).get('labels', [])}
    kinds = sorted(set(kinds) | {label.split('-', 1)[1] for label in release_labels if label != 'O'})
    write_json(directory / 'labels.json', ['O'] + [f'{prefix}-{kind}' for kind in kinds for prefix in ('B', 'I')])
    write_json(directory / 'statistics.json', statistics(rows))
    write_json(directory / 'split_manifest.json', {'seed': config['seed'], 'records': [
        {k: r[k] for k in ('record_id', 'source_id', 'source_record_id', 'group_id', 'component_id', 'split')} for r in rows],
        'discarded': discarded, 'official_split_policy': 'quarantine conflicting components; carve calibration from train'})
    write_jsonl(directory / 'rag_corpus.jsonl', rag)
    return finish_stage(directory, config, sources=sources, entity_types=kinds,
                        supervision='external annotations; see dataset provenance', test_opened_for_selection=False)


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m1')
