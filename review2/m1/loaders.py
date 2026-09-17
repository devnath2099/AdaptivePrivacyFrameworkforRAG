"""Pinned Hugging Face JSONL or checksum-verified local sources; no synthetic fallback."""
import json
import random
import urllib.request
from pathlib import Path
from review2.common import file_hash, write_json, write_jsonl, read_json, read_jsonl, digest


def acquire(spec, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if spec['kind'] == 'validated_cache':
        if spec.get('role', 'pii') == 'pii':
            raise ValueError('Legacy QA cache cannot supply PII gold')
        saved = read_json(spec['path'])
        if saved['content_hash'] != digest(saved['records']):
            raise ValueError('Legacy corpus cache content hash mismatch')
        if any(r.get('source_mode') != 'natural' for r in saved['records']):
            raise ValueError('Expected verified natural application corpus')
        selected = saved['records'][:spec.get('limit', len(saved['records']))]
        rows = [{'id': r['source_record_id'], 'source': r['source_dataset'],
                 'text': r['context_text'], 'query': r['query_text']} for r in selected]
        return rows, {'spec': spec, 'upstream_spec': saved['spec'], 'sha256': file_hash(spec['path']),
                      'records': len(rows), 'role': spec['role'], 'pii_supervision': False}
    if spec['kind'] == 'local':
        path = Path(spec['path'])
        if file_hash(path) != spec['sha256']:
            raise ValueError('Local dataset checksum mismatch')
        return read_jsonl(path), {'sha256': spec['sha256'], 'version': spec['version']}
    revision = spec['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Dataset revision must be an immutable commit hash')
    key = digest(spec)
    path, manifest = cache / (key + '.jsonl'), cache / (key + '.json')
    if path.exists() and manifest.exists():
        info = read_json(manifest)
        if info['sha256'] != file_hash(path):
            raise ValueError('Corrupt dataset cache')
        return read_jsonl(path), info
    base = f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{revision}/"
    rows = []
    scanned = {}
    sampling = spec.get('sampling', 'prefix')
    if sampling not in ('prefix', 'reservoir'):
        raise ValueError('Unknown acquisition sampling method')
    if 'limits_per_split' in spec and set(spec['limits_per_split']) != set(spec['files']):
        raise ValueError('Per-split acquisition limits must name every official split')
    for split, filename in spec['files'].items():
        count = 0
        pool = []
        limit = spec.get('limits_per_split', {}).get(split, spec.get('limit_per_split'))
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError('Acquisition limits must be positive integers or null')
        rng = random.Random(f"{spec.get('seed', 42)}:{split}")
        with urllib.request.urlopen(base + filename, timeout=120) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row['_official_split'] = split
                row['_source_record_id'] = f'{split}:{count}'
                count += 1
                if limit is None or len(pool) < limit:
                    pool.append(row)
                elif sampling == 'reservoir':
                    position = rng.randrange(count)
                    if position < limit:
                        pool[position] = row
                if sampling == 'prefix' and limit and count >= limit:
                    break
        scanned[split] = count
        rows.extend(sorted(pool, key=lambda row: int(row['_source_record_id'].rsplit(':', 1)[1])))
    if not rows:
        raise ValueError('No source records acquired')
    write_jsonl(path, rows)
    info = {'spec': spec, 'sha256': file_hash(path), 'records': len(rows),
            'sampling': sampling if spec.get('limit_per_split') or spec.get('limits_per_split') else 'complete release',
            'records_scanned_per_split': scanned}
    if spec.get('label_file'):
        with urllib.request.urlopen(base + spec['label_file'], timeout=120) as handle:
            info['label_mapping'] = json.load(handle)
    write_json(manifest, info)
    return rows, info
