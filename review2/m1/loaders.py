"""Pinned Hugging Face JSONL or checksum-verified local sources; no synthetic fallback."""
import json
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
    for split, filename in spec['files'].items():
        count = 0
        with urllib.request.urlopen(base + filename, timeout=120) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row['_official_split'] = split
                row['_source_record_id'] = f'{split}:{count}'
                rows.append(row)
                count += 1
                if spec.get('limit_per_split') and count >= spec['limit_per_split']:
                    break
    if not rows:
        raise ValueError('No source records acquired')
    write_jsonl(path, rows)
    info = {'spec': spec, 'sha256': file_hash(path), 'records': len(rows),
            'sampling': 'official split prefix; feasibility run only' if spec.get('limit_per_split') else 'complete release'}
    if spec.get('label_file'):
        with urllib.request.urlopen(base + spec['label_file'], timeout=120) as handle:
            info['label_mapping'] = json.load(handle)
    write_json(manifest, info)
    return rows, info
