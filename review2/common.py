"""Small artifact utilities adapted from validated_pipeline/src/validated/common.py."""
import hashlib
import json
import random
from collections import defaultdict, Counter
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'validation', 'calibration', 'test')


def record_subset(rows, limit, seed):
    """Fixed hash-ordered whole-group sample, selected without examining annotations."""
    if limit is None:
        return rows
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Record limit must be a positive integer')
    if len(rows) <= limit:
        return rows
    groups = defaultdict(list)
    for row in rows:
        key = row.get('component_id', row['record_id'])
        groups[key].append(row)
    selected = []
    for key in sorted(groups, key=lambda k: digest([seed, k])):
        group = sorted(groups[key], key=lambda row: row['record_id'])
        if len(selected) + len(group) <= limit:
            selected.extend(group)
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError('No complete group fits the requested record limit')
    return selected


def evaluation_cohort(rows, limit, seed, output):
    selected = record_subset(rows, limit, seed)
    write_json(output, {'available': len(rows), 'requested_limit': limit, 'selected': len(selected),
                       'seed': seed, 'selection': 'annotation-blind hash ordering of intact groups',
                       'sources': dict(Counter(r['source_id'] for r in selected)),
                       'record_ids': [r['record_id'] for r in selected]})
    return selected


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def provenance(config):
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.parent.as_posix()}',
                                            'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {'config': config, 'config_hash': digest(config), 'code_revision': revision,
            'code_hash': digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(ROOT.rglob('*.py'))}),
            'seed': config['seed'], 'schema_version': 1}


def load_stage(run, stage, parent=None):
    directory = Path(run) / stage
    manifest = read_json(directory / 'manifest.json')
    if parent is not None and manifest['parent_hash'] != parent:
        raise ValueError(f'{stage}: incompatible upstream artifact')
    for name, expected in manifest['files'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'{stage}: artifact changed: {name}')
    return manifest


def finish_stage(directory, config, parent_hash=None, **details):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'manifest.json' and not p.name.endswith('.tmp')}
    manifest = {**provenance(config), 'parent_hash': parent_hash, 'files': files, **details}
    write_json(directory / 'manifest.json', manifest)
    return digest(manifest)
