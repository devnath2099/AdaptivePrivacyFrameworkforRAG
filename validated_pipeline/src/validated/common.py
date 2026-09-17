from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
if PROJECT_ROOT.name != 'validated_pipeline':
    raise ValueError('PROJECT_ROOT must point to the validated_pipeline directory')


def path(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Paths must be relative to validated_pipeline')
    resolved = (PROJECT_ROOT / p).resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(p)


def write_jsonl(p, records):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(p)


def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def metadata(config, sources, count, split, lineage=None):
    try:
        commit = subprocess.check_output(['git', '-c', f'safe.directory={PROJECT_ROOT.parent.as_posix()}',
                                          'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(configuration=config, config_hash=digest(config), seed=int(config['seed']),
                timestamp=datetime.now(timezone.utc).isoformat(), dataset_sources=sources,
                split=split, record_count=count, git_commit=commit, checkpoint_lineage=lineage)


def require_match(actual, expected, description='manifest lineage'):
    if actual != expected:
        raise ValueError(f'Incompatible {description}; use a new run directory. Expected {expected}, got {actual}')


def task_size(spec):
    return len(spec['classes'] if spec['kind'] == 'categorical' else spec['labels'])
