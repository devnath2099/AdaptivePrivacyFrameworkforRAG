"""Small artifact utilities adapted from validated_pipeline/src/validated/common.py."""
import hashlib
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'validation', 'calibration', 'test')


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
