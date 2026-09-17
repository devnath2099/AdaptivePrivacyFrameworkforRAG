"""Standalone deterministic benchmark generation (never run heldout by default)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from validated.benchmark import generate
from validated.common import path, write_jsonl, write_json, metadata, file_hash

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--partition', choices=['train', 'heldout'], required=True)
    p.add_argument('--count', type=int, default=160)
    a = p.parse_args()
    rows = generate(a.partition, a.count)
    output = path(f'outputs/controlled_benchmark/{a.partition}.jsonl')
    write_jsonl(output, rows)
    write_json(output.with_suffix('.manifest.json'), dict(metadata({'seed':42, 'requested_count':a.count},
               'controlled_ontology_v1', len(rows), a.partition), sha256=file_hash(output)))
