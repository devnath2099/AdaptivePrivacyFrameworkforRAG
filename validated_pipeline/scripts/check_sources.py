"""Load all pinned sources and solve exact split quotas without any learned stages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from validated.common import path, metadata, write_json, digest
from validated.corpus import load_sources, split_exact

if __name__ == '__main__':
    cfg = yaml.safe_load(path('configs/pipeline.yaml').read_text())
    source = yaml.safe_load(path(cfg['source_manifest']).read_text())
    rows, audit = load_sources(source, cfg['profiles']['overnight_50k'])
    splits = split_exact(rows, cfg['profiles']['overnight_50k'], cfg['seed'])
    result = dict(provenance=metadata(cfg, source, sum(map(len,splits.values())), 'natural_all'),
                  source_audit=audit, counts={s:len(r) for s,r in splits.items()},
                  split_manifest_id=digest({s:[r['record_id'] for r in rs] for s,rs in splits.items()}),
                  result='actual source load and group-safe split passed; no training or test evaluation')
    write_json(path('outputs/source_check/result.json'),result)
    print(result['counts'])
    print(result['result'])
