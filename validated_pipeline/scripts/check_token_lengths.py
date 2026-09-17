"""Training-only tokenizer diagnostics before committing GPU time."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from transformers import AutoTokenizer
from validated.common import path,read_json,read_jsonl,write_json,metadata
from validated.model import token_diagnostics

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    a=p.parse_args()
    root=path(a.run)
    manifest=read_json(root/'manifest.json')
    cfg=manifest['provenance']['configuration']
    rows=read_jsonl(root/'corpus'/'train.jsonl')
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=cfg['model']['revision'],cache_dir=str(path('cache/models')))
    diagnostics=token_diagnostics(rows,tok)
    write_json(root/'pretraining_token_diagnostics.json',dict(provenance=metadata(cfg,manifest['provenance']['dataset_sources'],len(rows),'train'),diagnostics=diagnostics))
    print(diagnostics)
