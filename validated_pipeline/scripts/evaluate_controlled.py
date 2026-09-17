"""Explicit held-out ontology compliance; never natural-user or clinical gold accuracy."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from transformers import AutoTokenizer
from validated.common import path,read_json,write_json,file_hash,metadata
from validated.model import build_model,QueryDataset,checkpoint_contract
from validated.trainer import load_checkpoint,evaluate
from validated.benchmark import generate

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--count',type=int,default=800)
    a=p.parse_args()
    root=path(a.run)
    manifest=read_json(root/'manifest.json')
    cfg=manifest['provenance']['configuration']
    c=checkpoint_contract(cfg,read_json(root/'split_manifest.json')['id'],manifest['identity']['source_manifest_hash'],file_hash(root/'m3'/'best.pt'),'m4')
    c['run_identity']=manifest['identity']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(cfg).to(device)
    model.load_state_dict(load_checkpoint(root/'m4'/'best.pt',c)['model'])
    rows=generate('heldout',a.count,cfg['active_tasks'])
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=cfg['model']['revision'],cache_dir=str(path('cache/models')))
    result=evaluate(model,QueryDataset(rows,tok,cfg),cfg,device)
    for metric in result['tasks'].values():
        metric['ontology_label_agreement']=metric.pop('weak_label_agreement')
    result['interpretation']='controlled ontology-compliance evaluation; not natural-user generalization or gold clinical evaluation'
    write_json(root/'controlled_evaluation.json',dict(result=result,provenance=metadata(cfg,manifest['provenance']['dataset_sources'],len(rows),'controlled_heldout',file_hash(root/'m4'/'best.pt'))))
