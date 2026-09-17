"""Lightweight MC ranking/runtime diagnostic from selected M4; calibration subset only."""
import argparse
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from validated.common import path,read_json,read_jsonl,write_json,file_hash,metadata
from validated.model import build_model,QueryDataset,checkpoint_contract
from validated.trainer import load_checkpoint
from validated.uncertainty import collect
from validated.calibration import uncertainty

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--count',type=int,default=128)
    a=p.parse_args()
    if a.count < 2:
        raise ValueError('Ranking requires at least two calibration records')
    root=path(a.run)
    manifest=read_json(root/'manifest.json')
    cfg=manifest['provenance']['configuration']
    c=checkpoint_contract(cfg,read_json(root/'split_manifest.json')['id'],manifest['identity']['source_manifest_hash'],file_hash(root/'m3'/'best.pt'),'m4')
    c['run_identity']=manifest['identity']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(cfg).to(device)
    model.load_state_dict(load_checkpoint(root/'m4'/'best.pt',c)['model'])
    rows=read_jsonl(root/'labels'/'calibration.jsonl')[:a.count]
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=cfg['model']['revision'],cache_dir=str(path('cache/models')))
    data=QueryDataset(rows,tok,cfg)
    vectors,reports=[],[]
    for passes in cfg['m5']['diagnostic_passes']:
        torch.manual_seed(int(cfg['seed'])+501)
        start=time.perf_counter()
        values,_,_=collect(model,data,cfg,device,passes)
        vector=sum(uncertainty(values[k],s['kind'])['normalized_entropy'] for k,s in model.tasks.items())/len(model.tasks)
        vectors.append(vector.numpy())
        reports.append(dict(T_mc=passes,runtime_seconds=time.perf_counter()-start))
    for i,r in enumerate(reports):
        rho=float(spearmanr(vectors[i],vectors[-1]).statistic)
        r['spearman_vs_30']=rho if math.isfinite(rho) else None
    write_json(root/'mc_subset_diagnostic.json',dict(provenance=metadata(cfg,manifest['provenance']['dataset_sources'],len(rows),'calibration_subset',file_hash(root/'m4'/'best.pt')),
               temperatures=1.0,runs=reports,note='Uncalibrated ranking stability only; no parameter or threshold selected.'))
