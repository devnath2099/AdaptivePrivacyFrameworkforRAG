"""Explicit future M4 validation ablation; never launched by overnight profile."""
import argparse
import copy
import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from transformers import AutoTokenizer
from validated.common import path, read_json, read_jsonl, digest, file_hash, metadata, write_json, require_match
from validated.model import build_model, QueryDataset, checkpoint_contract
from validated.benchmark import augment
from validated.trainer import load_checkpoint, train


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,help='Project-relative completed run with selected revised M3')
    args=parser.parse_args()
    root=path(args.run)
    cfg=read_json(root/'manifest.json')['provenance']['configuration']
    source=yaml.safe_load(path(cfg['source_manifest']).read_text())
    split=read_json(root/'split_manifest.json')['id']
    identity=read_json(root/'run_identity.json')
    require_match(identity['source_manifest_hash'],digest(source))
    require_match(identity['config_hash'],digest(cfg))
    c3=checkpoint_contract(cfg,split,digest(source),None,'m3')
    c3['run_identity']=identity
    checkpoint=load_checkpoint(root/'m3'/'best.pt',c3)
    parent=file_hash(root/'m3'/'best.pt')
    rows={s:read_jsonl(root/'labels'/f'{s}.jsonl') for s in ['train','validation']}
    rows=augment(rows,cfg['augmentation'],cfg['active_tasks'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=cfg['model']['revision'],cache_dir=str(path('cache/models')))
    data={s:QueryDataset(rs,tokenizer,cfg) for s,rs in rows.items()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results=[]
    for epsilon,weight in itertools.product(cfg['m4']['epsilon_candidates'],cfg['m4']['lambda_adv_candidates']):
        experiment=copy.deepcopy(cfg)
        experiment['m4'].update(epsilon=float(epsilon),lambda_adv=float(weight))
        torch.manual_seed(int(cfg['seed']))
        model=build_model(experiment).to(device)
        model.load_state_dict(checkpoint['model'])
        contract=checkpoint_contract(experiment,split,digest(source),parent,'m4')
        out=root/'validation_sweep'/f'eps_{epsilon}_weight_{weight}'
        h=train(model,data['train'],data['validation'],experiment,device,out,contract,
                metadata(experiment,source,len(rows['train']),'train+validation',parent))
        results.append(dict(epsilon=epsilon,lambda_adv=weight,checkpoint_hash=h,history=read_json(out/'history.json')))
    write_json(root/'validation_sweep'/'results.json',{'results':results,'test_evaluated':False})
