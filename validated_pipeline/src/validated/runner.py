from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from .common import PROJECT_ROOT, digest, file_hash, metadata, path, read_json, read_jsonl, require_match, write_json, write_jsonl


def run(args):
    cfg = yaml.safe_load(path(args.config).read_text(encoding='utf-8'))
    source = yaml.safe_load(path(cfg['source_manifest']).read_text(encoding='utf-8'))
    if args.profile == 'm1_auxiliary_pii_benchmark':
        from .auxiliary import run_auxiliary
        return run_auxiliary(cfg, args)
    caps = cfg['profiles'][args.profile]
    directory = path(args.output or f'outputs/{args.profile}')
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(config_hash=digest(cfg), source_manifest_hash=digest(source), profile=args.profile, mc_diagnostic=args.mc_diagnostic,
                    code_hash=digest({p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p) for folder in ['src', 'controlled_privacy_benchmark']
                                      for p in (PROJECT_ROOT / folder).rglob('*') if p.suffix in ['.py', '.yaml']}))
    lock = directory / 'run_identity.json'
    if lock.exists():
        require_match(read_json(lock), identity)
    else:
        write_json(lock, identity)
    provenance = metadata(cfg, source, sum(caps.values()), 'natural_all', None)
    provenance['environment'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'datasets', 'snorkel', 'numpy', 'scipy', 'spacy']}
    write_json(directory / 'manifest.json', dict(identity=identity, provenance=provenance))

    def status(state, stage, **extra):
        write_json(directory / 'status.json', dict(state=state, stage=stage, provenance=provenance, **extra))
        print(f'{stage}: {state}', flush=True)

    def artifact_meta(p, records, split, parent=None):
        write_json(p.with_suffix(p.suffix + '.manifest.json'), dict(metadata(cfg, source, len(records), split, parent), sha256=file_hash(p)))

    def done(stage, files, lineage):
        write_json(directory / f'{stage}.done.json', dict(identity=identity, lineage=lineage,
                   artifacts={p.relative_to(directory).as_posix(): file_hash(p) for p in files}, provenance=provenance))

    def reusable(stage, lineage):
        p = directory / f'{stage}.done.json'
        if not p.exists():
            return False
        receipt = read_json(p)
        require_match(receipt['identity'], identity)
        require_match(receipt['lineage'], lineage)
        for name, h in receipt['artifacts'].items():
            if not (directory / name).is_file() or file_hash(directory / name) != h:
                raise ValueError(f'Corrupt stage artifact: {name}')
        return True

    stage = 'data'
    try:
        status('running', stage)
        from .corpus import load_sources, split_exact
        split_file = directory / 'split_manifest.json'
        if not reusable('data', identity['source_manifest_hash']):
            rows, audit = load_sources(source, caps)
            splits = split_exact(rows, caps, int(cfg['seed']))
            split_ids = {s: [r['record_id'] for r in records] for s, records in splits.items()}
            split_id = digest(split_ids)
            write_json(split_file, dict(id=split_id, records=split_ids, counts={s: len(r) for s, r in splits.items()},
                                       source_audit=audit, provenance=provenance))
            files = [split_file]
            for s, records in splits.items():
                p = directory / 'corpus' / f'{s}.jsonl'
                write_jsonl(p, records)
                artifact_meta(p, records, s)
                files += [p, p.with_suffix('.jsonl.manifest.json')]
            done('data', files, identity['source_manifest_hash'])
        split_id = read_json(split_file)['id']
        if args.until == 'data':
            status('partial_requested_stop', stage, split_manifest_id=split_id)
            return

        stage = 'm1_m2'
        status('running', stage)
        if not reusable(stage, split_id):
            from .evidence import extractor, extract_all
            from .labels import synthesize
            # The test JSONL is not opened here or by any default learning/evaluation stage.
            splits = {s: read_jsonl(directory / 'corpus' / f'{s}.jsonl') for s in ['train', 'calibration', 'validation']}
            nlp = extractor(cfg['evidence'])
            for s, rows in splits.items():
                extract_all(rows, nlp)
            results, models = synthesize(splits, cfg)
            import torch
            from .trainer import atomic_save
            files = []
            for s, result in results.items():
                p = directory / 'labels' / f'{s}.jsonl'
                write_jsonl(p, result['records'])
                artifact_meta(p, result['records'], s)
                d = directory / 'labels' / f'{s}_diagnostics.json'
                write_json(d, dict(provenance=metadata(cfg, source, len(splits[s]), s), diagnostics=result['diagnostics'], counts=result['counts']))
                files.extend([p, d, p.with_suffix('.jsonl.manifest.json')])
            p = directory / 'labels' / 'label_models.pt'
            atomic_save({'models': {k: v.state_dict() for k, v in models.items()}, 'provenance': provenance,
                         'fit_split': 'train', 'split_id': split_id}, p)
            files.append(p)
            comparison = directory / 'labels' / 'baseline_comparison.json'
            write_json(comparison, {'status': 'not_comparable', 'reason': 'Baseline combined query/context, different sources, taxonomy and splits; aggregate metrics are not a paired rule revision comparison.', 'provenance': provenance})
            files.append(comparison)
            done(stage, files, split_id)
        if args.until == 'm2':
            status('partial_requested_stop', stage)
            return

        import torch
        from transformers import AutoTokenizer
        from .benchmark import augment
        from .model import QueryDataset, build_model, checkpoint_contract, token_diagnostics
        from .trainer import train, load_checkpoint
        random.seed(int(cfg['seed']))
        np.random.seed(int(cfg['seed']))
        torch.manual_seed(int(cfg['seed']))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg['seed']))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        splits = {s: read_jsonl(directory / 'labels' / f'{s}.jsonl') for s in ['train', 'calibration', 'validation']}
        splits = augment(splits, cfg['augmentation'], cfg['active_tasks'])
        p = directory / 'labels' / 'augmented_train.jsonl'
        write_jsonl(p, splits['train'])
        artifact_meta(p, splits['train'], 'train')
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'], cache_dir=str(path('cache/models')))
        write_json(directory / 'token_diagnostics.json', dict(provenance=provenance, diagnostics=token_diagnostics(splits['train'], tokenizer)))
        datasets = {s: QueryDataset(rows, tokenizer, cfg) for s, rows in splits.items()}
        dev = cfg['training']['device']
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if dev == 'auto' else dev)
        model = build_model(cfg).to(device)
        parent = None
        contracts = {}
        for stage in ['m3', 'm4']:
            status('running', stage)
            contract = checkpoint_contract(cfg, split_id, identity['source_manifest_hash'], parent, stage)
            contract['run_identity'] = identity
            contracts[stage] = contract
            if not reusable(stage, contract):
                parent_hash = train(model, datasets['train'], datasets['validation'], cfg, device, directory / stage,
                                    contract, metadata(cfg, source, len(splits['train']), 'train+validation', parent))
                done(stage, [directory / stage / 'best.pt', directory / stage / 'last.pt', directory / stage / 'history.json'], contract)
            else:
                p = directory / stage / 'best.pt'
                model.load_state_dict(load_checkpoint(p, contract)['model'])
                parent_hash = file_hash(p)
            parent = parent_hash
        stage = 'm5'
        status('running', stage)
        if contracts['m4']['stage'] != 'm4':
            raise ValueError('M5 requires selected revised M4')
        if not reusable(stage, parent):
            from .uncertainty import run_m5
            files = run_m5(model, datasets, splits, cfg, device, directory / stage,
                           metadata(cfg, source, len(splits['validation']), 'calibration_fit+validation_evaluation', parent), args.mc_diagnostic)
            done(stage, files, parent)
        status('complete', stage, test_evaluated=False, selected_m4_sha256=parent)
    except BaseException as exc:
        status('failed', stage, error=f'{type(exc).__name__}: {exc}', test_evaluated=False)
        raise


def main():
    p = argparse.ArgumentParser(description='Independent validated M1-M5; untouched test never evaluated by default')
    p.add_argument('--profile', choices=['overnight_50k', 'scale_25k', 'scale_50k', 'm1_auxiliary_pii_benchmark'], default='overnight_50k')
    p.add_argument('--config', default='configs/pipeline.yaml')
    p.add_argument('--output', help='Project-relative isolated output directory')
    p.add_argument('--until', choices=['data', 'm2', 'm5'], default='m5')
    p.add_argument('--mc-diagnostic', action='store_true')
    run(p.parse_args())
