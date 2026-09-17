"""Frozen final test evaluation; selection is written before opening test records."""
from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from review2.common import load_stage, read_json, read_jsonl, write_json, write_jsonl, digest, finish_stage, seed_all
from review2.m2.alignment import encode
from review2.m2.inference import predict, evaluate as evaluate_detector
from review2.m3.evaluation import compare as compare_robustness
from review2.m4.uncertainty import sample_logits
from review2.m4.run import reliability_records
from review2.m4.metrics import reliability_metrics
from review2.m5.aggregation import aggregate
from review2.m6.selection import select


def evaluate(config, run_dir):
    run_dir = Path(run_dir)
    chain, previous = {}, None
    for stage in ('m1', 'm2', 'm3', 'm4', 'm5', 'm6'):
        manifest = load_stage(run_dir, stage, previous)
        previous = digest(manifest)
        chain[stage] = previous
    selection = {stage: read_json(run_dir / stage / 'selected.json') for stage in ('m2', 'm3', 'm4')}
    frozen = {'config_hash': digest(config), 'artifacts': chain, 'selection': selection}
    destination = run_dir / 'final_test'
    frozen_path = destination / 'frozen_selection.json'
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError('Final-test selection is already frozen; cannot silently retune this run')
    write_json(frozen_path, frozen)
    rows = read_jsonl(run_dir / 'm1/test.jsonl')
    labels = read_json(run_dir / 'm1/labels.json')
    counts = read_json(run_dir / 'm1/statistics.json')['train']['entities']
    comparison = {}
    for variant in read_json(run_dir / 'm2/comparison.json'):
        path = run_dir / 'm2' / variant['id'] / 'model'
        model = AutoModelForTokenClassification.from_pretrained(path).to(config['device'])
        tokenizer = AutoTokenizer.from_pretrained(path)
        data = encode(rows, tokenizer, labels, variant['max_length'])
        metrics, pred, _ = evaluate_detector(model, data, tokenizer, labels, config['training']['batch_size'], counts)
        comparison[variant['id']] = metrics
        write_jsonl(destination / f"{variant['id']}_predictions.jsonl", [{'record_id': r['record_id'], 'entities': p, 'split': 'test'} for r, p in zip(rows, pred)])
        if variant['id'] == selection['m2']['id']:
            robustness, _ = compare_robustness(model, rows, tokenizer, labels, variant['max_length'], config['training']['batch_size'], config['m3']['attacks'])
            write_json(destination / 'clean_robustness.json', robustness)
        del model
    path = run_dir / 'm3' / selection['m3']['id'] / 'model'
    model = AutoModelForTokenClassification.from_pretrained(path).to(config['device'])
    tokenizer = AutoTokenizer.from_pretrained(path)
    length = selection['m3']['max_length']
    robustness, pred = compare_robustness(model, rows, tokenizer, labels, length, config['training']['batch_size'], config['m3']['attacks'])
    write_json(destination / 'fgsm_robustness.json', robustness)
    write_jsonl(destination / 'fgsm_attack_predictions.jsonl', pred)
    data = encode(rows, tokenizer, labels, length)
    reliability = selection['m4']
    seed_all(config['seed'])
    if reliability['passes'] > 1:
        logits, _ = sample_logits(model, data, tokenizer, reliability['passes'], config['training']['batch_size'])
    else:
        _, logits, _ = predict(model, data, tokenizer, labels, config['training']['batch_size'])
        logits = [z.unsqueeze(0) for z in logits]
    probabilities = [(z / reliability['temperature']).softmax(-1) for z in logits]
    records = reliability_records(rows, data, probabilities, labels, reliability['method'], reliability['model_hash'])
    write_jsonl(destination / 'reliability.jsonl', records)
    targets = torch.cat([torch.tensor(e['labels']) for e in data])
    write_json(destination / 'calibration_metrics.json', reliability_metrics(torch.cat([p.mean(0) for p in probabilities]), targets))
    profiles, policies = read_json(run_dir / 'm6/profiles.json'), read_json(run_dir / 'm6/policies.json')
    for method in ('max', 'weighted', 'ds'):
        risks = [aggregate(r, config['m5'], method) for r in records]
        write_jsonl(destination / f'{method}_risk.jsonl', risks)
        write_jsonl(destination / f'{method}_decisions.jsonl', [select(r, policies, profiles, config['m6']) for r in risks])
    write_json(destination / 'detector_comparison.json', comparison)
    finish_stage(destination, config, previous, selection_frozen_before_test=True)
