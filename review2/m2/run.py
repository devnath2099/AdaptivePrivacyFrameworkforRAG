from pathlib import Path
import torch
from review2.common import read_json, read_jsonl, write_json, write_jsonl, load_stage, finish_stage, digest, seed_all
from .model import build, get_tokenizer
from .alignment import encode
from .training import train
from .inference import evaluate
from .metrics import entity_metrics
from .coverage import coverage
from .baselines import regex_predict


def run(config, run_dir):
    run_dir = Path(run_dir)
    parent = load_stage(run_dir, 'm1')
    labels = read_json(run_dir / 'm1/labels.json')
    rows = {s: read_jsonl(run_dir / f'm1/{s}.jsonl') for s in ('train', 'validation')}
    counts = read_json(run_dir / 'm1/statistics.json')['train']['entities']
    directory = run_dir / 'm2'
    results = []
    for spec in config['m2']['models']:
        tokenizer = get_tokenizer(spec)
        for loss in config['m2']['losses']:
            seed_all(config['seed'])
            name = spec['id'] + '_' + loss
            model = build(spec, labels, tokenizer).to(config['device'])
            report = coverage(rows['train'], tokenizer, config['m2']['lengths'], model)
            # Smallest candidate with configured gold-span coverage; no test access.
            acceptable = [r['length'] for r in report if r['affected_gold_span_fraction'] <= config['m2']['max_span_truncation']]
            length = min(acceptable) if acceptable else max(config['m2']['lengths'])
            examples = {s: encode(r, tokenizer, labels, length) for s, r in rows.items()}
            cfg = {**config['training'], 'loss': loss, 'seed': config['seed']}
            contract = {'data': digest(parent), 'model': spec, 'labels': labels, 'training': cfg, 'length': length, 'stage': 'm2'}
            info = train(model, tokenizer, labels, examples['train'], examples['validation'], cfg, directory / name, contract)
            metrics, predictions, _ = evaluate(model, examples['validation'], tokenizer, labels, cfg['batch_size'], counts)
            write_json(directory / name / 'coverage.json', report)
            write_json(directory / name / 'validation_metrics.json', metrics)
            write_jsonl(directory / name / 'validation_predictions.jsonl', [
                {'record_id': r['record_id'], 'split': 'validation', 'entities': p} for r, p in zip(rows['validation'], predictions)])
            results.append({'id': name, 'family': spec['family'], 'tiny_random_test_model': spec.get('tiny', False),
                            'max_length': length, 'loss': loss, **metrics, **info})
            del model
    primary = max((r for r in results if r['family'] == 'deberta'), key=lambda r: (r['f1'], -r['milliseconds_per_record']))
    regex = regex_predict(rows['validation'], labels)
    write_json(directory / 'regex_baseline.json', {**entity_metrics(rows['validation'], regex, counts),
               'record_coverage': sum(bool(p) for p in regex) / len(regex), 'role': 'auxiliary baseline, never gold'})
    write_json(directory / 'comparison.json', results)
    write_json(directory / 'selected.json', {'id': primary['id'], 'max_length': primary['max_length'],
               'reason': 'Best validation DeBERTa loss variant; architecture remains DeBERTa pending manual review',
               'best_overall_validation': max(results, key=lambda r: r['f1'])['id']})
    return finish_stage(directory, config, digest(parent))


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m2')
