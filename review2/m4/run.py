from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from review2.common import read_json, read_jsonl, write_json, write_jsonl, load_stage, digest, finish_stage, file_hash, seed_all, evaluation_cohort
from review2.m2.alignment import encode, decode
from review2.m2.inference import predict
from .calibration import fit_temperature
from .uncertainty import sample_logits, uncertainty
from .metrics import reliability_metrics, plot_metrics


def reliability_records(rows, examples, samples, labels, method, model_hash):
    output = []
    for row, example, p in zip(rows, examples, samples):
        info = uncertainty(p)
        mean = info['mean']
        entities = decode(mean.argmax(-1), example['offsets'], labels, mean)
        for entity in entities:
            entity['calibrated_confidence'] = entity['confidence']
            entity['uncertainty'] = float(info['entropy'][entity['token_indices']].max())
        valid = torch.tensor([a != b for a, b in example['offsets']])
        output.append({'record_id': row['record_id'], 'source_id': row['source_id'], 'split': row['split'],
                       'entities': entities, 'uncertainty': float(info['entropy'][valid].max()) if valid.any() else 1.,
                       'confidence': min((s['confidence'] for s in entities), default=float(mean[valid, 0].min()) if valid.any() else 0.),
                       'method': method, 'model_hash': model_hash,
                       'uncertainty_definition': 'maximum normalized predictive token entropy; conservative configured convention'})
    return output


def run(config, run_dir):
    run_dir = Path(run_dir)
    m1 = load_stage(run_dir, 'm1')
    m2 = load_stage(run_dir, 'm2', digest(m1))
    parent = load_stage(run_dir, 'm3', digest(m2))
    selected = read_json(run_dir / 'm3/selected.json')
    path = run_dir / 'm3' / selected['id'] / 'model'
    model = AutoModelForTokenClassification.from_pretrained(path).to(config['device'])
    tokenizer = AutoTokenizer.from_pretrained(path)
    labels = read_json(run_dir / 'm1/labels.json')
    rows = {s: read_jsonl(run_dir / f'm1/{s}.jsonl') for s in ('calibration', 'validation')}
    for split in rows:
        rows[split] = evaluation_cohort(rows[split], config.get('evaluation', {}).get('m4_' + split),
                                       config['seed'], run_dir / 'm4' / (split + '_cohort.json'))
    data = {s: encode(r, tokenizer, labels, selected['max_length']) for s, r in rows.items()}
    directory = run_dir / 'm4'
    directory.mkdir(parents=True, exist_ok=True)
    logits, times = {}, {}
    for split in rows:
        _, logits[split], times[split] = predict(model, data[split], tokenizer, labels, config['training']['batch_size'])
    targets = {s: torch.cat([torch.tensor(e['labels']) for e in data[s]]) for s in data}
    calibration = fit_temperature(torch.cat(logits['calibration']), targets['calibration'], 'calibration', config['m4']['iterations'])
    temperature = calibration['temperature']
    write_json(directory / 'temperature.json', calibration)
    torch.save({'logits': logits['calibration'], 'targets': targets['calibration'], 'split': 'calibration'}, directory / 'calibration_logits.pt')
    candidates = {'raw': [z.softmax(-1).unsqueeze(0) for z in logits['validation']],
                  'temperature': [(z / temperature).softmax(-1).unsqueeze(0) for z in logits['validation']]}
    latency = {k: times['validation'] for k in candidates}
    seed_all(config['seed'])
    samples, mc_times = sample_logits(model, data['validation'], tokenizer, max(config['m4']['passes']), config['training']['batch_size'])
    for count in config['m4']['passes']:
        name = f'temperature_mc{count}'
        candidates[name] = [(z[:count] / temperature).softmax(-1) for z in samples]
        latency[name] = mc_times[count - 1]
    torch.save({'samples': samples, 'targets': targets['validation'], 'split': 'validation'}, directory / 'validation_mc_logits.pt')
    comparisons = {}
    reference = torch.cat([p.mean(0) for p in candidates[f"temperature_mc{max(config['m4']['passes'])}"]])
    for name, probabilities in candidates.items():
        stats = [uncertainty(p) for p in probabilities]
        p = torch.cat([s['mean'] for s in stats])
        entropy = torch.cat([s['entropy'] for s in stats])
        metrics = reliability_metrics(p, targets['validation'], config['m4']['bins'], entropy)
        pii_targets = targets['validation'].clone()
        pii_targets[pii_targets == 0] = -100
        pii = reliability_metrics(p, pii_targets, config['m4']['bins'], entropy) if (pii_targets >= 0).any() else None
        comparisons[name] = {'metrics': metrics, 'pii_only_metrics': pii, 'seconds': latency[name],
                             'mean_absolute_probability_delta_to_max_mc': float((p - reference).abs().mean()),
                             'prediction_agreement_to_max_mc': float((p.argmax(-1) == reference.argmax(-1)).float().mean())}
    # No cost/accuracy weighted sum: minimum validation AURC under an explicit latency budget.
    feasible = [k for k in comparisons if latency[k] <= latency['raw'] * config['m4']['max_latency_ratio']]
    chosen = min(feasible, key=lambda k: (comparisons[k]['metrics']['aurc'], latency[k]))
    metadata = {'method': chosen, 'temperature': 1. if chosen == 'raw' else temperature,
                'passes': int(chosen.split('mc')[-1]) if '_mc' in chosen else 1,
                'criterion': 'minimum validation token AURC within configured raw-latency ratio',
                'model_hash': file_hash(run_dir / 'm3' / selected['id'] / 'best.pt')}
    write_json(directory / 'comparison.json', comparisons)
    write_json(directory / 'selected.json', metadata)
    for name, probabilities in candidates.items():
        write_jsonl(directory / f'{name}_reliability.jsonl', reliability_records(rows['validation'], data['validation'], probabilities,
                                                                               labels, name, metadata['model_hash']))
    write_jsonl(directory / 'reliability.jsonl', reliability_records(rows['validation'], data['validation'], candidates[chosen],
                                                                   labels, chosen, metadata['model_hash']))
    plot_metrics(comparisons, directory)
    return finish_stage(directory, config, digest(parent))


if __name__ == '__main__':
    from review2.pipeline import module_cli
    module_cli('m4')
