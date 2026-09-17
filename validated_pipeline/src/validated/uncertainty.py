import time
import torch
from torch.utils.data import DataLoader, Subset
from .calibration import enable_mc_dropout, fit_temperature, probability, mc_nll, ece, uncertainty
from .common import write_json, write_jsonl
from .trainer import move


def collect(model, dataset, cfg, device, passes):
    enable_mc_dropout(model)
    outputs, targets, masks = ({k: [] for k in model.tasks} for _ in range(3))
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=int(cfg['training']['batch_size']), shuffle=False):
            b = move(batch, device)
            draws = [model(input_ids=b['input_ids'], attention_mask=b['attention_mask']) for _ in range(int(passes))]
            for k in model.tasks:
                outputs[k].append(torch.stack([d[k] for d in draws]).cpu())
                targets[k].append(batch['targets'][k])
                masks[k].append(batch['observed'][k])
    return ({k: torch.cat(v, 1) for k, v in outputs.items()},
            {k: torch.cat(v) for k, v in targets.items()}, {k: torch.cat(v) for k, v in masks.items()})


def run_m5(model, datasets, records, cfg, device, directory, provenance, diagnostic=False):
    if 'test' in datasets or 'test' in records:
        raise ValueError('Default M5 does not accept untouched test')
    torch.manual_seed(int(cfg['seed']) + 500)
    calibration, targets, masks = collect(model, datasets['calibration'], cfg, device, cfg['m5']['T_mc'])
    temperatures = {k: fit_temperature(calibration[k], targets[k], masks[k], spec['kind'], 'calibration', cfg['m5']['temperature_iterations'])
                    for k, spec in model.tasks.items()}
    samples, val_targets, val_masks = collect(model, datasets['validation'], cfg, device, cfg['m5']['T_mc'])
    metrics, stats = {}, {}
    for k, spec in model.tasks.items():
        t = temperatures[k]['temperature']
        metrics[k] = {}
        for label, value in [('before', 1.0), ('after', t)]:
            metrics[k][label] = dict(nll=float(mc_nll(samples[k], val_targets[k], val_masks[k], spec['kind'], value)) if val_masks[k].sum() else None,
                                     ece=ece(probability(samples[k], spec['kind'], value), val_targets[k], val_masks[k], spec['kind'], int(cfg['m5']['calibration_bins'])))
        stats[k] = uncertainty(samples[k], spec['kind'], t)
    predictions = []
    for i, r in enumerate(records['validation']):
        taskwise = {k: {name: tensor[i].tolist() for name, tensor in stats[k].items()} for k in stats}
        predictions.append(dict(record_id=r['record_id'], tasks=taskwise,
                                composite_uncertainty=sum(v['normalized_entropy'] for v in taskwise.values()) / len(taskwise)))
    metrics_file = directory / 'calibration_metrics.json'
    prediction_file = directory / 'validation_uncertainty.jsonl'
    write_json(metrics_file, dict(provenance=provenance, temperatures=temperatures, validation_metrics=metrics,
               interpretation='NLL/ECE against observed weak targets, not gold-label calibration. Temperatures fitted on calibration only.',
               composite_definition='Unweighted mean of task entropies normalized by log(K) for categorical, log(2) for mean binary entropy. No threshold selected.'))
    write_jsonl(prediction_file, predictions)
    write_json(directory / 'validation_uncertainty.manifest.json', provenance)
    files = [metrics_file, prediction_file, directory / 'validation_uncertainty.manifest.json']
    if diagnostic:
        from scipy.stats import spearmanr
        runs, vectors = [], []
        subset = Subset(datasets['calibration'], range(min(128, len(datasets['calibration']))))
        for passes in cfg['m5']['diagnostic_passes']:
            torch.manual_seed(int(cfg['seed']) + 501)
            start = time.perf_counter()
            values, _, _ = collect(model, subset, cfg, device, passes)
            vector = sum(uncertainty(values[k], spec['kind'], temperatures[k]['temperature'])['normalized_entropy'] for k, spec in model.tasks.items()) / len(model.tasks)
            vectors.append(vector.numpy())
            runs.append(dict(passes=passes, runtime_seconds=time.perf_counter() - start, record_count=len(subset)))
        for i, r in enumerate(runs):
            rho = float(spearmanr(vectors[i], vectors[-1]).statistic)
            r['spearman_vs_30'] = rho if __import__('math').isfinite(rho) else None
        p = directory / 'mc_calibration_subset_diagnostic.json'
        write_json(p, dict(provenance=provenance, split='calibration', runs=runs))
        files.append(p)
    return files
