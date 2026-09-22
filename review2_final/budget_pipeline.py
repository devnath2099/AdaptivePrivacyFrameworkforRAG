"""One bounded research run; saved-model-only reference and demo evaluation."""
import gc
import math
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from .common import read, write, digest, seed_all, development_guard
from .m1 import audit, build, load_split, statistics
from .neural import (tokenizer_for, encode, loader, load_model, train_epoch,
                     predict_logits, decode, save_checkpoint, device_for)
from .experiments import environment, checkpoint_signature
from .m4 import softmax, fit_temperature, gold_array, sample_predictive, report, scale
from .metrics import entity_metrics
from .budget_sampling import ordered_groups, choose_training, demo_plans, metric_differences


class Budget:
    def __init__(self, seconds, started=None):
        self.started = time.time() if started is None else started
        self.deadline = self.started + seconds

    @property
    def remaining(self):
        return max(0., self.deadline-time.time())

    def check(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Three-hour budget exhausted. Saved artifacts remain; no automatic retraining.')


def release(model=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batch_data(rows, tokenizer, labels, cfg):
    windows = encode(rows, tokenizer, labels, cfg['max_length'])
    return windows, loader(windows, tokenizer, cfg['batch_size'])


def subset_windows(windows, rows, selected):
    ids = {r['record_id'] for r in selected}
    return [w for w in windows if rows[w['doc']]['record_id'] in ids]


def timed_predict(model, batches, rows, labels, device, budget, epsilon=0.):
    started = time.perf_counter()
    logits = predict_logits(model, batches, rows, len(labels), device, epsilon=epsilon,
                            budget_check=budget.check if budget else None)
    metrics = entity_metrics(rows, decode(logits, labels), sorted({t[2:] for t in labels if t != 'O'}))
    return logits, {'entities': metrics, 'seconds': time.perf_counter()-started,
                    'documents': len(rows), 'windows': len(batches.dataset)}


def probe(train, windows, tokenizer, labels, cfg, budget):
    """Disposable training-only probe; final M2 reloads pristine public weights."""
    started = time.perf_counter()
    groups, info = ordered_groups(train, cfg['seed'])
    selected = []
    for group in groups:
        selected.extend(group)
        if len(selected) >= max(24, info['anchor_documents']):
            break
    candidates = subset_windows(windows, train, selected)
    # Full-length/padded windows give a conservative memory/step-time probe.
    candidates.sort(key=lambda w: -len(w['input_ids']))
    needed = max(1, cfg['probe_steps']*cfg['batch_size'])
    chosen = candidates[:needed]
    batches = loader(chosen, tokenizer, cfg['batch_size'])
    device = device_for(cfg['device'])
    seed_all(cfg['seed'])
    model = load_model(labels, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    timings = {}
    for name, eps in [('clean', 0.), ('fgsm', cfg['fgsm_epsilon'])]:
        train_epoch(model, batches, optimizer, device, eps, max_steps=1, description=f'Probe warmup {name}', budget_check=budget.check)
        out = train_epoch(model, batches, optimizer, device, eps, description=f'Probe {name}', budget_check=budget.check)
        timings[name+'_step_seconds'] = out['seconds']/out['steps']
    # Use the same training documents for timed forward/white-box evaluation.
    _, pb = batch_data(selected, tokenizer, labels, cfg)
    for name, eps in [('forward', 0.), ('attack', cfg['fgsm_epsilon'])]:
        _, out = timed_predict(model, pb, selected, labels, device, budget, eps)
        timings[name+'_batch_seconds'] = out['seconds']/len(pb)
    timings.update({'seconds': time.perf_counter()-started, 'probe_document_ids': [r['record_id'] for r in selected],
                    'probe_training_statistics': statistics(selected), 'batch_size': cfg['batch_size'],
                    'note': 'Disposable model updates for timing only; not a research training run or checkpoint. Final training resets seed and reloads the public backbone.'})
    del optimizer, model
    release()
    return timings


def plan_run(train, windows, cal_windows, val_windows, manifest, timings, cfg, remaining):
    groups, info = ordered_groups(train, cfg['seed'])
    counts = Counter(train[w['doc']]['record_id'] for w in windows)
    batch = cfg['batch_size']
    nbcal, nbval = math.ceil(len(cal_windows)/batch), math.ceil(len(val_windows)/batch)
    # Only structural M1 counts, no test predictions or tokenization used for planning.
    density = max(len(cal_windows)/manifest['splits']['calibration']['documents'],
                  len(val_windows)/manifest['splits']['validation']['documents'])
    nbtest = math.ceil(density*manifest['splits']['test']['documents']/batch)
    f, a = timings['forward_batch_seconds'], timings['attack_batch_seconds']
    unit = cfg['m2_epochs']*timings['clean_step_seconds'] + cfg['m3_epochs']*timings['fgsm_step_seconds']
    candidates = []
    for maximum in sorted(set(cfg['mc_passes'])):
        passes = [p for p in cfg['mc_passes'] if p <= maximum]
        # Two detectors clean+attacked validation and reference; deterministic+MC M4;
        # independent load-only runs of every nested demo candidate plus a margin for anchors.
        demo_factor = sum(cfg['demo_fractions']) + .5
        fixed = 2*nbval*(f+a) + (nbcal+nbval)*(1+maximum)*f
        fixed += nbtest*(2*(f+a)+maximum*f) + demo_factor*nbtest*(1+maximum)*f
        fixed = fixed*cfg['safety_factor'] + cfg['overhead_reserve_seconds']
        try:
            rows, nw = choose_training(groups, info['anchor_documents'], counts, remaining-fixed,
                                       unit*cfg['safety_factor'], batch)
        except RuntimeError:
            continue
        candidates.append({'rows': rows, 'windows': nw, 'mc_passes': passes,
                           'estimated_remaining_seconds': fixed+math.ceil(nw/batch)*unit*cfg['safety_factor'],
                           'fixed_evaluation_reserve_seconds': fixed})
    if not candidates:
        raise RuntimeError('Probe predicts insufficient time for rare-preserving training AND full reference/M4 evaluation. No final training started.')
    best = max(candidates, key=lambda c: (len(c['rows']), max(c['mc_passes'])))
    rows = best.pop('rows')
    return rows, {**best, 'selection': info, 'training_record_ids': [r['record_id'] for r in rows],
                  'training_statistics': statistics(rows), 'available_training_documents': len(train),
                  'remaining_seconds_at_plan': remaining, 'estimated_reference_batches': nbtest,
                  'objective': 'Largest document count in deterministic rare-preserving multilabel order that fits conservative estimate; ties favor more MC passes.',
                  'limitations': 'Timing estimate, not a wall-clock or model-quality guarantee. One fixed epsilon; no learning-curve sweep or clean-continuation ablation under this budget.'}


def train_once(name, root, windows, tokenizer, labels, cfg, budget, source=None):
    folder = root/'models'/name
    done = folder/'completed.json'
    if done.exists():
        if checkpoint_signature(folder/'model') != read(done)['signature']:
            raise ValueError('Saved checkpoint integrity mismatch')
        print(f'{name}: loading completed training record; no retraining', flush=True)
        return read(done)
    if (folder/'started.json').exists():
        raise RuntimeError(f'{name} has an interrupted training attempt. Refusing to silently repeat it; saved partial checkpoint requires explicit recovery.')
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/'started.json', {'started': time.time(), 'source': str(source), 'config': cfg})
    seed_all(cfg['seed'])
    device = device_for(cfg['device'])
    model = load_model(labels, source, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, started = [], time.perf_counter()
    epochs = cfg['m2_epochs'] if name == 'm2' else cfg['m3_epochs']
    try:
        for epoch in range(epochs):
            batches = loader(windows, tokenizer, cfg['batch_size'], True, cfg['seed']+epoch)
            history.append(train_epoch(model, batches, optimizer, device,
                       epsilon=0. if name == 'm2' else cfg['fgsm_epsilon'],
                       description=f'{name.upper()} epoch {epoch+1}/{epochs}', budget_check=budget.check))
            save_checkpoint(model, tokenizer, folder/'model')
            write(folder/'history.json', history)
    except BaseException:
        save_checkpoint(model, tokenizer, folder/'interrupted_model')
        torch.save({'optimizer': optimizer.state_dict(), 'completed_epochs': len(history)}, folder/'interrupted_optimizer.pt')
        write(folder/'interrupted.json', {'history': history, 'seconds': time.perf_counter()-started,
                                        'status': 'Incomplete; not a final trained model'})
        raise
    result = {'history': history, 'seconds': time.perf_counter()-started, 'signature': checkpoint_signature(folder/'model')}
    write(done, result)
    del model, optimizer
    release()
    return result


def development_evaluation(root, val, vb, cal, cb, labels, cfg, passes, budget):
    device = device_for(cfg['device'])
    comparison = {}
    for name in ('m2', 'm3'):
        model = load_model(labels, root/'models'/name/'model', device)
        _, clean = timed_predict(model, vb, val, labels, device, budget)
        _, attack = timed_predict(model, vb, val, labels, device, budget, cfg['fgsm_epsilon'])
        comparison[name] = {'clean': clean, 'fgsm': attack,
                            'selection_score': (clean['entities']['micro']['f1']+attack['entities']['micro']['f1'])/2}
        del model
        release()
    write(root/'validation_comparison.json', comparison)
    selected = max(comparison, key=lambda n: comparison[n]['selection_score'])
    model = load_model(labels, root/'models'/selected/'model', device)
    yg = gold_array(cal, labels)
    logits, _ = timed_predict(model, cb, cal, labels, device, budget)
    temp = fit_temperature(softmax(np.concatenate(logits)), yg, budget.check)
    logits, timing = timed_predict(model, vb, val, labels, device, budget)
    deterministic = report(val, softmax(np.concatenate(logits)), labels, temp, cfg['ece_bins'])
    deterministic['seconds'] = timing['seconds']
    write(root/'deterministic_calibration.json', deterministic)
    temperatures = {}
    for count, p, u in sample_predictive(model, cb, cal, labels, device, passes, cfg['seed'], budget.check):
        temperatures[count] = fit_temperature(p, yg, budget.check)
    comparisons = []
    for count, p, u in sample_predictive(model, vb, val, labels, device, passes, cfg['seed']+10000, budget.check):
        item = report(val, p, labels, temperatures[count], cfg['ece_bins'], u['predictive_entropy'])
        item.update({'passes': count, 'inference_seconds': u['seconds']})
        comparisons.append(item)
        write(root/'mc_validation.json', comparisons)
    best = min(x['calibrated_token_metrics']['nll'] for x in comparisons)
    chosen = min((x for x in comparisons if x['calibrated_token_metrics']['nll'] <= best+cfg['mc_nll_tolerance']), key=lambda x: x['passes'])
    result = {'model': selected, 'checkpoint': f'models/{selected}/model', 'passes': chosen['passes'],
              'temperature': chosen['temperature'], 'deterministic_temperature': temp,
              'selection_basis': 'Validation only; mean clean/FGSM entity F1, then minimum MC passes within configured NLL tolerance.',
              'model_signature': checkpoint_signature(root/'models'/selected/'model')}
    write(root/'calibration.json', result)
    del model
    release()
    return result


def evaluate_saved(root, rows, cfg, budget=None, name='demo'):
    """Inference ONLY: no optimizer, calibration fitting, or training code."""
    root = Path(root)
    started = time.perf_counter()
    saved = read(root/'calibration.json')
    checkpoint = root/saved['checkpoint']
    if checkpoint_signature(checkpoint) != saved['model_signature']:
        raise ValueError('Saved model changed')
    labels = read(root/'labels.json')
    tokenizer = tokenizer_for(checkpoint)
    _, batches = batch_data(rows, tokenizer, labels, cfg)
    device = device_for(cfg['device'])
    model = load_model(labels, checkpoint, device)
    logits, deterministic = timed_predict(model, batches, rows, labels, device, budget)
    result = {'name': name, 'documents': len(rows), 'windows': len(batches.dataset), 'deterministic': deterministic}
    for count, p, u in sample_predictive(model, batches, rows, labels, device, [saved['passes']], cfg['seed']+20000,
                                        budget.check if budget else None):
        result['mc'] = report(rows, p, labels, saved['temperature'], cfg['ece_bins'], u['predictive_entropy'])
        result['mc']['inference_seconds'] = u['seconds']
    del model
    release()
    result['wall_seconds'] = time.perf_counter()-started
    return result


def reference_and_demos(root, data, cfg, budget):
    selected = read(root/'calibration.json')
    freeze = {'budget_run': str(root.resolve()), 'calibration_sha256': digest(root/'calibration.json'),
              'config_sha256': digest(root/'config.json'), 'plan_sha256': digest(root/'plan.json')}
    write(data/'TEST_OPENED.json', freeze)
    write(root/'FROZEN.json', freeze)
    rows = load_split(data, 'test')
    # Freeze membership before obtaining any predictions.
    plans = demo_plans(rows, cfg['seed']+1, cfg['demo_fractions'])
    write(root/'demo_subsets.json', plans)
    write(root/'reference_test.json', rows)
    labels = read(root/'labels.json')
    tokenizer = tokenizer_for(root/selected['checkpoint'])
    _, batches = batch_data(rows, tokenizer, labels, cfg)
    comparison = {}
    for name in ('m2', 'm3'):
        model = load_model(labels, root/'models'/name/'model', device_for(cfg['device']))
        _, clean = timed_predict(model, batches, rows, labels, cfg['device'], budget)
        _, attack = timed_predict(model, batches, rows, labels, cfg['device'], budget, cfg['fgsm_epsilon'])
        comparison[name] = {'clean': clean, 'fgsm': attack}
        write(root/'reference_robustness.json', comparison)
        del model
        release()
    reference = evaluate_saved(root, rows, cfg, budget, 'fixed_full_reference')
    write(root/'reference_results.json', reference)
    comparisons = []
    for i, candidate in enumerate(plans['candidates']):
        ids = set(candidate['record_ids'])
        subset = [r for r in rows if r['record_id'] in ids]
        measured = reference if len(subset) == len(rows) else evaluate_saved(root, subset, cfg, budget, f'demo_{len(subset)}')
        delta = metric_differences(reference['mc']['entities'], measured['mc']['entities'], cfg['demo_metric_tolerance'])
        comparisons.append({'candidate': i, 'statistics': statistics(subset), 'results': measured,
                            'comparison_to_reference': delta})
        write(root/'demo_comparison.json', comparisons)
    # Choose by measured runtime/size ONLY. Agreement is a reported outcome, not a search criterion.
    eligible = [x for x in comparisons if x['results']['wall_seconds'] <= cfg['demo_target_seconds']]
    if eligible:
        chosen = max(eligible, key=lambda x: x['statistics']['documents'])
        status = 'within_runtime_target'
    else:
        chosen = min(comparisons, key=lambda x: x['results']['wall_seconds'])
        status = 'runtime_target_not_met_even_at_smallest_planned_subset'
    ids = set(plans['candidates'][chosen['candidate']]['record_ids'])
    write(root/'demo_test.json', [r for r in rows if r['record_id'] in ids])
    write(root/'demo_selection.json', {'status': status, 'candidate': chosen['candidate'],
          'documents': len(ids), 'seconds': chosen['results']['wall_seconds'],
          'metric_agreement': chosen['comparison_to_reference'],
          'rule': 'Largest fixed nested subset measured within runtime target; no choice based on higher scores or agreement. Under five minutes is acceptable; no artificial delay.'})
    return {'reference': reference, 'demo_selection': read(root/'demo_selection.json')}


def export_bundle(root):
    destination = root.parent/(root.name+'_saved.zip')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as archive:
        for p in root.rglob('*'):
            if p.is_file():
                archive.write(p, Path(root.name)/p.relative_to(root))
    return str(destination)


def run_budgeted(output='review2_final/runs/three_hour', config_path=None, session_started=None):
    base = Path(__file__).resolve().parent
    root = Path(output)
    cfg = read(config_path or base/'budget_config.json')
    if (root/'COMPLETE.json').exists():
        print('Completed run found. No training is repeated. Use run_saved_demo() for inference.', flush=True)
        return read(root/'COMPLETE.json')
    if root.exists():
        raise RuntimeError('An incomplete run already exists. Preserving it and refusing automatic retraining. Review its state before recovery.')
    root.mkdir(parents=True)
    write(root/'config.json', cfg)
    budget = Budget(cfg['budget_seconds'], session_started)
    torch.set_num_threads(4)
    state = {'status': 'running', 'stage': 'M1', 'timings': {}}
    write(root/'STATE.json', state)
    try:
        device_for(cfg['device'])
        started = time.perf_counter()
        from .acquire import main as acquire
        acquire()
        data = base/'data/prepared'
        development_guard(data)
        if not (data/'manifest.json').exists():
            build(base/'data/raw/train.json', base/'kaggle_release.json', data,
                  {'train': .7, 'calibration': .1, 'validation': .1, 'test': .1}, cfg['seed'])
        else:
            audited, fresh = audit(base/'data/raw/train.json', read(base/'kaggle_release.json'))
            del audited
            if fresh['source']['source_sha256'] != read(data/'manifest.json')['source_sha256']:
                raise ValueError('Prepared splits do not match acquired release')
        manifest = read(data/'manifest.json')
        for name, sha in manifest['files'].items():
            if digest(data/f'{name}.json') != sha:
                raise ValueError('Split fingerprint changed')
        for name in ('audit.json', 'manifest.json', 'labels.json'):
            shutil.copy2(data/name, root/name)
        write(root/'environment.json', environment())
        print(read(root/'audit.json')['statistics'], flush=True)
        print('Split coverage warnings:', manifest['coverage_warnings'], flush=True)
        train, cal, val = (load_split(data, n) for n in ('train','calibration','validation'))
        labels = read(root/'labels.json')
        tokenizer = tokenizer_for()
        tw, _ = batch_data(train, tokenizer, labels, cfg)
        cw, cb = batch_data(cal, tokenizer, labels, cfg)
        vw, vb = batch_data(val, tokenizer, labels, cfg)
        state['timings']['M1_audit_split_tokenization'] = time.perf_counter()-started
        state['stage'] = 'timing_probe'
        write(root/'STATE.json', state)
        measured = probe(train, tw, tokenizer, labels, cfg, budget)
        write(root/'probe.json', measured)
        state['timings']['probe'] = measured['seconds']
        rows, plan = plan_run(train, tw, cw, vw, manifest, measured, cfg, budget.remaining)
        write(root/'plan.json', plan)
        write(root/'training_subset.json', {'record_ids': plan['training_record_ids'], 'statistics': statistics(rows)})
        print(f"Selected {len(rows)}/{len(train)} stratified training documents; MC budgets={plan['mc_passes']}; estimated remaining {plan['estimated_remaining_seconds']/60:.1f} min", flush=True)
        training_windows = subset_windows(tw, train, rows)
        for name in ('m2', 'm3'):
            state['stage'] = name.upper()
            write(root/'STATE.json', state)
            result = train_once(name, root, training_windows, tokenizer, labels, cfg, budget,
                                root/'models/m2/model' if name == 'm3' else None)
            state['timings'][name+'_training'] = result['seconds']
        state['stage'] = 'M4_and_validation'
        write(root/'STATE.json', state)
        started = time.perf_counter()
        development_evaluation(root, val, vb, cal, cb, labels, cfg, plan['mc_passes'], budget)
        state['timings']['M4_and_validation'] = time.perf_counter()-started
        state['stage'] = 'reference_and_load_only_demo_profiles'
        write(root/'STATE.json', state)
        started = time.perf_counter()
        reference_and_demos(root, data, cfg, budget)
        state['timings']['reference_and_demos'] = time.perf_counter()-started
        state.update({'status': 'complete', 'stage': 'complete', 'wall_seconds': time.time()-budget.started,
                      'within_budget_before_export': budget.remaining > 0,
                      'training_documents': len(rows), 'reference_test_documents': manifest['splits']['test']['documents'],
                      'demo': read(root/'demo_selection.json')})
        write(root/'STATE.json', state)
        write(root/'COMPLETE.json', state)
        bundle = export_bundle(root)
        print(f'Finished. Saved-model bundle: {bundle}', flush=True)
        return state
    except BaseException as exc:
        state.update({'status': 'incomplete', 'error': str(exc), 'wall_seconds': time.time()-budget.started})
        write(root/'STATE.json', state)
        raise


def run_saved_demo(output='review2_final/runs/three_hour'):
    root = Path(output)
    if not (root/'COMPLETE.json').exists():
        raise RuntimeError('Completed trained artifacts required; this function never trains.')
    cfg = read(root/'config.json')
    rows = read(root/'demo_test.json')
    result = evaluate_saved(root, rows, cfg)
    write(root/'latest_demo_results.json', result)
    print(f"Loaded saved model: {len(rows)} documents in {result['wall_seconds']:.1f}s", flush=True)
    return result
