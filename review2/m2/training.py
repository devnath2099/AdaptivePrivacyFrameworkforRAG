"""Explicit training loop with exact batch-resume state and validation-only selection."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from .alignment import collate
from .loss import token_loss, class_weights
from .inference import evaluate
from review2.common import digest, write_json


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def load_checkpoint(path, contract=None):
    # Only trusted checkpoints produced by this pipeline (contains Python/NumPy RNG state).
    state = torch.load(path, map_location='cpu', weights_only=False)
    if contract is not None and state['contract'] != contract:
        raise ValueError('Checkpoint contract mismatch')
    return state


def train(model, tokenizer, labels, train_data, validation_data, cfg, directory, contract, adv=None):
    if any(e['record']['split'] != 'train' for e in train_data):
        raise ValueError('Model fitting requires train split')
    if any(e['record']['split'] != 'validation' for e in validation_data):
        raise ValueError('Model selection requires validation split')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    weights = class_weights(train_data, len(labels)).to(device) if cfg['loss'] == 'weighted' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    epoch, offset, best_score, stale, history, elapsed_before = 0, 0, -1., 0, [], 0.
    if (directory / 'last.pt').exists():
        state = load_checkpoint(directory / 'last.pt', contract)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch, offset, best_score, stale, history = (state[k] for k in ('epoch', 'offset', 'best_score', 'stale', 'history'))
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        elapsed_before = state['elapsed_seconds']
    started = time.perf_counter()
    def save(name, e, o):
        atomic_save({'contract': contract, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': e, 'offset': o, 'best_score': best_score, 'stale': stale, 'history': history,
                     'elapsed_seconds': elapsed_before + time.perf_counter() - started,
                     'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                             'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}, directory / name)
    for e in range(epoch, cfg['epochs']):
        if stale >= cfg['patience']:
            break
        model.train()
        order = np.random.default_rng(cfg['seed'] + e).permutation(len(train_data))
        start = offset if e == epoch else 0
        losses = []
        for j in range(start, len(order), cfg['batch_size']):
            batch = collate([train_data[int(i)] for i in order[j:j + cfg['batch_size']]], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if adv is None:
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = token_loss(logits, batch['labels'], weights)
            else:
                from review2.m3.fgsm import adversarial_loss
                loss = adversarial_loss(model, batch, adv['epsilon'], adv['lambda_adv'], weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
            if (j // cfg['batch_size'] + 1) % cfg['checkpoint_every'] == 0:
                save('last.pt', e, j + cfg['batch_size'])
        metrics, _, _ = evaluate(model, validation_data, tokenizer, labels, cfg['batch_size'])
        score = metrics['f1']
        if adv is not None:
            robust, _, _ = evaluate(model, adv['validation_data'], tokenizer, labels, cfg['batch_size'])
            score = (score + robust['f1']) / 2
            metrics['obfuscated_f1'] = robust['f1']
        history.append({'epoch': e + 1, 'loss': sum(losses) / max(1, len(losses)), 'validation': metrics, 'selection_score': score})
        if score > best_score:
            best_score, stale = score, 0
            save('best.pt', e + 1, 0)
        else:
            stale += 1
        save('last.pt', e + 1, 0)
        write_json(directory / 'history.json', history)
        print(f"{directory.name} epoch {e + 1}: validation F1={metrics['f1']:.4f}", flush=True)
    best = load_checkpoint(directory / 'best.pt', contract)
    model.load_state_dict(best['model'])
    model.save_pretrained(directory / 'model')
    tokenizer.save_pretrained(directory / 'model')
    result = {'best_validation_score': best_score, 'epochs': len(history),
              'training_seconds': elapsed_before + time.perf_counter() - started,
              'parameters': sum(p.numel() for p in model.parameters()),
              'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'weights': weights.cpu().tolist() if weights is not None else None}
    write_json(directory / 'training.json', result)
    return result
