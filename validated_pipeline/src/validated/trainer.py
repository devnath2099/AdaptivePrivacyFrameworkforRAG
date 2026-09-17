from __future__ import annotations
import gc
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .adversarial import perturb, adversarial_loss
from .common import write_json, file_hash
from .model import loss, task_losses, validate_checkpoint


def move(batch, device):
    return {k: ({t: x.to(device) for t, x in v.items()} if isinstance(v, dict) else v.to(device)) for k, v in batch.items()}


def atomic_save(value, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(p)


def load_checkpoint(p, contract):
    # Only pipeline-created trusted local checkpoints, never downloaded arbitrary pickle files.
    c = torch.load(p, map_location='cpu', weights_only=False)
    validate_checkpoint(c, contract)
    return c


def evaluate(model, dataset, cfg, device, adversarial=False):
    model.eval()
    batch_size = int(cfg['m4']['adv_val_batch_size'] if adversarial else cfg['training']['batch_size'])
    numerators, counts, correct = ({k: 0.0 for k in model.tasks} for _ in range(3))
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        b = move(batch, device)
        if adversarial:
            adv, _ = perturb(model, b, cfg)
            with torch.no_grad():
                logits = model(inputs_embeds=adv, attention_mask=b['attention_mask'])
            del adv
        else:
            with torch.no_grad():
                logits = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'])
        values = task_losses(logits, b['targets'], b['observed'], model.tasks)
        for k, spec in model.tasks.items():
            n = float(b['observed'][k].sum())
            numerators[k] += float(values[k]) * n
            counts[k] += n
            agreement = ((logits[k].argmax(-1) == b['targets'][k].argmax(-1)) if spec['kind'] == 'categorical'
                         else ((logits[k] >= 0) == (b['targets'][k] >= .5)))
            correct[k] += float((agreement * b['observed'][k]).sum())
    report = {k: dict(observed_targets=counts[k], loss=numerators[k] / counts[k] if counts[k] else None,
                      weak_label_agreement=correct[k] / counts[k] if counts[k] else None) for k in model.tasks}
    total = sum(float(cfg['task_weights'][k]) * numerators[k] / counts[k] for k in model.tasks if counts[k])
    return dict(loss=total, tasks=report, interpretation='Agreement with weak targets; not natural-user task accuracy')


def train(model, train_data, val_data, cfg, device, directory, contract, provenance):
    stage = contract['stage']
    epochs = int(cfg['training'][f'{stage}_epochs'])
    batch_size = int(cfg['training']['batch_size'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']))
    last, best = directory / 'last.pt', directory / 'best.pt'
    epoch, batch_offset, best_loss, history = 0, 0, float('inf'), []
    if last.exists():
        c = load_checkpoint(last, contract)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        epoch, batch_offset, best_loss, history = c['epoch'], c['batch_offset'], c['best_loss'], c['history']
        torch.set_rng_state(c['rng']['torch'])
        random.setstate(c['rng']['python'])
        np.random.set_state(c['rng']['numpy'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(c['rng']['cuda'])

    def save(p, e, offset):
        atomic_save(dict(contract=contract, provenance=provenance, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=e, batch_offset=offset, best_loss=best_loss, history=history,
                         rng=dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                                  cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])), p)

    for e in range(epoch, epochs):
        model.train()
        order = np.random.default_rng(int(cfg['seed']) + e).permutation(len(train_data)).tolist()
        start = batch_offset if e == epoch else 0
        loader = DataLoader(Subset(train_data, order[start * batch_size:]), batch_size=batch_size,
                            generator=torch.Generator().manual_seed(int(cfg['seed']) + e), shuffle=False)
        for j, batch in enumerate(loader, start=start):
            b = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective = (adversarial_loss(model, b, cfg) if stage == 'm4' else
                         loss(model(input_ids=b['input_ids'], attention_mask=b['attention_mask']), b['targets'], b['observed'], model.tasks, cfg['task_weights']))
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite training loss')
            objective.backward()
            optimizer.step()
            if (j + 1) % int(cfg['training']['checkpoint_every_batches']) == 0:
                save(last, e, j + 1)
                print(f'{stage} epoch {e + 1}/{epochs} batch {j + 1}: checkpoint saved', flush=True)
            del objective, b
        clean = evaluate(model, val_data, cfg, device)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        robust = evaluate(model, val_data, cfg, device, adversarial=True) if stage == 'm4' else None
        selection = clean['loss'] if robust is None else clean['loss'] + float(cfg['m4']['lambda_adv']) * robust['loss']
        history.append(dict(epoch=e + 1, clean_validation=clean, fgsm_validation=robust, selection_loss=selection))
        if selection < best_loss:
            best_loss = selection
            save(best, e + 1, 0)
        save(last, e + 1, 0)
        write_json(directory / 'history.json', {'provenance': provenance, 'epochs': history})
        print(f'{stage} epoch {e + 1}: validation selection loss={selection:.6f}', flush=True)
    c = load_checkpoint(best, contract)
    model.load_state_dict(c['model'])
    return file_hash(best)
