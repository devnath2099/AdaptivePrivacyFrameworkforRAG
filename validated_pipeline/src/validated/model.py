from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .common import task_size, require_match


class PrivacyModel(nn.Module):
    def __init__(self, encoder, tasks, model_config):
        super().__init__()
        self.encoder, self.tasks, self.model_config = encoder, tasks, model_config
        for p in encoder.parameters():
            p.requires_grad_(False)
        n = int(model_config['unfreeze_last_n_layers'])
        layers = getattr(getattr(encoder, 'encoder', None), 'layer', [])
        if n < 0 or n > len(layers):
            raise ValueError('Invalid unfreeze_last_n_layers for encoder')
        if n:
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        self.dropout = nn.Dropout(float(model_config['dropout']))
        self.heads = nn.ModuleDict({k: nn.Linear(encoder.config.hidden_size, task_size(v)) for k, v in tasks.items()})

    def train(self, mode=True):
        super().train(mode)
        # Default frozen encoder is deterministic during training. M5 explicitly enables dropout.
        if int(self.model_config['unfreeze_last_n_layers']) == 0:
            self.encoder.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        pooled = self.dropout(enc.last_hidden_state[:, 0])
        return {k: head(pooled) for k, head in self.heads.items()}


def build_model(config):
    from transformers import AutoModel
    from .common import path
    enc = AutoModel.from_pretrained(config['model']['name'], revision=config['model']['revision'], cache_dir=str(path('cache/models')))
    return PrivacyModel(enc, config['active_tasks'], config['model'])


def task_losses(logits, targets, observed, tasks):
    losses = {}
    for k, spec in tasks.items():
        raw = (-(targets[k] * F.log_softmax(logits[k], -1)).sum(-1) if spec['kind'] == 'categorical'
               else F.binary_cross_entropy_with_logits(logits[k], targets[k], reduction='none'))
        mask = observed[k].to(raw)
        losses[k] = (raw * mask).sum() / mask.sum().clamp_min(1)
    return losses


def loss(logits, targets, observed, tasks, weights):
    require_match(set(weights), set(tasks), 'task weights')
    return sum(float(weights[k]) * v for k, v in task_losses(logits, targets, observed, tasks).items())


class QueryDataset(Dataset):
    def __init__(self, records, tokenizer, config):
        self.records = records
        self.tokens = tokenizer([r['query_normalized_text'] for r in records], padding='max_length',
                                truncation=True, max_length=int(config['model']['max_length']), return_tensors='pt')
        self.tasks = config['active_tasks']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return dict(input_ids=self.tokens['input_ids'][i], attention_mask=self.tokens['attention_mask'][i],
                    targets={k: torch.tensor(r['targets'][k], dtype=torch.float32) for k in self.tasks},
                    observed={k: torch.tensor(r['observed'][k], dtype=torch.float32) for k in self.tasks})


def token_diagnostics(records, tokenizer):
    import numpy as np
    lengths = [len(ids) for ids in tokenizer([r['query_normalized_text'] for r in records], truncation=False)['input_ids']]
    return dict(split='train', record_count=len(lengths), p50=float(np.percentile(lengths, 50)),
                p95=float(np.percentile(lengths, 95)), p99=float(np.percentile(lengths, 99)),
                percent_truncated_at_128=float(np.mean(np.array(lengths) > 128) * 100),
                percent_truncated_at_256=float(np.mean(np.array(lengths) > 256) * 100),
                estimated_256_vs_128_token_cost=2.0, estimated_attention_cost_ratio=4.0,
                cost_note='Analytical padded token/attention ratios, not measured total runtime or memory. Review truncation before changing configured length.')


def checkpoint_contract(config, split_id, source_hash, parent, stage):
    return dict(active_tasks=config['active_tasks'], head_shapes={k: task_size(v) for k, v in config['active_tasks'].items()},
                model_configuration=config['model'], training_configuration=config['training'],
                task_weights=config['task_weights'], adversarial_configuration=config['m4'],
                split_manifest_id=split_id, source_manifest_hash=source_hash, parent_checkpoint=parent,
                label_provenance=['snorkel_train_fit', 'synthetic_verified'] if config['augmentation']['enabled'] else ['snorkel_train_fit'],
                stage=stage)


def validate_checkpoint(checkpoint, contract):
    require_match(checkpoint.get('contract'), contract, 'checkpoint taxonomy/config/split/parent')
    for k, n in contract['head_shapes'].items():
        if checkpoint['model'][f'heads.{k}.weight'].shape[0] != n:
            raise ValueError(f'Incompatible head shape: {k}')
