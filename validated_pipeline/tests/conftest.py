import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pytest
import torch
import yaml
from types import SimpleNamespace
from validated.common import path
from validated.model import PrivacyModel


@pytest.fixture
def cfg():
    c = yaml.safe_load(path('configs/pipeline.yaml').read_text())
    c['weak_labels']['epochs'] = 10
    c['training'].update(m3_epochs=1, m4_epochs=1, batch_size=4, checkpoint_every_batches=1)
    c['m5'].update(T_mc=3, temperature_iterations=10)
    return c


class TinyEncoder(torch.nn.Module):
    """Offline contextual fixture, never a research model."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = torch.nn.Embedding(64, 12)
        self.project = torch.nn.Linear(12, 12)
        self.dropout = torch.nn.Dropout(.2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        mask = attention_mask.unsqueeze(-1)
        pooled = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=self.dropout(self.project(x + pooled)))


@pytest.fixture
def model(cfg):
    torch.manual_seed(42)
    return PrivacyModel(TinyEncoder(), cfg['active_tasks'], cfg['model'])


@pytest.fixture
def batch(cfg):
    tasks = cfg['active_tasks']
    return dict(input_ids=torch.randint(0, 64, (4, 8)), attention_mask=torch.tensor([[1,1,1,1,1,0,0,0]] * 4),
                targets={k: torch.tensor([[1.,0.,0.]] * 4) if k != 'privacy_relevant_intent' else torch.tensor([[1.,0.,0.,0.,0.]] * 4) for k in tasks},
                observed={k: torch.ones(4) if v['kind'] == 'categorical' else torch.ones(4, 3) for k, v in tasks.items()})
