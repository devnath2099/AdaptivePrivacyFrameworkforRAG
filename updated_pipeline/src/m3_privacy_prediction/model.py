"""M3 model: shared DeBERTa encoder with five task-specific prediction heads.

Architecture:
  DeBERTa encoder -> pooled representation h -> dropout -> five heads:
    sensitivity_logits:  [B, 3]
    intent_logits:       [B, 5]
    disclosure_scope_logits: [B, 2]
    entity_tags_logits:  [B, 4]
    threat_content_logits: [B, 3]
    representation:      [B, 768]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DebertaModel, DebertaTokenizer


class DeBERTaMultiTaskModel(nn.Module):
    """Shared DeBERTa encoder with five independent task heads.

    Parameters
    ----------
    model_name : str
        HuggingFace DeBERTa model identifier (default: microsoft/deberta-base).
    label_dims : dict
        Mapping from dimension name to number of labels.
    dropout_rate : float
        Dropout probability applied after the pooled representation.
    """

    def __init__(
        self,
        model_name="microsoft/deberta-base",
        label_dims=None,
        dropout_rate=0.1,
    ):
        super().__init__()
        if label_dims is None:
            label_dims = {
                "sensitivity": 3,
                "intent": 5,
                "disclosure_scope": 2,
                "entity_tags": 4,
                "threat_content": 3,
            }

        self.model_name = model_name
        self.label_dims = label_dims

        # Shared DeBERTa encoder
        self.encoder = DebertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # Five task-specific linear heads
        self.heads = nn.ModuleDict()
        for dim_name, num_labels in label_dims.items():
            self.heads[dim_name] = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """Forward pass.

        Returns a dict with logits and the pooled representation.
        Uses the CLS token (last_hidden_state[:, 0]) as the pooled
        representation since DebertaModel does not provide pooler_output.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        h = self.dropout(pooled)

        result = {"representation": h}
        for dim_name, head in self.heads.items():
            result[f"{dim_name}_logits"] = head(h)

        return result

    def get_representation(self, input_ids, attention_mask):
        """Return only the pooled representation (for M4/M5 reuse)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]
