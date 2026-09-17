"""M3 dataset: loads M1 train/validation text and aligns M2 probabilistic soft labels.

The dataset reads M1 JSONL records and matches them to M2 weak-label JSONL files
using `record_id`. It tokenizes the configured text field and returns dicts with:
  - input_ids, attention_mask
  - soft_targets: dict of {dimension: tensor(N_dim,)}
  - record_id for alignment verification
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from m1_data_integration.config import ReviewConfig


class M3Dataset(Dataset):
    """Loads M1 records, aligns M2 soft labels by record_id, tokenizes text.

    Parameters
    ----------
    records : List[UnifiedRecord]
        M1 records (train, validation, or test split).
    weak_labels_dir : str
        Path to outputs/m2_weak_labels/ containing *_weak_labels.jsonl files.
    label_dims : Dict[str, int]
        Mapping from dimension name to number of labels.
    text_field : str
        Which record field to send to DeBERTa. Default: 'normalized_text'.
    max_length : int
        Maximum token sequence length. Default: 128.
    tokenizer : PreTrainedTokenizer, optional
        If provided, use it directly. Otherwise loaded from model_name.
    """

    DIMENSION_LABEL_COUNTS = {
        "sensitivity": 3,
        "intent": 5,
        "disclosure_scope": 2,
        "entity_tags": 4,
        "threat_content": 3,
    }

    DIMENSION_FILES = {
        "sensitivity": "sensitivity_weak_labels.jsonl",
        "intent": "intent_weak_labels.jsonl",
        "disclosure_scope": "disclosure_scope_weak_labels.jsonl",
        "entity_tags": "entity_tags_weak_labels.jsonl",
        "threat_content": "threat_content_weak_labels.jsonl",
    }

    def __init__(
        self,
        records,
        weak_labels_dir,
        label_dims=None,
        text_field="normalized_text",
        max_length=128,
        tokenizer=None,
    ):
        from transformers import AutoTokenizer

        if label_dims is None:
            label_dims = self.DIMENSION_LABEL_COUNTS

        self.records = records
        self.text_field = text_field
        self.max_length = max_length
        self.label_dims = label_dims

        # Load all M2 weak labels indexed by record_id
        self.soft_labels: Dict[str, Dict[str, np.ndarray]] = {}
        weak_dir = Path(weak_labels_dir)
        for dim, fname in self.DIMENSION_FILES.items():
            path = weak_dir / fname
            dim_labels = {}
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    dim_labels[rec["record_id"]] = np.array(
                        rec["probabilities"], dtype=np.float32
                    )
            self.soft_labels[dim] = dim_labels

        # Build aligned dataset
        self.aligned_records = []
        self.aligned_targets = []  # list of dicts: {dim: np.ndarray}
        for rec in records:
            rid = rec.record_id
            targets = {}
            all_found = True
            for dim in label_dims:
                if rid in self.soft_labels[dim]:
                    targets[dim] = self.soft_labels[dim][rid]
                else:
                    all_found = False
                    break
            if all_found:
                self.aligned_records.append(rec)
                self.aligned_targets.append(targets)

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/deberta-base"
            )

    def __len__(self):
        return len(self.aligned_records)

    def __getitem__(self, idx):
        rec = self.aligned_records[idx]
        targets = self.aligned_targets[idx]

        text = getattr(rec, self.text_field, "")
        if text is None:
            text = ""

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Build target tensors
        target_tensors = {}
        for dim, arr in targets.items():
            target_tensors[dim] = torch.from_numpy(arr)

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "record_id": rec.record_id,
            "soft_targets": target_tensors,
        }
        return result

    def get_mini_batch(self, size=4):
        """Return a small batch for smoke testing."""
        indices = list(range(min(size, len(self))))
        return [self[i] for i in indices]

    @property
    def record_ids(self):
        return [r.record_id for r in self.aligned_records]

    @property
    def num_records(self):
        return len(self.aligned_records)


def build_m3_datasets(cfg):
    """Build train, validation, and test M3Datasets from M1 split files.

    Returns a dict with keys 'train', 'val', 'test', each containing
    an M3Dataset instance and the record counts.
    """
    from m1_data_integration.schemas import EvidenceBundle, UnifiedRecord

    m3_cfg = cfg.raw.get("m3", {})
    max_length = m3_cfg.get("max_length", 128)

    def load_records(path):
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ev = d.get("evidence")
                if isinstance(ev, dict):
                    ev = EvidenceBundle(
                        entities=ev.get("entities", []),
                        dependency_relations=ev.get("dependency_relations", []),
                        regex_matches=ev.get("regex_matches", {}),
                        embedding=ev.get("embedding"),
                    )
                rec = UnifiedRecord(
                    record_id=d["record_id"],
                    domain=d["domain"],
                    source_dataset=d["source_dataset"],
                    query_text=d.get("query_text", "") or "",
                    context_text=d.get("context_text", "") or "",
                    normalized_text=d.get("normalized_text", "") or "",
                    metadata=d.get("metadata", {}),
                    evidence=ev,
                    is_duplicate=d.get("is_duplicate", False),
                    split=d.get("split"),
                )
                records.append(rec)
        return records

    split_paths = {
        "train": cfg.resolve_output("m1_dataset").parent / "m1_train_dataset.jsonl",
        "val": cfg.resolve_output("m1_dataset").parent / "m1_validation_dataset.jsonl",
        "test": cfg.resolve_output("m1_dataset").parent / "m1_test_dataset.jsonl",
    }

    label_dims = {
        "sensitivity": 3,
        "intent": 5,
        "disclosure_scope": 2,
        "entity_tags": 4,
        "threat_content": 3,
    }

    datasets = {}
    for split_name, path in split_paths.items():
        records = load_records(path)
        ds = M3Dataset(
            records=records,
            weak_labels_dir=str(cfg.resolve_output("m2_weak_labels_dir")),
            label_dims=label_dims,
            text_field="normalized_text",
            max_length=max_length,
        )
        datasets[split_name] = ds

    return datasets


def validate_alignment(train_ds, val_ds, test_ds):
    """Verify that train/val/test record sets are disjoint and non-overlapping.

    Raises ValueError if any record_id appears in more than one split.
    """
    train_ids = set(train_ds.record_ids)
    val_ids = set(val_ds.record_ids)
    test_ids = set(test_ds.record_ids)

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    if tv or tt or vt:
        raise ValueError(
            f"Record overlap detected: train_val={len(tv)}, "
            f"train_test={len(tt)}, val_test={len(vt)}"
        )

    total = len(train_ids) + len(val_ids) + len(test_ids)
    return {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "total": total,
        "overlap": 0,
    }
