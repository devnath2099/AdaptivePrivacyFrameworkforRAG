"""Offline recompute of M2 diagnostics from already-saved full-scale artifacts.

Does NOT re-run M1, does NOT regenerate embeddings, does NOT rebuild LF matrices,
does NOT re-fit the generative model. It only re-derives the diagnostics summary
(label distributions, entropy, per-domain breakdowns) from the weak-label arrays
already written to outputs/m2_weak_labels/*.npy by a previous full run.

Purpose: after fixing the diagnostics code (multi-label dimensions must NOT be
summarised via argmax), recompute the *summary* on the existing weak-label
probabilities so the audit reflects independent per-category statistics rather
than a fake exclusive argmax distribution.

Preserves the generative-model method and LF coverage/abstention/conflict stats
from the previously saved m2_diagnostics.json (those were produced by the real fit
and remain correct).

Inputs (must already exist):
  - outputs/m2_weak_labels/<dim>_weak_labels.npy    (n x K)
  - outputs/m2_weak_labels/<dim>_weak_labels.jsonl  (record_id + label_names, in order)
  - outputs/m1_train_dataset.jsonl / m1_validation_dataset.jsonl / m1_test_dataset.jsonl
    (only used to map record_id -> domain)

Output (overwritten): outputs/m2_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")

from m1_data_integration.config import load_config
from m2_label_generation.diagnostics import dimension_diagnostics
from m2_label_generation.weak_labels import DimensionWeakLabelResult

DIMENSIONS: Dict[str, bool] = {
    "sensitivity": False,
    "intent": False,
    "disclosure_scope": False,
    "entity_tags": True,
    "threat_content": True,
}


@dataclass
class _StubGen:
    method: str = "snorkel_label_model"


@dataclass
class _StubLF:
    coverage: dict
    abstention_stats: dict
    conflict_stats: dict


class _LightRecord:
    def __init__(self, record_id: str, domain: str):
        self.record_id = record_id
        self.domain = domain


def _build_domain_map(cfg) -> Dict[str, str]:
    out_dir = cfg.resolve_output("m1_dataset").parent
    names = ("m1_train_dataset.jsonl", "m1_validation_dataset.jsonl", "m1_test_dataset.jsonl")
    domain_map: Dict[str, str] = {}
    for fname in names:
        path = out_dir / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                domain_map[rec["record_id"]] = rec.get("domain", "unknown")
    return domain_map


def main() -> None:
    cfg = load_config("configs/review1.yaml")
    weak_dir = cfg.resolve_output("m2_weak_labels_dir")
    diag_path = cfg.resolve_output("m2_diagnostics")

    print("Building record_id -> domain map from M1 split files...")
    domain_map = _build_domain_map(cfg)
    print(f"  {len(domain_map)} records mapped")

    old = {}
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)

    diagnostics: Dict[str, dict] = {}

    for dim, is_multi_label in DIMENSIONS.items():
        probs = np.load(weak_dir / f"{dim}_weak_labels.npy")
        jsonl_path = weak_dir / f"{dim}_weak_labels.jsonl"

        record_ids: List[str] = []
        label_names: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                record_ids.append(obj["record_id"])
                label_names = obj["label_names"]

        records = [_LightRecord(rid, domain_map.get(rid, "unknown")) for rid in record_ids]

        old_d = old.get(dim, {})
        res = DimensionWeakLabelResult(
            dimension=dim,
            lf_matrix_result=_StubLF(
                coverage=old_d.get("lf_coverage", {}),
                abstention_stats=old_d.get("lf_abstention", {}),
                conflict_stats=old_d.get("lf_conflict", {}),
            ),
            generative_result=_StubGen(method=old_d.get("generative_model_method", "snorkel_label_model")),
            weak_labels=probs,
            label_names=label_names,
            _is_multi_label=is_multi_label,
        )

        diag = dimension_diagnostics(res, records=records)
        diagnostics[dim] = diag

    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, default=str)

    print("Corrected diagnostics written to", diag_path)
    for dim, d in diagnostics.items():
        keys = list(d.get("label_distribution", {}).keys())
        print(f"  {dim}: method={d.get('generative_model_method')} "
              f"summary_keys={keys}")


if __name__ == "__main__":
    main()