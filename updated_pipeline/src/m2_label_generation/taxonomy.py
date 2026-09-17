"""Label taxonomy for the five M2 privacy dimensions.

All label spaces are read from `configs/review1.yaml` (label_taxonomy)
rather than hard-coded, per the "no silently invented labels" rule.

`entity_tags` and `threat_content` are both implemented as MULTI-LABEL
binary dimensions (one independent present/absent sub-problem per
category), since a record can simultaneously contain multiple entity
types (e.g. both a person name and an email) or exhibit multiple threat
categories at once -- forcing a single mutually-exclusive class here
was undercounting coverage. `threat_content`'s three categories remain
exactly as specified in the project brief (re_identification,
attribute_inference, membership_inference); no extra categories are
added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

ABSTAIN = -1

MULTI_CLASS_DIMENSIONS = ["sensitivity", "intent", "disclosure_scope"]
MULTI_LABEL_DIMENSIONS = ["entity_tags", "threat_content"]  # binary per-category, evaluated independently


@dataclass
class DimensionSpec:
    name: str
    labels: List[str]
    is_multi_label: bool = False

    @property
    def num_classes(self) -> int:
        # For multi-class dims: K = number of classes.
        # For multi-label dims: each category is a binary (K=2) sub-problem.
        return 2 if self.is_multi_label else len(self.labels)

    def label_index(self, label_name: str) -> int:
        return self.labels.index(label_name)


def build_taxonomy(label_taxonomy_cfg: Dict[str, List[str]]) -> Dict[str, DimensionSpec]:
    specs: Dict[str, DimensionSpec] = {}
    for dim in MULTI_CLASS_DIMENSIONS:
        specs[dim] = DimensionSpec(name=dim, labels=list(label_taxonomy_cfg[dim]), is_multi_label=False)
    for dim in MULTI_LABEL_DIMENSIONS:
        specs[dim] = DimensionSpec(name=dim, labels=list(label_taxonomy_cfg[dim]), is_multi_label=True)
    return specs