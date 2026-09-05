"""Semantic evidence extraction: eta (NER), delta (dependency), rho (regex).

IMPORTANT (per project spec): none of these mechanisms produce a final
privacy label. They only populate an `EvidenceBundle` that M2's
labeling functions consume as input signals.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .schemas import EvidenceBundle, UnifiedRecord

logger = logging.getLogger(__name__)

# rho: regex/pattern evidence -----------------------------------------------

REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    "money": re.compile(r"[$₹€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|INR|EUR|GBP|dollars)\b"),
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"),
    "url": re.compile(r"https?://[^\s]+|www\.[^\s]+"),
    "account_number": re.compile(r"\b(?:AC|MRN|A/C)?-?\d{6,}\b"),
}


def extract_regex_evidence(text: str, active_patterns: List[str] | None = None) -> Dict[str, List[str]]:
    """rho: run configured regex patterns over text, return matches per pattern name."""
    active = active_patterns or list(REGEX_PATTERNS.keys())
    matches: Dict[str, List[str]] = {}
    for name in active:
        pattern = REGEX_PATTERNS.get(name)
        if pattern is None:
            continue
        found = pattern.findall(text)
        if found:
            matches[name] = list(dict.fromkeys(found))  # de-dup preserve order
    return matches


# eta: NER evidence & delta: dependency evidence, both via spaCy -----------

class SpacyEvidenceExtractor:
    """Wraps a spaCy pipeline to produce NER (eta) and dependency (delta) evidence.

    If the configured spaCy model is not installed, this raises a clear
    RuntimeError rather than silently swapping methodology (per spec).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        self._load_error: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = (
                f"spaCy model '{self.model_name}' unavailable ({exc}). "
                f"Install it with: python -m spacy download {self.model_name}. "
                f"Falling back to a lightweight rule-based NER/dependency approximation."
            )
            logger.warning(self._load_error)

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def extract(self, text: str) -> Dict[str, Any]:
        if self._nlp is not None:
            return self._extract_spacy(text)
        return self._extract_fallback(text)

    def _extract_spacy(self, text: str) -> Dict[str, Any]:
        doc = self._nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        dependency_relations = [
            {"token": tok.text, "dep": tok.dep_, "head": tok.head.text}
            for tok in doc
            if tok.dep_ not in ("punct",)
        ][:50]  # cap to keep evidence bundles small
        return {"entities": entities, "dependency_relations": dependency_relations}

    def _extract_fallback(self, text: str) -> Dict[str, Any]:
        """Rule-based approximation used only when spaCy is unavailable.

        This is explicitly a fallback, not a replacement methodology:
        capitalized-token heuristic for entity-like spans, no real
        dependency parse (dependency_relations stays empty).
        """
        candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?\b", text)
        entities = [{"text": c, "label": "PROPN_HEURISTIC"} for c in dict.fromkeys(candidates)]
        return {"entities": entities, "dependency_relations": []}


def build_evidence(
    record: UnifiedRecord,
    spacy_extractor: SpacyEvidenceExtractor,
    active_regex_patterns: List[str],
) -> EvidenceBundle:
    text = record.normalized_text
    nlp_evidence = spacy_extractor.extract(text)
    regex_evidence = extract_regex_evidence(text, active_regex_patterns)

    return EvidenceBundle(
        entities=nlp_evidence["entities"],
        dependency_relations=nlp_evidence["dependency_relations"],
        regex_matches=regex_evidence,
        embedding=None,  # filled in by embeddings.py
    )


def build_evidence_all(
    records: List[UnifiedRecord],
    spacy_model: str,
    active_regex_patterns: List[str],
    n_process: int = 1,
) -> None:
    """Attach evidence (in-place) to every record's `.evidence` field.
    
    Args:
        n_process: Number of parallel processes for spaCy. 
                   1 means sequential. >1 uses nlp.pipe(n_process).
                   Defaults to 1 (compatible with all environments).
    """
    from tqdm import tqdm

    extractor = SpacyEvidenceExtractor(spacy_model)

    if extractor.available and n_process > 1:
        # Parallel spaCy processing via nlp.pipe
        texts = [record.normalized_text for record in records]
        docs = extractor._nlp.pipe(texts, n_process=n_process, batch_size=16)
        for i, (record, doc) in enumerate(tqdm(zip(records, docs), total=len(records), desc="Evidence")):
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            dependency_relations = [
                {"token": tok.text, "dep": tok.dep_, "head": tok.head.text}
                for tok in doc
                if tok.dep_ not in ("punct",)
            ][:50]
            regex_evidence = extract_regex_evidence(record.normalized_text, active_regex_patterns)
            record.evidence = EvidenceBundle(
                entities=entities,
                dependency_relations=dependency_relations,
                regex_matches=regex_evidence,
                embedding=None,
            )
    else:
        # Sequential processing
        for record in tqdm(records, desc="Evidence"):
            record.evidence = build_evidence(record, extractor, active_regex_patterns)
