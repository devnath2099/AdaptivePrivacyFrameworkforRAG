"""Semantic evidence extraction: eta (NER), delta (dependency), rho (regex).

IMPORTANT (per project spec): none of these mechanisms produce a final
privacy label. They only populate an `EvidenceBundle` that M2's
labeling functions consume as input signals.
"""
from __future__ import annotations

import json
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
    checkpoint_dir: str = "outputs/cache",
    checkpoint_interval_minutes: int = 10,
    partition_id: int = 0,
    num_partitions: int = 1,
) -> None:
    """Attach evidence (in-place) to every record's `.evidence` field.

    Supports:
    - Parallel spaCy via nlp.pipe(n_process)
    - Checkpoint saving every ~10 minutes so progress isn't lost
    - Partitioning for multi-notebook processing (partition_id/num_partitions)

    Args:
        n_process: Number of parallel processes for spaCy. 1 = sequential.
        checkpoint_dir: Directory to save/load checkpoints.
        checkpoint_interval_minutes: Save checkpoint every N minutes.
        partition_id: This notebook's partition index (0-based).
        num_partitions: Total number of partitions (notebooks).
    """
    import os
    import time

    from tqdm import tqdm

    # Apply partition filtering
    if num_partitions > 1:
        records = records[partition_id::num_partitions]
        print(f"[Partition {partition_id}/{num_partitions}] Processing {len(records)} records")

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"evidence_checkpoint_{partition_id}.jsonl")
    done_path = os.path.join(checkpoint_dir, f"evidence_done_{partition_id}.marker")

    # Check for existing checkpoint
    if os.path.exists(done_path):
        print(f"[Partition {partition_id}] Checkpoint already exists, skipping evidence")
        # Load cached records back into the list
        _load_partition_checkpoint(records, checkpoint_path, partition_id)
        return

    extractor = SpacyEvidenceExtractor(spacy_model)

    if extractor.available and n_process > 1:
        texts = [record.normalized_text for record in records]
        docs = extractor._nlp.pipe(texts, n_process=n_process, batch_size=16)
        _process_with_checkpoint(records, docs, extractor, active_regex_patterns,
                                  checkpoint_path, done_path, checkpoint_interval_minutes, partition_id)
    else:
        _process_with_checkpoint_sequential(records, extractor, active_regex_patterns,
                                             checkpoint_path, done_path, checkpoint_interval_minutes, partition_id)


def _load_partition_checkpoint(records: List[UnifiedRecord], checkpoint_path: str, partition_id: int) -> None:
    """Load cached evidence from checkpoint file back into records (in-place)."""
    if not os.path.exists(checkpoint_path):
        return
    loaded = 0
    cached_records: List[UnifiedRecord] = []
    with open(checkpoint_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record = UnifiedRecord.from_dict(data)
            if record.evidence is not None:
                loaded += 1
            cached_records.append(record)
    # Update records in-place
    records.clear()
    records.extend(cached_records)
    print(f"[Partition {partition_id}] Loaded {loaded} records from checkpoint")


def _process_with_checkpoint_sequential(
    records: List[UnifiedRecord],
    extractor,
    active_regex_patterns: List[str],
    checkpoint_path: str,
    done_path: str,
    interval_minutes: int,
    partition_id: int,
) -> None:
    """Process records sequentially with periodic checkpoint saves."""
    import time
    from tqdm import tqdm

    total = len(records)
    last_checkpoint = time.time()
    interval_seconds = interval_minutes * 60

    for i, record in enumerate(tqdm(records, desc=f"Evidence P{partition_id}")):
        record.evidence = build_evidence(record, extractor, active_regex_patterns)

        # Save checkpoint every interval_minutes
        now = time.time()
        if now - last_checkpoint >= interval_seconds:
            _save_checkpoint(records[:i + 1], checkpoint_path, partition_id)
            last_checkpoint = now
            print(f"[Partition {partition_id}] Checkpoint saved at record {i + 1}/{total}")

    # Final checkpoint
    _save_checkpoint(records, checkpoint_path, partition_id)
    with open(done_path, "w") as f:
        json.dump({"n_records": len(records), "complete": True}, f)
    print(f"[Partition {partition_id}] Evidence complete for {len(records)} records")


def _process_with_checkpoint(
    records: List[UnifiedRecord],
    docs,
    extractor,
    active_regex_patterns: List[str],
    checkpoint_path: str,
    done_path: str,
    interval_minutes: int,
    partition_id: int,
) -> None:
    """Process with parallel spaCy and periodic checkpoint saves."""
    import time
    from tqdm import tqdm

    total = len(records)
    last_checkpoint = time.time()
    interval_seconds = interval_minutes * 60
    texts = [record.normalized_text for record in records]

    for i, (record, doc) in enumerate(tqdm(zip(records, docs), total=total, desc=f"Evidence P{partition_id}")):
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

        now = time.time()
        if now - last_checkpoint >= interval_seconds:
            _save_checkpoint(records[:i + 1], checkpoint_path, partition_id)
            last_checkpoint = now
            print(f"[Partition {partition_id}] Checkpoint saved at record {i + 1}/{total}")

    _save_checkpoint(records, checkpoint_path, partition_id)
    with open(done_path, "w") as f:
        json.dump({"n_records": len(records), "complete": True}, f)
    print(f"[Partition {partition_id}] Evidence complete for {len(records)} records")


def _save_checkpoint(records: List[UnifiedRecord], checkpoint_path: str, partition_id: int) -> None:
    """Save current state of records to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), default=str) + "\n")
