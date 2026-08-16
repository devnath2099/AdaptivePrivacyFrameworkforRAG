"""Per-dataset loader adapters for the four Review 1 datasets.

Design decision (project-specific, not from any paper):
Each loader first attempts to pull real samples via the HuggingFace
`datasets` library, extracting dataset-specific fields via a small
per-dataset spec (HF id/config/split + field-name mapping). If that
fails (offline environment, gated dataset, network restrictions) the
loader falls back to a small, clearly labeled *synthetic* generator so
the rest of the pipeline (M1 -> M2 -> validation) stays runnable
end-to-end on a laptop. Fallback records are tagged
`source_mode="synthetic_fallback"`; real records are tagged
`source_mode="real"` (or `"real_cached"` when served from the local
cache instead of the network), so nobody mistakes one for the other.

Local dataset caching (project-specific, see LocalDatasetCache below):
extracted (query, answer) rows are cached to `data_cache.root` as
JSONL, decoupled from `max_samples_per_dataset` -- the cache stores at
least CACHE_MIN_ROWS rows (or `max_samples`, whichever is larger) on
first download, so later runs with the same or a smaller sample size
reuse the cache instead of re-downloading.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schemas import RawRecord

logger = logging.getLogger(__name__)


def _hc_extract(row: Dict[str, Any]):
    """lavita/ChatDoctor-HealthCareMagic-100k (split 'train'): fields
    instruction, input, output. `input` is the patient's question,
    `output` is the doctor's response.
    """
    return row.get("input") or row.get("instruction") or "", row.get("output") or ""


def _fiqa_extract(row: Dict[str, Any]):
    """BeIR/fiqa, config 'corpus' (split 'corpus'): fields _id, title, text.
    This is a document corpus, not question/answer pairs, so the passage
    `text` is used as the query_text and `title` (often empty) as context.
    """
    return row.get("text") or "", row.get("title") or ""


def _hotpot_extract(row: Dict[str, Any]):
    """hotpotqa/hotpot_qa, config 'distractor' (split 'validation'): fields
    id, question, answer, type, level, supporting_facts, context
    (context.sentences is a list of lists of strings per supporting title).
    """
    ctx = row.get("context")
    ctx_text = " ".join(sum(ctx.get("sentences", []), [])) if isinstance(ctx, dict) else ""
    return row.get("question") or "", (row.get("answer") or "") + " " + ctx_text


def _nq_extract(row: Dict[str, Any]):
    """sentence-transformers/natural-questions (split 'train'): flat fields
    query, answer -- no nested 'question.text' structure (that was the old
    natural_questions dataset's schema; this one is different).
    """
    return row.get("query") or "", row.get("answer") or ""


# Each spec bundles: dataset naming, default HF split/config, a
# row -> (query, answer) extractor, and a small fallback template set with
# a per-record kwargs formatter, so the loader class below needs no
# per-dataset subclassing. hf_split/hf_config defaults here are overridden
# by `split`/`hf_config` keys in configs/review1.yaml's `datasets` section
# when present.
DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "healthcare": {
        "source_dataset": "healthcare_magic_100k", "domain": "medical",
        "hf_split": "train", "hf_config": None, "extract": _hc_extract,
        "fallback_templates": [
            ("I am {age} years old and have had chest pain for two days, my name is {name}.",
             "Chest pain warrants urgent evaluation; please see a cardiologist."),
            ("My patient ID is {pid}, diagnosed with type 2 diabetes, blood sugar 210 mg/dL.",
             "Recommend metformin dosage review and dietary counseling."),
            ("Can you explain what a fever above 102F means for a {age} year old child?",
             "High fever in children should be monitored and a pediatrician consulted."),
            ("I live at {addr} and my doctor prescribed lisinopril for hypertension.",
             "Lisinopril is an ACE inhibitor commonly used for blood pressure control."),
        ],
        "fallback_kwargs": lambda rng, i: dict(age=rng.randint(20, 80), name=f"Patient{i}",
                                                pid=f"MRN-{1000+i}", addr=f"{i} Elm Street"),
    },
    "fiqa": {
        "source_dataset": "fiqa_2018", "domain": "financial",
        "hf_split": "corpus", "hf_config": "corpus", "extract": _fiqa_extract,
        "fallback_templates": [
            ("My account number is {acct} and my balance dropped by $5,200 this month, why?",
             "Large balance drops are often due to scheduled transfers or fees; check statements."),
            ("Should I invest $10,000 in index funds or pay off my credit card debt first?",
             "Generally pay off high-interest debt before investing surplus cash."),
            ("I earn ${salary} a year, is that enough to qualify for a mortgage?",
             "Mortgage eligibility depends on debt-to-income ratio, not income alone."),
            ("My routing number is 021000021, can someone use it to steal from me?",
             "A routing number alone cannot be used to withdraw funds without authorization."),
        ],
        "fallback_kwargs": lambda rng, i: dict(acct=f"AC-{500000+i}", salary=40000 + i * 500),
    },
    "hotpotqa": {
        "source_dataset": "hotpotqa", "domain": "multi_hop_qa",
        "hf_split": "validation", "hf_config": "distractor", "extract": _hotpot_extract,
        "fallback_templates": [
            ("Which university did the founder of {company}, who was born in {city}, attend?",
             "This requires linking a company's founder biography with their education record."),
            ("The author of {book} also wrote a sequel; what year was the sequel published, "
             "and where was the author's hometown of {city} located?",
             "Cross-document reasoning across bibliography and geography records is required."),
            ("Is the actor who played the lead in {movie} the same person married to {name}?",
             "Answering requires linking cast records with public biographical databases."),
        ],
        "fallback_kwargs": lambda rng, i: dict(company=f"Company{i}", city=f"City{i}", book=f"Book{i}",
                                                movie=f"Movie{i}", name=f"Person{i}"),
    },
    "nq": {
        "source_dataset": "natural_questions", "domain": "general_qa",
        "hf_split": "train", "hf_config": None, "extract": _nq_extract,
        "fallback_templates": [
            ("What is the capital of {country}?", "The capital is a well-known public fact."),
            ("How tall is {landmark}?", "Height information about the landmark is public knowledge."),
            ("When was {event} held?", "The event date is a matter of public historical record."),
            ("Who wrote {book}?", "Authorship is a publicly documented fact."),
        ],
        "fallback_kwargs": lambda rng, i: dict(country=f"Country{i}", landmark=f"Landmark{i}",
                                                event=f"Event{i}", book=f"Book{i}"),
    },
}


class LocalDatasetCache:
    """Local JSONL cache of extracted (query, answer) rows, one file per
    dataset key under `data_cache.root`. Decoupled from
    `max_samples_per_dataset`: whatever is downloaded gets cached in full,
    so a later run that asks for the same or fewer rows never touches the
    network again.
    """

    def __init__(
        self,
        root: Path,
        enabled: bool = True,
        prefer_local: bool = True,
        download_if_missing: bool = True,
        local_datasets: Optional[Dict[str, str]] = None,
    ):
        self.root = Path(root)
        self.enabled = enabled
        self.prefer_local = prefer_local
        self.download_if_missing = download_if_missing
        self.local_datasets = local_datasets or {}

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset_key: str) -> Path:
        return self.root / f"{dataset_key}.jsonl"

    def read(self, dataset_key: str) -> Optional[List[Dict[str, Any]]]:
        path = self._path(dataset_key)
        if not (self.enabled and self.prefer_local and path.exists()):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Local cache for '%s' unreadable (%s); ignoring cache.", dataset_key, exc)
            return None

    def write(self, dataset_key: str, rows: List[Dict[str, Any]]) -> None:
        if not (self.enabled and self.download_if_missing):
            return
        path = self._path(dataset_key)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        logger.info("Cached %d rows for '%s' at %s", len(rows), dataset_key, path)

    @classmethod
    def from_config(cls, cfg) -> "LocalDatasetCache":
        cache_cfg = cfg.raw.get("data_cache", {})

        root = Path(cache_cfg.get("root", "data/raw"))

        # Relative roots remain project-relative.
        # Absolute Windows/Linux paths are used directly.
        if not root.is_absolute():
            root = cfg.path.parent.parent / root

        return cls(
            root=root,
            enabled=cache_cfg.get("enabled", True),
            prefer_local=cache_cfg.get("prefer_local", True),
            download_if_missing=cache_cfg.get("download_if_missing", True),
            local_datasets=cache_cfg.get("local_datasets", {}),
        )


class DatasetLoader:
    """Table-driven loader adapter shared by all four Review 1 datasets."""

    # Minimum number of rows fetched (and cached) on first download,
    # independent of how many the current run actually processes via
    # max_samples_per_dataset -- keeps the cache useful across runs that
    # request fewer records than a previous run.
    CACHE_MIN_ROWS = 200

    def __init__(self, dataset_key: str, dataset_cfg: Dict[str, Any], max_samples: int, seed: int,
                 cache: Optional[LocalDatasetCache] = None):
        self.dataset_key = dataset_key
        self.spec = DATASET_SPECS[dataset_key]
        self.dataset_cfg = dataset_cfg
        self.max_samples = max_samples
        self.seed = seed
        self.cache = cache
        self.source_dataset = self.spec["source_dataset"]
        self.domain = self.spec["domain"]
        self.hf_split = dataset_cfg.get("split") or self.spec["hf_split"]
        self.hf_config = dataset_cfg.get("hf_config") or self.spec["hf_config"]

    def load(self) -> List[RawRecord]:
        """Load records using local HF data first, then cache, remote HF, fallback."""

        # ---------------------------------------------------------
        # 1. Prefer locally saved HuggingFace dataset
        # ---------------------------------------------------------
        if self.cache is not None and self.cache.prefer_local:
            try:
                fetch_n = self.max_samples

                rows = self._fetch_rows_from_local_disk(fetch_n)

                if rows:
                    logger.info(
                        "%s: loaded %d real rows from local HuggingFace dataset",
                        self.source_dataset,
                        len(rows),
                    )

                    return self._rows_to_records(
                        rows[: self.max_samples],
                        "real_local",
                    )

            except Exception as exc:
                logger.info(
                    "%s: local dataset unavailable (%s); checking JSONL cache",
                    self.source_dataset,
                    exc,
                )

        # ---------------------------------------------------------
        # 2. Existing JSONL cache
        # ---------------------------------------------------------
        if self.cache is not None:
            cached_rows = self.cache.read(self.dataset_key)

            if cached_rows is not None and len(cached_rows) >= self.max_samples:
                logger.info(
                    "%s: reusing %d cached rows from %s",
                    self.source_dataset,
                    len(cached_rows),
                    self.cache.root,
                )

                return self._rows_to_records(
                    cached_rows[: self.max_samples],
                    "real_cached",
                )

        # ---------------------------------------------------------
        # 3. Existing remote HuggingFace fallback
        # ---------------------------------------------------------
        try:
            fetch_n = max(self.max_samples, self.CACHE_MIN_ROWS)

            rows = self._fetch_rows_from_hf(fetch_n)

            if rows:
                logger.info(
                    "%s: downloaded %d real rows via HF",
                    self.source_dataset,
                    len(rows),
                )

                if self.cache is not None:
                    self.cache.write(self.dataset_key, rows)

                return self._rows_to_records(
                    rows[: self.max_samples],
                    "real",
                )

        except Exception as exc:
            logger.warning(
                "%s: HF load failed (%s); using synthetic fallback",
                self.source_dataset,
                exc,
            )

        # ---------------------------------------------------------
        # 4. Existing synthetic fallback
        # ---------------------------------------------------------
        return self._load_synthetic_fallback()

    def _fetch_rows_from_hf(self, n: int) -> List[Dict[str, Any]]:
        """Stream up to `n` rows from HF and extract (query, answer) pairs."""
        from datasets import load_dataset

        hf_id = self.dataset_cfg["hf_id"]
        ds = (load_dataset(hf_id, self.hf_config, split=self.hf_split, streaming=True)
              if self.hf_config else load_dataset(hf_id, split=self.hf_split, streaming=True))

        extract: Callable = self.spec["extract"]
        rows: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            query, answer = extract(row)
            rows.append({"row_index": i, "query": str(query), "answer": str(answer)})
        return rows
    def _fetch_rows_from_local_disk(self, n: int) -> List[Dict[str, Any]]:
        from datasets import load_from_disk
        local_mapping = self.cache.local_datasets if self.cache is not None else {}

        local_name = local_mapping.get(self.dataset_key)
        if not local_name:
            raise FileNotFoundError(
                f"No local dataset mapping configured for '{self.dataset_key}'."
            )

        dataset_path = self.cache.root / local_name

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Local dataset not found for '{self.dataset_key}': {dataset_path}"
            )

        logger.info(
            "%s: loading local HuggingFace dataset from %s",
            self.source_dataset,
            dataset_path,
        )

        ds = load_from_disk(str(dataset_path))

        # save_to_disk() may produce either a Dataset or DatasetDict.
        if hasattr(ds, "keys"):
            if self.hf_split not in ds:
                raise KeyError(
                    f"Split '{self.hf_split}' not found in local dataset "
                    f"'{dataset_path}'. Available splits: {list(ds.keys())}"
                )
            ds = ds[self.hf_split]

        n = min(n, len(ds))

        # Dataset.select() creates a lightweight view over the requested rows.
        ds = ds.select(range(n))

        extract: Callable = self.spec["extract"]
        rows: List[Dict[str, Any]] = []

        for i, row in enumerate(ds):
            query, answer = extract(row)
            rows.append(
                {
                    "row_index": i,
                    "query": str(query),
                    "answer": str(answer),
                }
            )

        return rows
    def _rows_to_records(self, rows: List[Dict[str, Any]], source_mode: str) -> List[RawRecord]:
        return [
            RawRecord(
                source_dataset=self.source_dataset, domain=self.domain,
                query_text=row["query"], context_text=row["answer"],
                metadata={"row_index": row["row_index"], "source_mode": source_mode},
            )
            for row in rows
        ]

    def _load_synthetic_fallback(self) -> List[RawRecord]:
        rng = random.Random(self.seed + (hash(self.dataset_key) % 1000))
        templates = self.spec["fallback_templates"]
        kwargs_fn = self.spec["fallback_kwargs"]
        out = []
        for i in range(self.max_samples):
            q, a = templates[i % len(templates)]
            q = q.format(**kwargs_fn(rng, i))
            out.append(RawRecord(
                source_dataset=self.source_dataset, domain=self.domain,
                query_text=q, context_text=a,
                metadata={"row_index": i, "source_mode": "synthetic_fallback"},
            ))
        return out


def _make_loader(key: str):
    return lambda dataset_cfg, max_samples, seed, cache=None: DatasetLoader(
        key, dataset_cfg, max_samples, seed, cache=cache)


LOADER_REGISTRY: Dict[str, Callable[..., DatasetLoader]] = {key: _make_loader(key) for key in DATASET_SPECS}


def load_all_datasets(cfg) -> Dict[str, List[RawRecord]]:
    """Load raw records for every configured dataset key -> {key: [RawRecord]}."""
    cache = LocalDatasetCache.from_config(cfg)
    results: Dict[str, List[RawRecord]] = {}
    for key in DATASET_SPECS:
        loader = DatasetLoader(key, cfg.datasets[key], cfg.max_samples[key], cfg.seed, cache=cache)
        results[key] = loader.load()
    return results