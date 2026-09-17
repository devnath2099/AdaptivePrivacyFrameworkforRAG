"""Contextual embedding evidence: epsilon.

Uses a lightweight pretrained SentenceTransformer model (configurable).
If the model cannot be loaded (no internet / package missing), a
deterministic hashing-based pseudo-embedding fallback is used so the
pipeline stays runnable, and this is logged clearly as a fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import List

import numpy as np

from .schemas import UnifiedRecord

logger = logging.getLogger(__name__)

DEFAULT_DIM = 384  # matches all-MiniLM-L6-v2 output dimension


class EmbeddingEvidenceExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 16, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dim = DEFAULT_DIM
        self._try_load()

    def _try_load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SentenceTransformer model '%s' unavailable (%s); "
                "using deterministic hashing-based pseudo-embedding fallback.",
                self.model_name, exc,
            )
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False))
        return np.stack([self._hashing_pseudo_embedding(t) for t in texts])

    def _hashing_pseudo_embedding(self, text: str) -> np.ndarray:
        """Deterministic, seed-free pseudo-embedding via repeated hashing.

        NOT a semantic embedding; purely a structural placeholder that
        keeps downstream shape/consistency checks meaningful when the
        real model cannot be loaded.
        """
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split() or [""]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def build_embeddings_all(records: List[UnifiedRecord], model_name: str,
                          batch_size: int, device: str,
                          checkpoint_dir: str = "outputs/cache",
                          partition_id: int = 0,
                          num_partitions: int = 1) -> None:
    """Attach embedding evidence (in-place) to every record's `.evidence.embedding`.

    Args:
        checkpoint_dir: Directory for checkpoint markers.
        partition_id: This notebook's partition index (0-based).
        num_partitions: Total number of partitions.
    """
    import os

    from tqdm import tqdm

    os.makedirs(checkpoint_dir, exist_ok=True)
    done_path = os.path.join(checkpoint_dir, f"embeddings_done_{partition_id}.marker")

    if os.path.exists(done_path):
        print(f"[Partition {partition_id}] Embeddings checkpoint exists, skipping")
        return

    extractor = EmbeddingEvidenceExtractor(model_name, batch_size, device)

    # Apply partition filtering
    if num_partitions > 1:
        records = records[partition_id::num_partitions]
        print(f"[Partition {partition_id}] Embedding {len(records)} records")

    texts = [r.normalized_text for r in records]
    if not texts:
        return
    vectors = extractor.encode(texts)
    for record, vec in tqdm(zip(records, vectors), total=len(records), desc=f"Embeddings P{partition_id}"):
        if record.evidence is None:
            continue
        record.evidence.embedding = vec.tolist()

    with open(done_path, "w") as f:
        json.dump({"n_records": len(records), "complete": True}, f)
    print(f"[Partition {partition_id}] Embeddings complete for {len(records)} records")
