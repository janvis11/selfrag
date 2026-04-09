"""Vector retriever — loads FAISS index and returns scored chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.schemas import Chunk

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class VectorRetriever:
    """Thin wrapper around a FAISS index + metadata for semantic search."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self._index_dir = index_dir or settings.faiss_dir_path
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []
        self._load()

    # ── internal ─────────────────────────────────────────────────

    def _load(self) -> None:
        index_path = self._index_dir / "index.faiss"
        meta_path = self._index_dir / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                "FAISS index not found at %s — run /api/rag/reindex first.",
                self._index_dir,
            )
            return
        self._index = faiss.read_index(str(index_path))
        self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded FAISS index with %d vectors.", self._index.ntotal
        )

    def reload(self) -> None:
        """Re-read the persisted index (e.g. after a reindex)."""
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    # ── public API ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[Chunk]:
        """Return the *top_k* most similar chunks above *min_score*."""
        if not self.is_ready:
            return []

        top_k = top_k or settings.TOP_K
        min_score = min_score if min_score is not None else settings.MIN_SCORE

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, top_k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            meta = self._metadata[idx]
            results.append(
                Chunk(
                    doc=meta["doc"],
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    score=float(score),
                )
            )
        return results
