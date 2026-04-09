"""Grounding critic — checks whether a draft answer is supported by retrieved chunks."""

from __future__ import annotations

import logging
import re

from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── Helpers ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── Grounding check ─────────────────────────────────────────────

def check_grounding(
    draft: str,
    retrieved_texts: list[str],
    threshold: float = 0.45,
) -> dict:
    """Check whether each sentence in *draft* is grounded in *retrieved_texts*.

    Returns a dict with keys:
    - ``is_grounded``: True if all sentences pass
    - ``unsupported``: list of sentences below threshold
    - ``scores``: list of (sentence, score) tuples
    """
    if not retrieved_texts:
        return {"is_grounded": False, "unsupported": [draft], "scores": []}

    model = _get_model()
    sentences = _split_sentences(draft)
    if not sentences:
        return {"is_grounded": True, "unsupported": [], "scores": []}

    # Encode
    sent_embs = model.encode(sentences, normalize_embeddings=True)
    chunk_embs = model.encode(retrieved_texts, normalize_embeddings=True)

    unsupported: list[str] = []
    scores: list[tuple[str, float]] = []

    for idx, sent_emb in enumerate(sent_embs):
        # cosine similarity (vectors already normalised)
        sims = np.dot(chunk_embs, sent_emb)
        best_score = float(np.max(sims))
        scores.append((sentences[idx], best_score))
        if best_score < threshold:
            unsupported.append(sentences[idx])

    return {
        "is_grounded": len(unsupported) == 0,
        "unsupported": unsupported,
        "scores": scores,
    }
