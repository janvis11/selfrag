"""Knowledge-base ingestion: read markdown → chunk → embed → build FAISS index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Embedding model (loaded once) ────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── Token-based chunking ─────────────────────────────────────────

def _tokenize_count(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _chunk_text(
    text: str,
    max_tokens: int = settings.CHUNK_TOKENS,
    overlap_tokens: int = settings.CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks of roughly *max_tokens* tokens."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap_tokens
    return chunks


# ── Read knowledge base ──────────────────────────────────────────

def _read_kb_files(kb_dir: Path | None = None) -> list[dict]:
    """Return a list of ``{doc, text}`` dicts from markdown files."""
    kb_dir = kb_dir or settings.knowledge_base_dir
    docs: list[dict] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if content.strip():
            docs.append({"doc": md_file.name, "text": content})
    return docs


# ── Build index ──────────────────────────────────────────────────

def build_index(kb_dir: Path | None = None) -> dict:
    """Build (or rebuild) the FAISS index from knowledge-base markdown files.

    Returns summary dict with ``documents_indexed`` and ``total_chunks``.
    """
    docs = _read_kb_files(kb_dir)
    if not docs:
        raise RuntimeError("No markdown files found in knowledge_base directory.")

    model = _get_model()
    all_chunks: list[dict] = []

    for doc in docs:
        raw_chunks = _chunk_text(doc["text"])
        for idx, chunk_text in enumerate(raw_chunks):
            all_chunks.append(
                {
                    "doc": doc["doc"],
                    "chunk_id": f"{doc['doc'].removesuffix('.md')}#{idx}",
                    "text": chunk_text,
                }
            )

    logger.info("Embedding %d chunks …", len(all_chunks))
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (inner-product on normalised vectors ≈ cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    out_dir = settings.faiss_dir_path
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "Index built: %d docs, %d chunks, dim=%d",
        len(docs),
        len(all_chunks),
        dim,
    )
    return {"documents_indexed": len(docs), "total_chunks": len(all_chunks)}
