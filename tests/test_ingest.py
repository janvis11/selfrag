"""Tests for knowledge-base ingestion."""

import json
import pytest
from pathlib import Path

from app.rag.ingest import build_index, _chunk_text, _read_kb_files
from app.core.config import settings


def test_read_kb_files():
    """KB directory should contain at least 6 markdown files."""
    docs = _read_kb_files()
    assert len(docs) >= 6
    for doc in docs:
        assert doc["doc"].endswith(".md")
        assert len(doc["text"]) > 100


def test_chunk_text_produces_chunks():
    sample = "This is a test sentence. " * 200
    chunks = _chunk_text(sample, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Each chunk should have content
    for c in chunks:
        assert len(c.strip()) > 0


def test_chunk_text_short_text():
    """Text shorter than max_tokens should produce exactly 1 chunk."""
    short = "Hello, this is a short sentence."
    chunks = _chunk_text(short, max_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1


def test_build_index_creates_files(tmp_path: Path):
    """build_index should create index.faiss and metadata.json."""
    # Override FAISS_DIR to temp
    original_dir = settings.FAISS_DIR
    settings.FAISS_DIR = str(tmp_path / "faiss_test")

    try:
        result = build_index()
        assert result["documents_indexed"] >= 6
        assert result["total_chunks"] > 0

        idx_path = Path(settings.FAISS_DIR) / "index.faiss"
        meta_path = Path(settings.FAISS_DIR) / "metadata.json"
        assert idx_path.exists()
        assert meta_path.exists()

        # Metadata should be valid JSON list
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(meta, list)
        assert len(meta) == result["total_chunks"]
        for chunk in meta:
            assert "doc" in chunk
            assert "chunk_id" in chunk
            assert "text" in chunk
    finally:
        settings.FAISS_DIR = original_dir
