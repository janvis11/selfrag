"""Tests for the vector retriever."""

import json
import pytest
from pathlib import Path

from app.rag.ingest import build_index
from app.rag.retriever import VectorRetriever
from app.core.config import settings


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory) -> Path:
    """Build a FAISS index in a temp dir for the test module."""
    d = tmp_path_factory.mktemp("faiss_retriever_test")
    original = settings.FAISS_DIR
    settings.FAISS_DIR = str(d)
    build_index()
    settings.FAISS_DIR = original
    return d


def test_retriever_loads(index_dir: Path):
    retriever = VectorRetriever(index_dir=index_dir)
    assert retriever.is_ready


def test_retriever_returns_chunks(index_dir: Path):
    retriever = VectorRetriever(index_dir=index_dir)
    results = retriever.search("What documents are required for a personal loan?", top_k=5)
    assert len(results) > 0
    for chunk in results:
        assert chunk.score > 0
        assert len(chunk.text) > 0
        assert chunk.doc.endswith(".md")


def test_retriever_respects_top_k(index_dir: Path):
    retriever = VectorRetriever(index_dir=index_dir)
    results = retriever.search("processing fees", top_k=3)
    assert len(results) <= 3


def test_retriever_empty_index():
    """Retriever with a non-existent index should return empty results."""
    retriever = VectorRetriever(index_dir=Path("nonexistent_dir_12345"))
    assert not retriever.is_ready
    results = retriever.search("anything")
    assert results == []


def test_retriever_relevant_docs(index_dir: Path):
    """Search for fees-related query should return chunks from fees_charges.md."""
    retriever = VectorRetriever(index_dir=index_dir)
    results = retriever.search("What are the processing fees and charges?", top_k=5)
    docs_found = {r.doc for r in results}
    assert "fees_charges.md" in docs_found
