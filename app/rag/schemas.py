"""Pydantic request / response models for the Self-RAG pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    session_id: str = Field(..., description="Unique session / conversation identifier")
    user_query: str = Field(..., min_length=1, description="The user's question")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context (e.g. customer_city, product)",
    )


# ── Chunk / Citation ───────────────────────────────────────────────

class Chunk(BaseModel):
    """A single retrieved text chunk with metadata."""
    doc: str = Field(..., description="Source document filename")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk content")
    score: float = Field(0.0, description="Similarity score")


class Citation(BaseModel):
    doc: str
    chunk_id: str


# ── Debug / internal ───────────────────────────────────────────────

class DebugInfo(BaseModel):
    retrieved_chunks: int = 0
    retrieval_confidence: str = "none"  # none | low | medium | high
    grounding_pass: bool | None = None
    gate_intent: str = "unknown"
    gate_should_retrieve: bool = False


# ── Response ───────────────────────────────────────────────────────

class RAGQueryResponse(BaseModel):
    intent: str
    used_retrieval: bool
    answer: str
    disclaimers: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    debug: DebugInfo = Field(default_factory=DebugInfo)


# ── Reindex ────────────────────────────────────────────────────────

class ReindexResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int = 0
    total_chunks: int = 0
    message: str = ""
