"""RAG API routes — /api/rag/* endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.rag.schemas import RAGQueryRequest, RAGQueryResponse, ReindexResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base() -> ReindexResponse:
    """Rebuild embeddings and refresh the vector store from ``app/knowledge_base/``."""
    from app.rag import ingest, pipeline
    try:
        result = ingest.build_index()
        # Reload the in-memory retriever so subsequent queries use the new index
        pipeline.reload_retriever()
        return ReindexResponse(
            status="ok",
            documents_indexed=result["documents_indexed"],
            total_chunks=result["total_chunks"],
            message="Knowledge base reindexed successfully.",
        )
    except Exception as exc:
        logger.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """Run the Self-RAG pipeline: gate → retrieve → draft → critique → revise."""
    from app.rag import pipeline
    try:
        return pipeline.run(request)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
