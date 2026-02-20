"""FastAPI application entry point for the Self-RAG module."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_rag import router as rag_router
from app.core.config import settings

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# ── App factory ──────────────────────────────────────────────────

app = FastAPI(
    title="Self-RAG Module — Tata Capital BFSI Chatbot",
    description=(
        "Self-RAG service for loan FAQs, policy explanations, fees/docs info, "
        "and sales objection handling. Uses retrieval-augmented generation with "
        "grounding checks and mandatory disclaimers."
    ),
    version="1.0.0",
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(rag_router)


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "env": settings.APP_ENV,
        "vector_store": settings.VECTOR_STORE,
    }
