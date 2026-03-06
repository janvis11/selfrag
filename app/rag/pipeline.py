"""Self-RAG pipeline orchestrator: gate → retrieve → draft → critique → revise."""

from __future__ import annotations

import logging
from typing import Any

from groq import Groq

from app.core.config import settings
from app.rag import gate, critic
from app.rag.retriever import VectorRetriever
from app.rag.schemas import (
    Chunk,
    Citation,
    DebugInfo,
    RAGQueryRequest,
    RAGQueryResponse,
)

logger = logging.getLogger(__name__)

# ── Singleton retriever (reloaded on /reindex) ───────────────────

_retriever: VectorRetriever | None = None


def get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever


def reload_retriever() -> None:
    global _retriever
    _retriever = VectorRetriever()


# ── Disclaimers ──────────────────────────────────────────────────

_STANDARD_DISCLAIMERS: list[str] = [
    "Subject to verification and credit checks.",
    "Final terms may vary based on individual eligibility and applicable policies.",
]


# ── Groq LLM helpers ─────────────────────────────────────────────

def _groq_client() -> Groq:
    return Groq(api_key=settings.GROQ_API_KEY)


def _llm_draft(query: str, chunks: list[Chunk], context: dict[str, Any]) -> str:
    """Use Groq LLM to draft an answer grounded on retrieved chunks."""
    passages = "\n\n---\n\n".join(
        f"[Source: {c.doc} | Chunk: {c.chunk_id}]\n{c.text}" for c in chunks
    )

    ctx_str = ""
    if context:
        ctx_str = "\nCustomer context: " + ", ".join(
            f"{k}={v}" for k, v in context.items()
        )

    system_prompt = (
        "You are a helpful, professional customer support assistant for Tata Capital, "
        "a Non-Banking Financial Company (NBFC) in India.\n\n"
        "RULES:\n"
        "1. Answer ONLY based on the retrieved passages below. Do NOT invent information.\n"
        "2. NEVER promise loan approval — it is always subject to credit assessment.\n"
        "3. NEVER fabricate rates, fees, or charges not mentioned in the passages.\n"
        "4. If the passages do not contain enough information, say so honestly.\n"
        "5. Keep your answer clear, structured, and professional.\n"
        "6. Where appropriate, include relevant disclaimers.\n"
        "7. Do NOT include any underwriting decision logic (credit score thresholds, "
        "approval/rejection rules, EMI math).\n"
    )

    user_prompt = (
        f"Customer question: {query}{ctx_str}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Provide a helpful, grounded answer based on the above passages."
    )

    client = _groq_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _llm_revise(draft: str, unsupported: list[str], chunks: list[Chunk]) -> str:
    """Ask the LLM to remove unsupported claims and produce a revised answer."""
    passages = "\n\n---\n\n".join(
        f"[{c.doc}]\n{c.text}" for c in chunks
    )

    system_prompt = (
        "You are a compliance reviewer for a financial services company.\n"
        "Review the draft answer and remove or rewrite any sentences that are not "
        "supported by the provided source passages.\n"
        "Keep all sentences that ARE supported. Do NOT add new information.\n"
        "Ensure the revised answer includes appropriate disclaimers.\n"
    )

    user_prompt = (
        f"Draft answer:\n{draft}\n\n"
        f"Unsupported sentences (REMOVE or REWRITE these):\n"
        + "\n".join(f"- {s}" for s in unsupported)
        + f"\n\nSource passages:\n{passages}\n\n"
        "Produce the revised, grounded answer."
    )

    client = _groq_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


# ── Retrieval confidence ────────────────────────────────────────

def _assess_confidence(chunks: list[Chunk]) -> str:
    if not chunks:
        return "none"
    avg_score = sum(c.score for c in chunks) / len(chunks)
    if avg_score >= 0.60:
        return "high"
    if avg_score >= 0.45:
        return "medium"
    return "low"


# ── Main pipeline ────────────────────────────────────────────────

def run(request: RAGQueryRequest) -> RAGQueryResponse:
    """Execute the full Self-RAG pipeline and return a response."""

    # 1. Gate — intent classification
    intent, should_retrieve = gate.classify(request.user_query)
    logger.info("Gate: intent=%s, should_retrieve=%s", intent, should_retrieve)

    debug = DebugInfo(
        gate_intent=intent,
        gate_should_retrieve=should_retrieve,
    )

    # If out-of-scope, return early
    if not should_retrieve:
        return RAGQueryResponse(
            intent=intent,
            used_retrieval=False,
            answer=(
                "I'm sorry, this question appears to be outside the scope of "
                "loan-related FAQs, policies, or sales information that I can "
                "assist with. Please contact our customer care for further help."
            ),
            disclaimers=[],
            citations=[],
            debug=debug,
        )

    # 2. Retrieve
    retriever = get_retriever()
    if not retriever.is_ready:
        return RAGQueryResponse(
            intent=intent,
            used_retrieval=False,
            answer=(
                "The knowledge base has not been indexed yet. "
                "Please ask an administrator to run the /api/rag/reindex endpoint first."
            ),
            disclaimers=[],
            citations=[],
            debug=debug,
        )

    chunks = retriever.search(request.user_query)
    confidence = _assess_confidence(chunks)
    debug.retrieved_chunks = len(chunks)
    debug.retrieval_confidence = confidence

    # Fallback if retrieval is weak
    if settings.ENABLE_FALLBACK and (confidence == "none" or confidence == "low"):
        return RAGQueryResponse(
            intent=intent,
            used_retrieval=True,
            answer=(
                "I found limited information to answer your question confidently. "
                "Could you please rephrase your query or provide more details? "
                "Alternatively, you can reach out to our customer care team who "
                "will be happy to assist you with accurate information."
            ),
            disclaimers=_STANDARD_DISCLAIMERS,
            citations=[],
            debug=debug,
        )

    # 3. Draft answer via Groq LLM
    draft = _llm_draft(request.user_query, chunks, request.context)
    logger.info("Draft generated (%d chars)", len(draft))

    # 4. Critique — grounding check
    answer = draft
    if settings.ENABLE_GROUNDING_CHECK:
        grounding = critic.check_grounding(
            draft, [c.text for c in chunks]
        )
        debug.grounding_pass = grounding["is_grounded"]

        if not grounding["is_grounded"]:
            logger.warning(
                "Grounding check failed — %d unsupported sentences, revising…",
                len(grounding["unsupported"]),
            )
            # 5. Revise
            answer = _llm_revise(draft, grounding["unsupported"], chunks)
    else:
        debug.grounding_pass = None

    # Build citations from chunks used
    citations = [
        Citation(doc=c.doc, chunk_id=c.chunk_id) for c in chunks
    ]

    return RAGQueryResponse(
        intent=intent,
        used_retrieval=True,
        answer=answer,
        disclaimers=_STANDARD_DISCLAIMERS,
        citations=citations,
        debug=debug,
    )
