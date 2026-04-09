# self-rag module - tata capital bfsi chatbot

self-rag (self-reflective retrieval-augmented generation) module for loan faqs, policy explanations, fees/documentation info, and sales objection handling.

> **scope rule**: self-rag is used **only** for explanations (faq / sales / policy). it must **not** be used for underwriting decision logic (credit score thresholds, emi math, approval/rejection rules).

## what this module does

when a user asks a policy/faq/sales question — e.g. *"what documents are required?"*, *"is foreclosure allowed?"*, *"what are the charges?"* — the module:

1. **gates** — decides whether retrieval is necessary (intent classification)
2. **retrieves** — pulls the most relevant chunks from the approved knowledge base via faiss
3. **drafts** — generates a response using groq llm grounded on retrieved content
4. **self-critiques** — checks whether the response is grounded (no unsupported claims)
5. **revises** — removes unsupported claims, adds mandatory disclaimers
6. **falls back** safely if retrieval confidence is low

## tech stack

| layer | technology |
|-------|-----------|
| runtime | python 3.11+, fastapi, uvicorn |
| llm | groq api (llama 3.3 70b) |
| embeddings | sentence-transformers (`all-minilm-l6-v2`) |
| vector store | faiss (local) |
| schemas | pydantic v2 |
| testing | pytest |

## project structure

```
self_rag/
├── app/
│   ├── main.py                 # fastapi entry
│   ├── core/
│   │   └── config.py           # settings (from .env)
│   ├── rag/
│   │   ├── schemas.py          # request/response models
│   │   ├── ingest.py           # chunk + embed + build faiss index
│   │   ├── retriever.py        # vector search abstraction
│   │   ├── gate.py             # intent classifier (retrieval gate)
│   │   ├── critic.py           # grounding checker
│   │   └── pipeline.py         # full self-rag pipeline orchestrator
│   ├── knowledge_base/         # approved kb content (6 markdown files)
│   └── api/
│       └── routes_rag.py       # /api/rag/* endpoints
├── tests/                      # test suite
├── data/faiss_index/           # faiss artifacts (generated)
├── .env                        # environment config
├── requirements.txt
└── README.md
```

## quick start

### 1. install dependencies

```bash
python -m venv .venv
# windows
.venv\Scripts\activate
# linux/mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. configure

edit `.env` and set your groq api key:

```
GROQ_API_KEY=your-actual-groq-api-key
```

### 3. run the server

```bash
uvicorn app.main:app --reload --port 8001
```

### 4. index the knowledge base

```bash
curl -X POST http://localhost:8001/api/rag/reindex
```

### 5. query

```bash
curl -X POST http://localhost:8001/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "user_query": "what documents do i need for a personal loan?",
    "context": {"product": "personal_loan"}
  }'
```

**response:**
```json
{
  "intent": "docs",
  "used_retrieval": true,
  "answer": "…grounded response…",
  "disclaimers": ["subject to verification and credit checks.", "…"],
  "citations": [{"doc": "eligibility_docs.md", "chunk_id": "eligibility_docs#3"}],
  "debug": {
    "retrieved_chunks": 5,
    "retrieval_confidence": "high",
    "grounding_pass": true,
    "gate_intent": "docs",
    "gate_should_retrieve": true
  }
}
```

## api endpoints

| method | path | description |
|--------|------|-------------|
| `GET` | `/health` | health check |
| `POST` | `/api/rag/reindex` | rebuild faiss index from kb |
| `POST` | `/api/rag/query` | run self-rag pipeline |

## configuration (.env)

| variable | default | description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | your groq api key (required) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | groq model name |
| `VECTOR_STORE` | `faiss` | vector store backend |
| `CHUNK_TOKENS` | `500` | token chunk size |
| `CHUNK_OVERLAP` | `80` | overlap between chunks |
| `TOP_K` | `5` | number of chunks to retrieve |
| `MIN_SCORE` | `0.35` | minimum similarity threshold |
| `ENABLE_GROUNDING_CHECK` | `true` | enable grounding critic |
| `ENABLE_FALLBACK` | `true` | enable safe fallback |

## testing

```bash
pytest tests/ -q
```

