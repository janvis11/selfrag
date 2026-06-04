# self-rag module - tata capital bfsi 

self-rag (self-reflective retrieval-augmented generation) module for loan faqs, policy explanations, fees/documentation info, and sales objection handling.

> **scope rule**: self-rag is used **only** for explanations (faq / sales / policy). it must **not** be used for underwriting decision logic (credit score thresholds, emi math, approval/rejection rules).

## what this module does

when a user asks a policy/faq/sales question - e.g. *"what documents are required?"*, *"is foreclosure allowed?"*, *"what are the charges?"* - the module:

1. **gates** - decides whether retrieval is necessary (intent classification)
2. **retrieves** - pulls the most relevant chunks from the approved knowledge base via faiss
3. **drafts** - generates a response using groq llm grounded on retrieved content
4. **self-critiques** - checks whether the response is grounded (no unsupported claims)
5. **revises** - removes unsupported claims, adds mandatory disclaimers
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

