# DocMind — Architecture & Design Decisions

This document captures the architecture, design decisions, and trade-offs behind DocMind.
It is intended as a reference for contributors and as a record of *why* the system looks the way it does.

---

## 1. Problem Statement

Existing LLM-based document Q&A systems fail on three fronts:

1. **Hallucinations** — LLMs invent facts that are not present in source documents.
2. **Single-shot reasoning** — Complex queries (compare, multi-step) require decomposition, not a single prompt.
3. **No quality control** — Answers are returned without validation or grounding checks.

DocMind addresses these via **grounded retrieval (RAG)**, **multi-agent decomposition**, and **automated LLM-as-judge evaluation**.

---

## 2. High-Level Design

DocMind implements the **Supervisor multi-agent pattern**:

- A central **Supervisor** decides workflow at each step.
- **Specialised agents** handle retrieval, summarisation, comparison, and evaluation.
- A **shared state** passes between agents via LangGraph.
- Every answer is **grounded in retrieved source chunks** with explicit citations.

```text
                   User Query
                        ↓
              ┌─────────────────┐
              │   SUPERVISOR    │  ← LangGraph orchestrator
              └────────┬────────┘
                       ↓
   ┌───────────┬───────┴───────┬───────────┐
   ↓           ↓               ↓           ↓
┌─────────┐ ┌─────────┐    ┌──────────┐ ┌─────────┐
│Retriever│ │Summarise│    │Comparator│ │Evaluator│
└────┬────┘ └─────────┘    └──────────┘ └─────────┘
     ↓
┌─────────┐
│  FAISS  │  ← Vector store of document chunks
└─────────┘
     ↓
Grounded Answer
with Citations
```

---

## 3. Component Breakdown

### 3.1 Ingestion Pipeline (`backend/core/ingestion.py`)

- **Loaders:** PDF, DOCX, TXT via LangChain document loaders.
- **Splitter:** `RecursiveCharacterTextSplitter` with `chunk_size=512`, `overlap=50`.
- **Rationale:** Recursive splitter preserves semantic boundaries (paragraphs → sentences → words) better than fixed-size splitting.

### 3.2 Embedding & Storage (`backend/core/embeddings.py`, `vector_store.py`)

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- **Vector store:** FAISS with `IndexFlatL2` (exact search).
- **Rationale:** Free, fully local, no external dependency for the MVP. Can swap to Chroma or Pinecone later without architectural change.

### 3.3 Agent Layer (`backend/agents/`)

| Agent | Role | Model / Tools |
| --- | --- | --- |
| **Supervisor** | Decides next action based on state | LLM with routing prompt |
| **Retriever** | Semantic search over vector store | FAISS + embeddings |
| **Summariser** | Condenses retrieved chunks | Fine-tuned LoRA model |
| **Comparator** | Cross-document comparison | LLM with comparison prompt |
| **Evaluator** | LLM-as-judge quality validation | LLM with evaluation rubric |

### 3.4 Orchestration (`backend/core/graph.py`)

- Built with **LangGraph** as a stateful graph.
- Conditional edges allow loops (e.g. Evaluator can send the workflow back to Retriever if grounding fails).
- All node calls traced via **LangFuse**.

### 3.5 Fine-Tuning (`fine_tuning/`)

- **Base model:** Llama-3.2-1B (small enough to train on free Colab GPU).
- **Method:** LoRA (`r=8, alpha=16`) via Hugging Face PEFT.
- **Dataset:** ~500 document → summary instruction pairs.
- **Training:** Google Colab T4 GPU, 1–2 epochs.
- **Rationale:** Demonstrates real fine-tuning workflow without requiring expensive infrastructure.

### 3.6 Evaluation

- **Retrieval metrics:** precision@k and recall@k on a labelled test set.
- **Answer metrics:** LLM-as-judge scoring for groundedness, completeness, and citation accuracy.
- **Latency:** p50/p95 per agent and end-to-end.

---

## 4. Key Design Decisions

| Decision | Choice | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Orchestration | **LangGraph** | LangChain chains, AutoGen, CrewAI | Needs state, loops, and conditional routing — chains alone cannot do this cleanly. |
| Vector DB | **FAISS** | Chroma, Pinecone, Weaviate | Zero-setup, fully local, no API costs. Easy to swap out later. |
| Embedding model | **`all-MiniLM-L6-v2`** | OpenAI `text-embedding-3-small`, Cohere | Free, local, strong quality for general text. Removes external API dependency. |
| LLM provider | **Groq (Llama 3.1 8B)** | OpenAI, Anthropic | Free tier with very fast inference (~500 tokens/sec). |
| Fine-tuning | **LoRA on Llama-3.2-1B** | Full fine-tune, prompt-only | LoRA + small base model is trainable on free GPU and proves PEFT skill. |
| Backend | **FastAPI** | Flask, Django | Native async, automatic OpenAPI docs, modern Python type hints. |
| Frontend | **React + TypeScript** | Streamlit, plain HTML | Closer to what production systems use; better portfolio signal. |
| Observability | **LangFuse** | Custom logging, Langsmith | Free tier, purpose-built for LLM tracing. |

---

## 5. State Schema

The LangGraph state object passes information between agents:

```python
class DocMindState(TypedDict):
    question: str                       # Original user query
    retrieved_chunks: list[dict]        # Output of Retriever
    summaries: dict[str, str]           # Per-document summaries from Summariser
    draft_answer: str                   # Output of Comparator
    evaluation: dict                    # Evaluator's grounding/quality scores
    citations: list[dict]               # Source references with page numbers
    next_agent: str                     # Routing decision made by Supervisor
    iteration: int                      # Retry counter (max 2)
```

Each agent reads and writes only the keys relevant to its role.

---

## 6. Failure Modes & Mitigations

| Failure Mode | Mitigation |
| --- | --- |
| Retrieval misses relevant chunks | Hybrid search (semantic + keyword) — planned for v2. |
| Hallucinated citations | Evaluator verifies every citation against the source chunk text. |
| LLM API failure | Retry with exponential backoff; fallback to local Llama via Ollama. |
| Fine-tuned model underperforms base | A/B benchmark in `fine_tuning/`; revert to base model if regressed. |
| Infinite agent loops | Hard iteration cap (`max_iterations=2`) on Evaluator → Retriever loop. |

---

## 7. Non-Functional Requirements

| Requirement | Target |
| --- | --- |
| End-to-end query latency (p95) | < 8 seconds |
| Retrieval precision@5 | ≥ 0.7 on labelled set |
| Citation accuracy | ≥ 0.9 |
| Test coverage | ≥ 70% of `backend/` |
| Dockerised cold start | < 30 seconds |

---

## 8. Out of Scope (v1)

The following are intentionally excluded from the v1 build to keep scope realistic:

- Multi-tenant authentication and authorisation.
- Real-time collaborative editing.
- Non-English documents.
- Image and table extraction from PDFs.
- Streaming token-level responses.

These are candidates for v2 and are tracked in [README → Roadmap](./README.md#roadmap).

---

## 9. References

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain RAG cookbook](https://python.langchain.com/docs/use_cases/question_answering/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [LangFuse docs](https://langfuse.com/docs)
