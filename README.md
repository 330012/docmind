# DocMind

> **Multi-agent GenAI document intelligence platform with RAG, LangGraph orchestration, and a fine-tuned foundation model.**

DocMind turns any collection of documents into an interactive knowledge platform. It uses a supervised multi-agent system to retrieve, summarise, compare, and evaluate information across thousands of pages — grounded in source citations and self-checked for quality.

> 🚧 **Status:** Active development — see [Roadmap](#roadmap) for progress.

---

## ✨ Features

- 🧠 **Multi-agent reasoning** — Supervisor coordinates specialised agents (Retriever, Summariser, Comparator, Evaluator) via LangGraph
- 🔍 **Production RAG pipeline** — FAISS vector store, semantic chunking, citation-grounded answers
- 🎯 **Fine-tuned foundation model** — LoRA/PEFT on a base LLM for domain-specific summarisation
- 📊 **LLM-as-judge evaluation** — Automated quality scoring of every answer
- 🛰️ **Full observability** — LangFuse tracing of every agent call, prompt, and token
- 🐳 **Production deployment** — Dockerised, CI/CD via GitHub Actions
- 🧪 **Tested** — pytest suite covering ingestion, retrieval, and agent workflows

---

## 🏗️ Architecture

DocMind follows the **Supervisor multi-agent pattern**:

```text
User Query
    ↓
Supervisor (LangGraph) — decides workflow
    ↓
┌────────────┬────────────┬────────────┐
Retriever  Summariser  Comparator  Evaluator
    ↓            ↓            ↓            ↓
  FAISS      LLM API     LLM API   LLM-as-Judge
    ↓
Grounded Answer with Citations
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for detailed design decisions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM orchestration | LangChain, LangGraph |
| Foundation model | Llama 3.1 8B (Groq) + LoRA-tuned Llama-3.2-1B |
| Fine-tuning | PyTorch, Hugging Face PEFT |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS |
| Backend | FastAPI, Pydantic, Python 3.11 |
| Frontend | React, TypeScript, Tailwind CSS |
| Observability | LangFuse |
| Testing | pytest |
| Containerisation | Docker, docker-compose |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- An LLM API key (Groq, OpenAI, or Anthropic)

### Local setup

```bash
git clone https://github.com/330012/docmind.git
cd docmind
python -m venv venv
source venv/bin/activate            # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # Add your API keys
uvicorn backend.main:app --reload
```

Visit `http://localhost:8000/docs` for the API documentation.

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=backend
```

---

## 🗺️ Roadmap

- [ ] Document ingestion (PDF, DOCX) with semantic chunking
- [ ] Embedding pipeline with sentence-transformers
- [ ] FAISS vector store integration
- [ ] Basic RAG retrieval chain
- [ ] FastAPI backend with upload + query endpoints
- [ ] LangGraph supervisor and four specialist agents
- [ ] LoRA fine-tuning of base model
- [ ] LLM-as-judge evaluation harness
- [ ] React frontend
- [ ] Docker + GitHub Actions CI/CD
- [ ] Live demo deployment

---

## 🎓 About

Built by [Gaurav Khunt](https://www.linkedin.com/in/gauravkhunt) as a portfolio project demonstrating production-grade multi-agent GenAI engineering for AI internship applications in Germany.

## 📜 License

MIT — see [LICENSE](./LICENSE) for details.
