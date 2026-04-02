# 🤖 AI Interviewer

An intelligent, adaptive interview assistant powered by **LangGraph**, **Ollama LLMs**, **Qdrant hybrid search**, and **FastAPI**. It conducts realistic technical interviews by generating personalised questions from a candidate's resume and a curated question bank, evaluating answers in real time, and providing structured feedback.

---

## ✨ Features

- **Adaptive questioning** — starts with RAG-retrieved questions from a curated bank, then switches to LLM-generated questions as the session progresses
- **Resume-aware context** — extracts and embeds resume content so questions are tailored to the candidate
- **Hybrid search (dense + sparse BM25)** — Qdrant RRF fusion for high-quality question retrieval
- **Dynamic difficulty** — supervisor agent adjusts difficulty (`easy / medium / hard`) based on rolling score averages
- **Multi-topic coverage** — tracks covered topics and weak areas to ensure a thorough interview
- **Stateless REST API** — clean FastAPI endpoints; state is passed back by the client so the server is horizontally scalable
- **PDF resume ingestion** — upload a PDF and get text extracted automatically via PyMuPDF

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                    │
│  POST /interview/start   POST /interview/answer             │
│  POST /resume/upload     GET  /health                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  InterviewGraph │  (LangGraph StateGraph)
                    └────────┬────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
  ┌───────▼──────┐  ┌────────▼───────┐  ┌───────▼──────┐
  │  Supervisor  │  │ QuestionAgent  │  │  Evaluator   │
  │    Agent     │  │ (RAG / LLM)    │  │    Agent     │
  └──────────────┘  └────────┬───────┘  └──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
   ┌──────────▼──────────┐     ┌────────────▼────────────┐
   │  QuestionEmbeddings  │     │   ResumeEmbedder         │
   │  (Qdrant hybrid)     │     │   (Qdrant hybrid)        │
   └──────────────────────┘     └─────────────────────────┘
```

### Key components

| File | Role |
|------|------|
| `app/main.py` | FastAPI app, startup hooks, all HTTP routes |
| `agents/main_agent.py` | Builds and compiles the LangGraph `StateGraph` |
| `agents/supervisor_agent.py` | Routing logic — decides next node, adjusts difficulty & topic |
| `agents/question_agent.py` | `RagQuestionAgent` (first 3 Qs) and `LLMQuestionAgent` (thereafter) |
| `agents/evaluator_agent.py` | Scores answers and appends them to history |
| `agents/state.py` | `InterviewState` — the single Pydantic model that flows through the graph |
| `models/model_loader.py` | Loads Ollama embedding model and LLM |
| `qdrant/qdrant.py` | `QdrantHybridClient` — async Qdrant wrapper with RRF fusion search |
| `Data/question.py` | `QuestionEmbeddings` — upserts and searches the question collection |
| `utils/Data_ingestion.py` | PDF loader (`Docloader`) and text chunker (`Chunking`) |

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.10 | |
| [uv](https://github.com/astral-sh/uv) | latest | Fast package manager |
| [Ollama](https://ollama.com) | latest | Runs local LLMs |
| [Qdrant](https://qdrant.tech) | latest | Vector database |

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/AI_interviewer.git
cd AI_interviewer
```

### 2 — Install dependencies

```bash
uv sync
```

> All runtime packages (FastAPI, LangGraph, LlamaIndex, Qdrant client, etc.) are listed in `requirements.txt`. Add them to `pyproject.toml` dependencies or install directly:
>
> ```bash
> uv pip install -r requirements.txt
> ```

### 3 — Pull Ollama models

```bash
# Embedding model
ollama pull qwen3-embedding:4b

# LLM (replace with any model you prefer)
ollama pull kimi-k2-thinking:cloud
```

### 4 — Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use the [Qdrant cloud](https://cloud.qdrant.io) free tier and set the URL in `qdrant/qdrant.py`.

### 5 — Run the server

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Reference

### Health

```
GET /health
```

```json
{ "status": "healthy" }
```

---

### Start an interview

```
POST /interview/start
```

**Body**

```json
{
  "user_id": "alice",
  "domain": "tech",
  "topic": "RAG",
  "difficulty": "medium"
}
```

**Response**

```json
{
  "question": "Can you explain how retrieval-augmented generation improves LLM accuracy?",
  "state": { ... }
}
```

---

### Submit an answer

```
POST /interview/answer
```

**Body**

```json
{
  "user_id": "alice",
  "answer": "RAG combines a retriever with a generator so the model ...",
  "state": { ... }
}
```

**Response**

```json
{
  "question": "What chunking strategy would you use for long documents?",
  "feedback": "Good answer! You correctly identified the two-stage pipeline.",
  "state": { ... },
  "done": false
}
```

> Pass the `state` object returned by each response into the next request — the server is stateless.

---

### Upload a resume

```
POST /resume/upload?user_id=alice
Content-Type: multipart/form-data
```

**Response**

```json
{
  "user_id": "alice",
  "filename": "alice_cv.pdf",
  "characters_extracted": 3821,
  "preview": "Alice Smith — Software Engineer ..."
}
```

---

## 🗂️ Project Structure

```
AI_interviewer/
├── app/
│   └── main.py                # FastAPI entry point
├── agents/
│   ├── main_agent.py          # LangGraph graph builder
│   ├── supervisor_agent.py    # Routing + difficulty control
│   ├── question_agent.py      # RAG & LLM question agents
│   ├── evaluator_agent.py     # Answer scoring
│   ├── feedback_agent.py      # Feedback generation
│   └── state.py               # InterviewState schema
├── models/
│   └── model_loader.py        # Ollama LLM + embedding loader
├── qdrant/
│   └── qdrant.py              # Async Qdrant hybrid client
├── Data/
│   └── question.py            # Question embeddings (upsert + search)
├── utils/
│   ├── Data_ingestion.py      # PDF loader & text chunker
│   ├── difficulty.py          # Difficulty helpers
│   └── domain.py              # Domain/topic definitions
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Qdrant URL | `qdrant/qdrant.py` | `http://localhost:6333` |
| Embedding model | `models/model_loader.py` | `qwen3-embedding:4b` |
| LLM model | `models/model_loader.py` | `kimi-k2-thinking:cloud` |
| Collection name | `Data/question.py` | `question_collection` |
| Max questions | `app/main.py` | `10` |
| RAG → LLM switch | `agents/supervisor_agent.py` | after 3 questions |

---

## 🧠 How the Interview Flow Works

```
start
  │
  ▼
supervisor ──► (question_count < 3) ──► rag_question_agent
  │                                            │
  │            (question_count >= 3) ──► llm_question_agent
  │                                            │
  ◄──────────────── await_user_answer ◄────────┘
  │
  ▼
evaluator_agent  (scores answer, appends to history)
  │
  ▼
feedback_agent   (generates conversational feedback)
  │
  ▼
supervisor  (adjusts difficulty, picks next topic)
  │
  └──► repeat until question_count == 10
```

1. **Supervisor** decides the next node based on `state.step` and `state.mode`.
2. For the first 3 questions, **RagQuestionAgent** retrieves a matching question from Qdrant using hybrid search, then personalises it with resume context via the LLM.
3. After 3 questions, **LLMQuestionAgent** generates entirely new questions from resume + conversation history.
4. **EvaluatorAgent** scores each answer (0 – 1) and stores `{question, answer, score}` in `state.history`.
5. **FeedbackAgent** produces natural-language feedback using the LLM.
6. The supervisor re-evaluates difficulty and topic before the next round.

---

## 🛠️ Development

### Running tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
```

### Adding a new question domain

1. Add the domain and its topics to `utils/domain.py`.
2. Upsert question embeddings for the new domain via `Data/question.py`.
3. Update `DOMAIN_TOPICS` in `agents/supervisor_agent.py`.

### Seeding the question bank

```python
from qdrant.qdrant import QdrantHybridClient
from Data.question import QuestionEmbeddings
import asyncio

questions = [
    {"question": "What is a vector database?", "domain": "tech", "topic": "Vector DB", "difficulty": "easy"},
    # ... more questions
]

async def seed():
    client = QdrantHybridClient()
    await client.create_collection("question_collection")
    embedder = QuestionEmbeddings(qdrant=client)
    dense, sparse, qs = embedder._embed_documents(questions)
    await embedder.upsert_documents(dense, sparse, qs)

asyncio.run(seed())
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT © 2026 Tanishq
