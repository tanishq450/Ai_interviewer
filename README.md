# 🤖 AI Interviewer

An intelligent, voice-enabled interview assistant powered by **LangGraph**, **Ollama LLMs**, **Qdrant hybrid search**, and **FastAPI**. It conducts realistic technical interviews by generating personalised questions from a candidate's resume and a curated question bank, evaluating answers in real time, and providing structured feedback — all with **text-to-speech questions** and **speech-to-text answers**.

---

## ✨ Features

- **Voice-first experience** — the AI speaks every question via local TTS; candidates can answer by voice (STT) or text
- **Adaptive questioning** — starts with RAG-retrieved questions from a curated bank, then switches to LLM-generated questions as the session progresses
- **Resume-aware context** — uploads a PDF resume and auto-detects domain, topic, and difficulty from its content
- **Hybrid search (dense + sparse BM25)** — Qdrant RRF fusion for high-quality question retrieval
- **Dynamic difficulty** — supervisor agent adjusts difficulty (`easy / medium / hard`) based on rolling score averages
- **Multi-topic coverage** — tracks covered topics and weak areas to ensure a thorough interview
- **Stateless REST API** — clean FastAPI endpoints; state is passed back by the client so the server is horizontally scalable
- **Voice-mode frontend** — beautiful single-page app with waveform visualisation, audio playback, live transcript preview, and a results screen

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI  (main.py)                           │
│  POST /interview/start        POST /interview/answer                │
│  POST /interview/answer-voice GET  /audio/{audio_id}               │
│  POST /resume/upload          GET  /health                          │
└────────────────────────────┬────────────────────────────────────────┘
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
   │  QuestionEmbeddings  │     │      LocalTTSService     │
   │  (Qdrant hybrid)     │     │   LocalSTTService        │
   └──────────────────────┘     └─────────────────────────┘
```

### Key components

| File | Role |
|------|------|
| `main.py` | FastAPI app — startup, all HTTP routes, TTS/STT wiring |
| `agents/main_agent.py` | Builds and compiles the LangGraph `StateGraph` |
| `agents/supervisor_agent.py` | Routing logic — decides next node, adjusts difficulty & topic |
| `agents/question_agent.py` | `RagQuestionAgent` (first 3 Qs) and `LLMQuestionAgent` (thereafter) |
| `agents/evaluator_agent.py` | Scores answers and appends them to history |
| `agents/feedback_agent.py` | Generates conversational, natural-language feedback |
| `agents/state.py` | `InterviewState` — the single Pydantic model flowing through the graph |
| `models/model_loader.py` | Loads Ollama embedding (`qwen3-embedding:4b`) and LLM (`qwen3.5:397b-cloud`) |
| `qdrant/qdrant.py` | `QdrantHybridClient` — async Qdrant wrapper with RRF fusion search |
| `Data/question.py` | `QuestionEmbeddings` — upserts and searches the question collection |
| `utils/Data_ingestion.py` | PDF loader (`Docloader`) and text chunker (`Chunking`) |
| `utils/voice_tts.py` | `LocalTTSService` — synthesises WAV files from text |
| `utils/voice_stt.py` | `LocalSTTService` — transcribes audio files to text |
| `utils/domain.py` | Domain / topic definitions |
| `utils/difficulty.py` | Difficulty helpers |
| `frontend/` | Single-page voice-first interview UI (HTML + CSS + JS) |

---

## 🖥️ Frontend

The `frontend/` directory contains a standalone single-page application with four screens:

| Screen | Description |
|--------|-------------|
| **Landing** | Hero page with feature badges |
| **Setup** | Resume PDF drag-and-drop upload + User ID input |
| **Interview** | Split layout — chat transcript on left, controls on right |
| **Results** | Score grid, weak-area tags, per-question feedback |

The interview screen supports two answer modes:
- 🎙️ **Voice** — hold-to-record with live canvas waveform and transcript preview
- ✍️ **Text** — plain textarea with submit button

Audio questions are fetched from `/audio/{audio_id}` and auto-played when each question loads.

To serve the frontend locally:

```bash
cd frontend
python serve.py
# opens http://localhost:3000
```

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

> All runtime packages (FastAPI, LangGraph, LlamaIndex, Qdrant client, etc.) are listed in `requirements.txt`. You can also install directly:
>
> ```bash
> uv pip install -r requirements.txt
> ```

### 3 — Pull Ollama models

```bash
# Embedding model
ollama pull qwen3-embedding:4b

# LLM
ollama pull qwen3.5:397b-cloud
```

### 4 — Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use the [Qdrant cloud](https://cloud.qdrant.io) free tier and set the URL in `qdrant/qdrant.py`.

### 5 — Run the server

```bash
uvicorn main:app --reload
# or
python main.py
```

The API will be available at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Reference

### Health

```
GET /
```

```json
{ "status": "ok" }
```

---

### Upload a resume

```
POST /resume/upload
Content-Type: multipart/form-data
```

**Fields**: `file` (PDF), `user_id` (form field _or_ query param `?user_id=alice`)

**Response**

```json
{
  "user_id": "alice",
  "characters": 3821,
  "preview": "Alice Smith — Software Engineer …",
  "detected_domain": "tech",
  "detected_topic": "RAG",
  "detected_difficulty": "medium"
}
```

> Resume profiles are persisted to `data/user_profiles.json` between restarts.

---

### Start an interview

```
POST /interview/start
```

**Body**

```json
{ "user_id": "alice" }
```

> `domain`, `topic`, and `difficulty` are inferred from the uploaded resume automatically. You can override them in the body if needed.

**Response**

```json
{
  "question": "Can you explain how retrieval-augmented generation improves LLM accuracy?",
  "state": { "…": "…" },
  "audio_id": "a1b2c3d4"
}
```

Fetch the audio with `GET /audio/{audio_id}` (returns a WAV file).

---

### Submit a text answer

```
POST /interview/answer
Content-Type: application/json
```

**Body**

```json
{
  "user_id": "alice",
  "answer": "RAG combines a retriever with a generator so the model …",
  "state": { "…": "…" }
}
```

**Response**

```json
{
  "question": "What chunking strategy would you use for long documents?",
  "feedback": "Good answer! You correctly identified the two-stage pipeline.",
  "state": { "…": "…" },
  "done": false,
  "audio_id": "e5f6g7h8"
}
```

---

### Submit a voice answer

```
POST /interview/answer-voice
Content-Type: multipart/form-data
```

**Fields**: `user_id`, `state` (serialised JSON string), `file` (audio file — WAV, MP3, etc.)

**Response** — same shape as `/interview/answer`, plus:

```json
{
  "transcript": "RAG combines a retriever with a generator …",
  "…": "…"
}
```

---

> **Stateless protocol**: always pass the `state` object returned by each response into the next request. The server holds no per-session state.

---

## 🗂️ Project Structure

```
AI_interviewer/
├── main.py                    # FastAPI entry point (run with uvicorn or python main.py)
├── agents/
│   ├── main_agent.py          # LangGraph graph builder
│   ├── supervisor_agent.py    # Routing + difficulty control
│   ├── question_agent.py      # RAG & LLM question agents
│   ├── evaluator_agent.py     # Answer scoring
│   ├── feedback_agent.py      # Feedback generation
│   └── state.py               # InterviewState Pydantic schema
├── models/
│   └── model_loader.py        # Ollama LLM + embedding loader
├── qdrant/
│   └── qdrant.py              # Async Qdrant hybrid client (RRF fusion)
├── Data/
│   └── question.py            # Question embeddings (upsert + search)
├── utils/
│   ├── Data_ingestion.py      # PDF loader & text chunker
│   ├── voice_tts.py           # Local TTS service (WAV synthesis)
│   ├── voice_stt.py           # Local STT service (audio transcription)
│   ├── difficulty.py          # Difficulty helpers
│   └── domain.py              # Domain/topic definitions
├── frontend/
│   ├── index.html             # Single-page voice-first UI
│   ├── style.css              # Glassmorphism dark-mode styles
│   ├── app.js                 # Frontend logic (voice, fetch, state mgmt)
│   └── serve.py               # Dev server for the frontend
├── data/
│   └── user_profiles.json     # Persisted resume profiles (auto-created)
├── tmp_audio/                 # Temp WAV files for TTS (auto-created)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Qdrant URL | `qdrant/qdrant.py` | `http://localhost:6333` |
| Embedding model | `models/model_loader.py` | `qwen3-embedding:4b` |
| LLM model | `models/model_loader.py` | `qwen3.5:397b-cloud` |
| Question collection | `Data/question.py` | `question_collection` |
| Max questions | `main.py` | `10` |
| RAG → LLM switch | `agents/supervisor_agent.py` | after 3 questions |
| User profiles path | `main.py` | `data/user_profiles.json` |

---

## 🧠 How the Interview Flow Works

```
Upload resume (POST /resume/upload)
  │  → domain / topic / difficulty inferred and saved
  ▼
POST /interview/start  →  supervisor  →  rag_question_agent (Q 1-3)
                                    └→  llm_question_agent  (Q 4-10)
                                              │
                                         TTS synthesise
                                              │
                               Client plays audio, user answers
                                              │
POST /interview/answer (text) ─────────────────┤
POST /interview/answer-voice (audio) ──────────┘
  │  → STT transcribe (voice path only)
  ▼
evaluator_agent  (scores answer 0–1, appends to history)
  ▼
feedback_agent   (generates natural-language feedback)
  ▼
supervisor  (adjusts difficulty, picks next topic)
  │
  └──► repeat until question_count == 10
         └─► return done: true, show results screen
```

1. **Supervisor** decides the next node based on `state.step` and `state.mode`.
2. For the first 3 questions, **RagQuestionAgent** retrieves a matching question from Qdrant using hybrid (dense + BM25) search, then personalises it with resume context via the LLM.
3. After 3 questions, **LLMQuestionAgent** generates entirely new questions from resume + conversation history.
4. **EvaluatorAgent** scores each answer (0–1) and stores `{question, answer, score}` in `state.history`.
5. **FeedbackAgent** produces natural-language feedback using the LLM.
6. The supervisor re-evaluates difficulty and topic before the next round.
7. Audio for each question is saved to `tmp_audio/` and served via `GET /audio/{audio_id}`.

---

## 🐳 Docker

```bash
docker build -t ai-interviewer .
docker run -p 8000:8000 ai-interviewer
```

> Make sure Qdrant and Ollama are reachable from inside the container (update URLs accordingly).

---

## 🛠️ Development

### Adding a new question domain

1. Add the domain and its topics to `utils/domain.py`.
2. Upsert question embeddings for the new domain via `Data/question.py`.
3. Update `DOMAIN_TOPICS` in `agents/supervisor_agent.py`.
4. Add keyword hints to `infer_profile_from_resume()` in `main.py` if you want auto-detection.

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
