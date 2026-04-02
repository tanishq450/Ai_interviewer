from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import uvicorn
from loguru import logger

from agents.main_agent import InterviewGraph
from agents.state import InterviewState
from models.model_loader import ModelLoader
from qdrant.qdrant import QdrantHybridClient
from Data.question import QuestionEmbeddings
from utils.Data_ingestion import Docloader
import uuid


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AI Interviewer",
    description="An AI-powered interview assistant using RAG + LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Shared singletons (initialised on startup)
# ─────────────────────────────────────────────
model_loader: ModelLoader = None
qdrant_client: QdrantHybridClient = None
question_embedder: QuestionEmbeddings = None
interview_graph: InterviewGraph = None
doc_loader: Docloader = None


@app.on_event("startup")
async def startup():
    global model_loader, qdrant_client, question_embedder, interview_graph, doc_loader

    logger.info("Starting AI Interviewer …")

    model_loader = ModelLoader()
    llm = model_loader.load_llm()

    qdrant_client = QdrantHybridClient()

    # Ensure the question collection exists
    await qdrant_client.create_collection("question_collection")

    question_embedder = QuestionEmbeddings(qdrant=qdrant_client)

    # A minimal resume embedder shim (search returns [] until a resume is uploaded)
    class _ResumeEmbedder:
        async def search(self, user_id: str, topic: str):
            return []

    interview_graph = InterviewGraph(
        resume_embedder=_ResumeEmbedder(),
        question_embedder=question_embedder,
        llm=llm,
    )

    doc_loader = Docloader()
    logger.info("AI Interviewer ready.")


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class StartInterviewRequest(BaseModel):
    user_id: uuid.UUID
    domain: str = "tech"
    topic: Optional[str] = None
    difficulty: str = "medium"


class AnswerRequest(BaseModel):
    user_id: str
    answer: str
    # Pass the current state back so the server stays stateless
    state: dict


class StartInterviewResponse(BaseModel):
    question: str
    state: dict


class AnswerResponse(BaseModel):
    question: Optional[str] = None
    feedback: Optional[str] = None
    state: dict
    done: bool = False


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Interviewer is running"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.post("/interview/start", response_model=StartInterviewResponse, tags=["Interview"])
async def start_interview(req: StartInterviewRequest):
    """
    Start a new interview session.
    Returns the first question and the initial state dict.
    """
    state = InterviewState(
        user_id=req.user_id,
        domain=req.domain,
        topic=req.topic,
        difficulty=req.difficulty,
    )

    try:
        result = interview_graph.run(state)
        result_state = result if isinstance(result, dict) else result.dict()
        question = result_state.get("current_question", "Tell me about yourself.")
        return StartInterviewResponse(question=question, state=result_state)
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/answer", response_model=AnswerResponse, tags=["Interview"])
async def submit_answer(req: AnswerRequest):
    """
    Submit a candidate's answer and receive the next question or final feedback.
    """
    try:
        state = InterviewState(**req.state)
        state.last_answer = req.answer

        result = interview_graph.run(state)
        result_state = result if isinstance(result, dict) else result.dict()

        question = result_state.get("current_question")
        feedback_list = result_state.get("feedback", [])
        latest_feedback = feedback_list[-1] if feedback_list else None

        done = result_state.get("question_count", 0) >= 10

        return AnswerResponse(
            question=question,
            feedback=latest_feedback,
            state=result_state,
            done=done,
        )
    except Exception as e:
        logger.error(f"Error processing answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume/upload", tags=["Resume"])
async def upload_resume(user_id: str, file: UploadFile = File(...)):
    """
    Upload a PDF resume for a user.  The text is extracted and returned.
    Future: embed and store in Qdrant for personalised questions.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = doc_loader.load_pdf(tmp_path)
        if not text:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        logger.info(f"Resume uploaded for user {user_id} ({len(text)} chars)")
        return {
            "user_id": user_id,
            "filename": file.filename,
            "characters_extracted": len(text),
            "preview": text[:300] + "…" if len(text) > 300 else text,
        }
    finally:
        os.unlink(tmp_path)


@app.get("/interview/summary/{user_id}", tags=["Interview"])
async def get_summary(user_id: str, state: dict = None):
    """
    Placeholder endpoint – in a real setup you'd retrieve the session
    from a store. For now, accepts the state dict via query/body.
    """
    return {
        "user_id": user_id,
        "message": "Pass the state dict to /interview/answer to continue, "
                   "or build a session store for persistent summaries.",
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
