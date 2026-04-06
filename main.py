from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import json
import json_repair
import tempfile
import os
import uvicorn
from loguru import logger
import uuid

from agents.main_agent import InterviewGraph
from agents.state import InterviewState
from models.model_loader import ModelLoader
from qdrant.qdrant import QdrantHybridClient
from Data.question import QuestionEmbeddings
from Data.resume import ResumeEmbedder
from utils.Data_ingestion import Docloader
from utils.voice_tts import LocalTTSService
from utils.voice_stt import LocalSTTService
from typing import List, Dict, Any
import re


# -------------------------------
# Utils
# -------------------------------
def create_user():
    return str(uuid.uuid4())


# -------------------------------
# App setup
# -------------------------------
app = FastAPI(
    title="AI Interviewer",
    description="AI Interview system using RAG + LangGraph",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"422 Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Globals (initialized on startup)
# -------------------------------
model_loader = None
qdrant_client = None
question_embedder = None
resume_embedder = None
interview_graph = None
doc_loader = None
tts_service = None
stt_service = None
user_profiles = {}
# Resolve relative to this file so load/save works regardless of uvicorn cwd.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROFILES_PATH = os.path.join(_BASE_DIR, "data", "user_profiles.json")


def _load_user_profiles():
    global user_profiles
    if not os.path.exists(USER_PROFILES_PATH):
        return
    try:
        with open(USER_PROFILES_PATH, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)
        logger.info(f"Loaded {len(user_profiles)} resume profile(s) from disk")
    except Exception as e:
        logger.warning(f"Could not load {USER_PROFILES_PATH}: {e}")


def _save_user_profiles():
    os.makedirs(os.path.dirname(USER_PROFILES_PATH) or ".", exist_ok=True)
    try:
        with open(USER_PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(user_profiles, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save user profiles: {e}")


def infer_profile_from_resume(text: str) -> dict:
    lowered = (text or "").lower()

    tech_hits = sum(
        1 for w in ["python", "sql", "api", "llm", "machine learning", "backend", "fastapi"] if w in lowered
    )
    finance_hits = sum(
        1 for w in ["finance", "valuation", "equity", "portfolio", "market", "investment"] if w in lowered
    )
    hr_hits = sum(
        1 for w in ["recruitment", "hiring", "people ops", "talent", "onboarding", "employee"] if w in lowered
    )

    scores = {"tech": tech_hits, "finance": finance_hits, "hr": hr_hits}
    domain = max(scores, key=scores.get) if any(scores.values()) else "general"

    topic_map = {
        "tech": "RAG",
        "finance": "valuation",
        "hr": "behavioral",
        "general": "general",
    }

    return {
        "domain": domain,
        "topic": topic_map[domain],
        "difficulty": "medium",
    }


async def extract_resume_topics(text: str, llm) -> List[Dict[str, Any]]:
    """Extract key topics/skills from resume text using LLM."""
    prompt = f"""
Extract key technical topics and skills from this resume.
Return ONLY a JSON array with format: [{{"name": "skill/topic name", "weight": 0.5-1.0}}]
- weight: 1.0 for core expertise, 0.7 for familiar, 0.5 for mentioned
- Include 5-10 key topics max
- Focus on technical skills and domain knowledge

Resume:
{text[:2000]}
"""
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
        elif hasattr(llm, "complete"):
            result = llm.complete(prompt)
            result = getattr(result, "text", result)
        else:
            result = llm.chat([{"role": "user", "content": prompt}])
            result = getattr(getattr(result, "message", None), "content", result)

        result_str = str(result)
        # Try to extract JSON array from response
        
        json_match = re.search(r'\[.*\]', result_str, re.DOTALL)
        if json_match:
            result_str = json_match.group()

        topics = json_repair.loads(result_str)
        if isinstance(topics, list) and all(isinstance(t, dict) and "name" in t for t in topics):
            return topics
    except Exception as e:
        logger.warning(f"Topic extraction failed: {e}")

    # Fallback: extract basic topics from resume text
    return _fallback_topics(text)


def _fallback_topics(text: str) -> List[Dict[str, Any]]:
    """Extract basic topics from resume when LLM extraction fails."""
    lowered = (text or "").lower()

    tech_keywords = {
        "python": 1.0, "sql": 0.9, "api": 0.8, "fastapi": 1.0, "django": 0.9,
        "llm": 1.0, "rag": 1.0, "machine learning": 1.0, "deep learning": 1.0,
        "transformers": 0.9, "pytorch": 0.9, "tensorflow": 0.9, "nlp": 0.9,
        "vector database": 0.9, "qdrant": 1.0, "pinecone": 0.8, "langchain": 0.9,
        "docker": 0.8, "kubernetes": 0.8, "aws": 0.8, "gcp": 0.8, "azure": 0.8,
        "react": 0.8, "javascript": 0.8, "typescript": 0.8, "nodejs": 0.8,
        "backend": 0.9, "frontend": 0.8, "full stack": 0.9, "microservices": 0.9,
    }

    found = []
    for topic, weight in tech_keywords.items():
        if topic in lowered:
            found.append({"name": topic, "weight": weight})

    # Return top 8 by weight
    found.sort(key=lambda x: x["weight"], reverse=True)
    return found[:8] if found else [{"name": "general", "weight": 0.5}]


# -------------------------------
# Startup
# -------------------------------
@app.on_event("startup")
async def startup():
    global model_loader, qdrant_client, question_embedder, resume_embedder, interview_graph, doc_loader, tts_service, stt_service

    logger.info("Starting AI Interviewer...")

    model_loader = ModelLoader()
    llm = model_loader.load_llm()

    qdrant_client = QdrantHybridClient()

    try:
        await qdrant_client.create_collection("question_collection")
        await qdrant_client.create_collection("resume_collection")
    except Exception as e:
        logger.warning(f"Qdrant not available: {e}")

    question_embedder = QuestionEmbeddings(qdrant=qdrant_client)
    resume_embedder = ResumeEmbedder(qdrant=qdrant_client)

    interview_graph = InterviewGraph(
        resume_embedder=resume_embedder,
        question_embedder=question_embedder,
        llm=llm,
    )

    doc_loader = Docloader()
    tts_service = LocalTTSService()
    stt_service = LocalSTTService()

    _load_user_profiles()

    logger.info("AI Interviewer ready.")


# -------------------------------
# Schemas
# -------------------------------
class StartInterviewRequest(BaseModel):
    user_id: Optional[str] = None
    domain: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None


class AnswerRequest(BaseModel):
    user_id: str
    answer: str
    state: Any  # kept permissive; validated inside the handler for clearer errors


class StartInterviewResponse(BaseModel):
    question: str
    state: dict
    audio_id: Optional[str] = None


class AnswerResponse(BaseModel):
    question: Optional[str] = None
    feedback: Optional[str] = None
    state: dict
    done: bool = False
    audio_id: Optional[str] = None
    transcript: Optional[str] = None


# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/interview/start", response_model=StartInterviewResponse)
async def start_interview(
    request: Request,
    req: StartInterviewRequest = Body(default_factory=StartInterviewRequest),
):

    user_id = (req.user_id or request.query_params.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(
            422,
            "user_id is required: put it in the JSON body (e.g. {\"user_id\": \"test\"}) or use ?user_id=test",
        )
    # In-memory dict can be empty vs on-disk file (wrong cwd before fix, or reload timing).
    profile = user_profiles.get(user_id)
    if not profile:
        _load_user_profiles()
        profile = user_profiles.get(user_id)
    if not profile:
        raise HTTPException(
            400,
            "Resume profile not found. Upload a resume first with the same user_id (see data/user_profiles.json after an upload).",
        )

    state = InterviewState(
        user_id=user_id,
        domain=profile.get("domain", "general"),
        topic=profile.get("topic", "general"),
        difficulty=profile.get("difficulty", "medium"),
        resume_topics=profile.get("resume_topics", []),
    )

    try:
        result = await interview_graph.run(state)

        # FIX: correct extraction
        if not isinstance(result, dict) or "state" not in result:
            raise Exception("Invalid graph response")

        result_state = result["state"]

        question = result_state.current_question or "Tell me about yourself."
        from starlette.concurrency import run_in_threadpool
        audio_id = await run_in_threadpool(tts_service.synthesize, question)

        return StartInterviewResponse(
            question=question,
            state=result_state.model_dump(),
            audio_id=audio_id,
        )

    except Exception as e:
        logger.error(f"Start error: {e}")
        raise HTTPException(500, str(e))


@app.post(
    "/interview/answer",
    response_model=AnswerResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": AnswerRequest.model_json_schema()
                }
            },
            "required": True,
        }
    }
)
async def submit_answer(request: Request):
    body_text = (await request.body()).decode("utf-8")
    
    try:
        data = json_repair.loads(body_text)
    except Exception as e:
        logger.error(f"Cannot parse JSON body: {e} | body: {body_text}")
        raise HTTPException(400, "Invalid JSON body provided.")

    try:
        req = AnswerRequest(**data)
    except Exception as e:
        raise HTTPException(422, f"Validation failure: {e}")

    answer = req.answer
    raw_state = req.state

    # Validate and coerce state — log details if it fails
    try:
        if isinstance(raw_state, str):
            try:
                raw_state = json.loads(raw_state, strict=False)
            except Exception:
                pass
                
        if not isinstance(raw_state, dict):
            raw_state = dict(raw_state)
            
        state = InterviewState.model_validate(raw_state)
    except Exception as ve:
        logger.error(f"State validation failed: {ve} | raw state: {raw_state}")
        raise HTTPException(400, f"Invalid state: {ve}")

    state.last_answer = answer

    try:
        result = await interview_graph.run(state)

        # FIX: correct extraction
        if not isinstance(result, dict) or "state" not in result:
            raise Exception("Invalid graph response")

        result_state = result["state"]

        question = result_state.current_question
        feedback = result_state.feedback[-1] if result_state.feedback else None

        done = result_state.question_count >= 10
        from starlette.concurrency import run_in_threadpool
        audio_id = await run_in_threadpool(tts_service.synthesize, question) if question and not done else None

        return AnswerResponse(
            question=question,
            feedback=feedback,
            state=result_state.model_dump(),
            done=done,
            audio_id=audio_id,
        )

    except Exception as e:
        logger.error(f"Answer error: {e}")
        raise HTTPException(500, str(e))


@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    audio_path = tts_service.get_audio_path(audio_id)
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(404, "Audio not found")
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{audio_id}.wav")


@app.post("/interview/answer-voice", response_model=AnswerResponse)
async def submit_answer_voice(
    user_id: str = Form(...),
    state: str = Form(...),
    file: UploadFile = File(...),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from starlette.concurrency import run_in_threadpool
        transcript = await run_in_threadpool(stt_service.transcribe, tmp_path)
        if not transcript:
            raise HTTPException(422, "Could not transcribe audio")

        try:
            parsed_state = json_repair.loads(state)
            state_obj = InterviewState(**parsed_state)
        except Exception:
            raise HTTPException(400, "Invalid state")

        state_obj.user_id = user_id
        state_obj.last_answer = transcript

        result = await interview_graph.run(state_obj)
        if not isinstance(result, dict) or "state" not in result:
            raise Exception("Invalid graph response")

        result_state = result["state"]
        question = result_state.current_question
        feedback = result_state.feedback[-1] if result_state.feedback else None
        done = result_state.question_count >= 10
        audio_id = await run_in_threadpool(tts_service.synthesize, question) if question and not done else None

        return AnswerResponse(
            question=question,
            feedback=feedback,
            state=result_state.model_dump(),
            done=done,
            audio_id=audio_id,
            transcript=transcript,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice answer error: {e}")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id_form: Optional[str] = Form(None),
    user_id_query: Optional[str] = Query(None, alias="user_id", description="Same as form field user_id; either works"),
):

    user_id = (user_id_form or user_id_query or "").strip()
    if not user_id:
        raise HTTPException(
            422,
            "user_id is required: use form field user_id or query ?user_id=your-id",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from starlette.concurrency import run_in_threadpool
        text = await run_in_threadpool(doc_loader.load_pdf, tmp_path)

        if not text:
            raise HTTPException(422, "No text extracted")

        # Ingest into Qdrant for this user
        await resume_embedder.ingest(user_id, text)

        profile = infer_profile_from_resume(text)
        # Extract detailed topics from resume
        llm = model_loader.load_llm()
        resume_topics = await extract_resume_topics(text, llm)
        profile["resume_topics"] = resume_topics
        user_profiles[user_id] = profile
        _save_user_profiles()

        return {
            "user_id": user_id,
            "characters": len(text),
            "preview": text[:300],
            "detected_domain": profile["domain"],
            "detected_topic": profile["topic"],
            "detected_difficulty": profile["difficulty"],
            "resume_topics": resume_topics,
        }

    finally:
        os.unlink(tmp_path)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)