from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class InterviewState(BaseModel):
    # --- Core ---
    user_id: Optional[str] = None
    domain: str = "tech"
    topic: Optional[str] = None
    difficulty: str = "medium"

    # --- Resume ---
    resume_text: Optional[str] = None

    # --- Runtime ---
    current_question: Optional[str] = None
    last_answer: Optional[str] = None
    step: str = "question"   # important for supervisor

    # --- History (FIXED) ---
    history: List[Dict[str, Any]] = Field(default_factory=list)

    # --- Tracking ---
    scores: List[int] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)
    topics_covered: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)

    # --- Mode ---
    mode: str = "RAG"   # RAG → first 3, LLM → after