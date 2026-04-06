from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Step(str, Enum):
    QUESTION = "question"
    EVALUATE = "evaluate"
    FEEDBACK = "feedback"


class InterviewState(BaseModel):
    # --- Core ---
    user_id: Optional[str] = None
    domain: str = "tech"
    topic: Optional[str] = None
    difficulty: str = "medium"

    # --- Resume ---
    resume_text: Optional[str] = None
    resume_topics: List[Dict[str, Any]] = Field(default_factory=list)
    resume_chunks: List[str] = Field(default_factory=list)

    # --- Runtime ---
    current_question: Optional[str] = None
    current_context: List[str] = Field(default_factory=list)
    last_answer: Optional[str] = None
    step: Step = Step.QUESTION

    # --- Tracking ---
    question_count: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)

    topics_covered: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)

    question_embeddings: List[List[float]] = Field(default_factory=list)

    # --- Mode ---
    mode: str = "RAG"