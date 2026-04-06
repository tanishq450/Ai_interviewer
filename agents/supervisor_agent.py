from loguru import logger
from agents.state import InterviewState
import random


class SupervisorAgent:

    def __init__(self):
        self.logger = logger

    def _avg_last(self, scores, k=2):
        if not scores:
            return None
        tail = scores[-k:]
        return sum(tail) / len(tail)

    def _update_difficulty(self, state:InterviewState):
        avg = self._avg_last(state.scores, k=2)

        if avg is None:
            return state.difficulty

        if avg >= 0.8:
            return "hard"
        elif avg <= 0.4:
            return "easy"
        return "medium"


    def _next_topic(self, state:InterviewState):
        candidates = []

        # 1. Weak areas → highest priority
        for t in state.weak_areas:
            if state.topics_covered.count(t) < 2:
                candidates.append((t, 1.0))  # high priority

        # 2. Resume topics
        for t in getattr(state, "resume_topics", []):
            name = t["name"] if isinstance(t, dict) else t
            weight = t.get("weight", 0.7) if isinstance(t, dict) else 0.7

            penalty = state.topics_covered.count(name) * 0.3
            score = weight - penalty

            candidates.append((name, score))

        # 3. Fallback (only if resume is weak)
        if not candidates:
            fallback = {
                "tech": ["RAG", "FastAPI", "Vector DB"],
                "finance": ["valuation", "risk", "markets"],
            }
            pool = fallback.get(state.domain, ["general"])
            return random.choice(pool)

        # 4. Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 5. Add randomness (avoid repetition)
        top_k = candidates[:3]
        chosen = random.choice(top_k)

        return chosen[0]

    def run(self, state:InterviewState):

        self.logger.info(f"Step: {state.step}")

        # -------- MODE --------
        state.mode = "RAG" if state.question_count < 3 else "LLM"

        # -------- DIFFICULTY --------
        state.difficulty = self._update_difficulty(state)

        # -------- WAIT FOR ANSWER --------
        if state.step == "question" and state.current_question and not state.last_answer:
            return {"goto": "await_user_answer", "state": state, "inputs": {}}

        # -------- EVALUATE --------
        if state.step == "question" and state.last_answer:
            state.step = "evaluate"
            return {
                "goto": "evaluator_agent",
                "state": state,
                "inputs": {
                    "question": state.current_question,
                    "answer": state.last_answer
                }
            }

        # -------- FEEDBACK --------
        if state.step == "feedback":
            return {
                "goto": "feedback_agent",
                "state": state,
                "inputs": {
                    "last_eval": state.history[-1] if state.history else None
                }
            }

        # -------- NEXT QUESTION --------
        state.step = "question"

        state.topic = self._next_topic(state)

        state.topics_covered.append(state.topic)

        if state.mode == "RAG":
            return {"goto": "rag_question_agent", "state": state, "inputs": {}}

        return {"goto": "llm_question_agent", "state": state, "inputs": {}}