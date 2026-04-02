from loguru import logger
from agents.state import InterviewState


class SupervisorAgent:

    def __init__(self):
        self.logger = logger

    def _avg_last(self, scores, k=2):
        if not scores:
            return None
        tail = scores[-k:]
        return sum(tail) / len(tail)

    def _update_difficulty(self, state: InterviewState):
        avg = self._avg_last(state.scores, k=2)

        if avg is None:
            return state.difficulty

        if avg >= 0.8:
            return "hard"
        elif avg <= 0.4:
            return "easy"
        return "medium"

    def _next_topic(self, state):

        DOMAIN_TOPICS = {
            "tech": ["RAG", "FastAPI", "Vector DB"],
            "finance": ["valuation", "risk", "markets"],
        }

        pool = DOMAIN_TOPICS.get(state.domain, [])

        for t in state.weak_areas:
            if t not in state.topics_covered:
                return t

        for t in pool:
            if t not in state.topics_covered:
                return t

        return pool[0] if pool else "general"

    def run(self, state: InterviewState):

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

        if state.topic not in state.topics_covered:
            state.topics_covered.append(state.topic)

        if state.mode == "RAG":
            return {"goto": "rag_question_agent", "state": state, "inputs": {}}

        return {"goto": "llm_question_agent", "state": state, "inputs": {}}