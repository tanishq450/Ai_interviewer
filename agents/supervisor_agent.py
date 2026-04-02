from loguru import logger
from state import InterviewState


class SupervisorAgent:

    def __init__(self):
        self.logger = logger

    # -------- HELPERS --------
    def _avg_last(self, scores, k=2):
        if not scores:
            return None
        tail = scores[-k:]
        return sum(tail) / len(tail)

    def _update_difficulty(self, state:InterviewState):
        avg = self._avg_last(state.scores, k=2)

        if avg is None:
            return state.difficulty

        if avg >= 8:
            return "hard"
        elif avg <= 4:
            return "easy"
        return "medium"

    def _next_topic(self, state):

        DOMAIN_TOPICS = {
            "tech": ["RAG", "FastAPI", "Vector DB"],
            "finance": ["valuation", "risk", "markets"],
            "economics": ["inflation", "GDP", "policy"],
        }

        pool = DOMAIN_TOPICS.get(state.domain, [])

        # prioritize weak areas
        for t in state.weak_areas:
            if t not in state.topics_covered:
                return t

        # new topics
        for t in pool:
            if t not in state.topics_covered:
                return t

        return pool[0] if pool else "general"

    # -------- MAIN RUN --------
    def run(self, state:InterviewState):

        self.logger.info(f"Supervisor step: {state.step}")

        # -------- MODE SWITCH --------
        if len(state.history) < 3:
            state.mode = "RAG"
        else:
            state.mode = "LLM"

        # -------- DIFFICULTY UPDATE --------
        state.difficulty = self._update_difficulty(state)

        # -------- FLOW CONTROL --------

        # 1. Wait for user answer
        if state.step == "question" and state.current_question and not state.last_answer:
            return {
                "goto": "await_user_answer",
                "state": state,
                "inputs": {}
            }

        # 2. Evaluate answer
        if state.last_answer and state.step in ["question", "evaluate"]:
            state.step = "evaluate"

            return {
                "goto": "evaluator_agent",
                "state": state,
                "inputs": {
                    "question": state.current_question,
                    "answer": state.last_answer
                }
            }

        # 3. Feedback
        if state.step == "feedback":
            return {
                "goto": "feedback_agent",
                "state": state,
                "inputs": {
                    "last_eval": state.history[-1] if state.history else None
                }
            }

        # 4. Generate next question
        state.step = "question"

        next_topic = self._next_topic(state)
        state.topic = next_topic

        if next_topic not in state.topics_covered:
            state.topics_covered.append(next_topic)

        # -------- ROUTING --------
        if state.mode == "RAG":
            return {
                "goto": "rag_question_agent",
                "state": state,
                "inputs": {}
            }

        return {
            "goto": "llm_question_agent",
            "state": state,
            "inputs": {}
        }