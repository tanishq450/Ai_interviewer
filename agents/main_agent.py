from agents.state import InterviewState
from agents.supervisor_agent import SupervisorAgent
from agents.question_agent import RagQuestionAgent, LLMQuestionAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.feedback_agent import FeedbackAgent


class InterviewGraph:

    def __init__(self, resume_embedder, question_embedder, llm):

        self.supervisor = SupervisorAgent()
        self.rag_question = RagQuestionAgent(resume_embedder, question_embedder, llm)
        self.llm_question = LLMQuestionAgent(resume_embedder, llm)
        self.evaluator = EvaluatorAgent()
        self.feedback = FeedbackAgent(llm)

    async def run(self, state: InterviewState):
        if state.step == "question" and state.current_question and not state.last_answer:
            return {"state": state}

        if state.last_answer:
            eval_result = self.evaluator.run(
                state,
                {
                    "question": state.current_question or "",
                    "answer": state.last_answer,
                },
            )
            state = eval_result["state"]

            feedback_result = self.feedback.run(
                state,
                {"last_eval": state.history[-1] if state.history else {}},
            )
            state = feedback_result["state"]

        # First question is always LLM-driven, then switch to RAG for early rounds.
        if state.question_count == 0:
            state.mode = "LLM"
        elif state.question_count < 3:
            state.mode = "RAG"
        else:
            state.mode = "LLM"
        state.difficulty = self.supervisor._update_difficulty(state)
        state.topic = self.supervisor._next_topic(state)

        if state.topic not in state.topics_covered:
            state.topics_covered.append(state.topic)

        if state.mode == "RAG":
            result = await self.rag_question.run(state)
        else:
            result = await self.llm_question.run(state)

        return {"state": result["state"]}