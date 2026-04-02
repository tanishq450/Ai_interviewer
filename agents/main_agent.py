from langgraph.graph import StateGraph, END
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

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(InterviewState)

        # Nodes
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("rag_question", self.rag_question.run)
        workflow.add_node("llm_question", self.llm_question.run)
        workflow.add_node("evaluator", self.evaluator.run)
        workflow.add_node("feedback", self.feedback.run)

        workflow.set_entry_point("supervisor")

        # Dynamic routing (IMPORTANT)
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["goto"],
            {
                "rag_question": "rag_question",
                "llm_question": "llm_question",
                "evaluator": "evaluator",
                "feedback": "feedback",
                "await_user_answer": END
            }
        )

        # Back edges
        workflow.add_edge("rag_question", "supervisor")
        workflow.add_edge("llm_question", "supervisor")
        workflow.add_edge("evaluator", "supervisor")
        workflow.add_edge("feedback", "supervisor")

        return workflow.compile()

    def run(self, initial_state: InterviewState):
        return self.graph.invoke(initial_state)