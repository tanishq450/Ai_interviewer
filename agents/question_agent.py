from agents.state import InterviewState
from models.model_loader import ModelLoader


class RagQuestionAgent:

    def __init__(self, resume_embedder, question_embedder, llm=None):
        self.resume_embedder = resume_embedder
        self.question_embedder = question_embedder
        self.llm = llm or ModelLoader().load_llm()

    async def run(self, state: InterviewState):

        questions = await self.question_embedder.search(
            state.domain,
            state.topic,
            state.difficulty,
            state.topic
        )

        base_q = questions[0] if questions and len(questions) > 0 else "Explain this concept."

        resume_docs = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        context = "\n".join(resume_docs) if resume_docs else ""

        prompt = f"""
        Ask this question naturally:

        Question: {base_q}
        Context: {context}

        Keep it simple and conversational.
        """

        question = str(self.llm.invoke(prompt))

        state.current_question = question
        state.step = "question"
        state.question_count += 1

        return {"goto": "await_user_answer", "state": state, "inputs": {}}


class LLMQuestionAgent:

    def __init__(self, resume_embedder, llm):
        self.resume_embedder = resume_embedder
        self.llm = llm

    async def run(self, state: InterviewState):

        resume_docs = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        context = "\n".join(resume_docs) if resume_docs else ""

        history = state.history[-2:] if state.history else []

        prompt = f"""
        Domain: {state.domain}
        Topic: {state.topic}
        Difficulty: {state.difficulty}

        Context:
        {context}

        History:
        {history}

        Ask ONE new question. Do not repeat.
        """

        question = str(self.llm.invoke(prompt))

        state.current_question = question
        state.step = "question"
        state.question_count += 1

        return {"goto": "await_user_answer", "state": state, "inputs": {}}