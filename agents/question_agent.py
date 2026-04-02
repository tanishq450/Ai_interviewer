from state import InterviewState
from models.model_loader import ModelLoader

class RagQuestionAgent:

    def __init__(self, resume_embedder, question_embedder, llm):
        self.resume_embedder = resume_embedder
        self.question_embedder = question_embedder
        self.llm = ModelLoader().load_llm()

    async def run(self, state:InterviewState):

        # -------- 1. Get base question from DB --------
        questions = await self.question_embedder.search(
            state.domain,
            state.topic,
            state.difficulty,
            state.topic  
        )

        base_q = questions[0] if questions else "Explain this concept."

        # -------- 2. Get resume context --------
        resume_docs = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        context = "\n".join(resume_docs)

        # -------- 3. LLM asks the question (rephrase only) --------
        prompt = f"""
        You are a professional interviewer.

        Ask this question naturally:

        Question: {base_q}

        Candidate context:
        {context}

        Rules:
        - Do NOT change meaning
        - Do NOT increase difficulty
        - Keep it conversational
        """

        question = self.llm.invoke(prompt)

        state.current_question = question
        state.step = "question"

        return {
            "goto": "await_user_answer",
            "state": state,
            "inputs": {}
        }


class LLMQuestionAgent:

    def __init__(self, resume_embedder, llm):
        self.resume_embedder = resume_embedder
        self.llm = llm

    async def run(self, state:InterviewState):

        # -------- 1. Resume context --------
        resume_docs = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        context = "\n".join(resume_docs)

        # -------- 2. Use history --------
        history = state.history[-2:] if state.history else []

        prompt = f"""
        You are an expert interviewer.

        Domain: {state.domain}
        Topic: {state.topic}
        Difficulty: {state.difficulty}

        Candidate context:
        {context}

        Previous interaction:
        {history}

        Instructions:
        - Ask ONE new question
        - Do NOT repeat previous questions
        - Go deeper based on answers
        - Increase difficulty gradually
        """

        question = self.llm.invoke(prompt)

        state.current_question = question
        state.step = "question"

        return {
            "goto": "await_user_answer",
            "state": state,
            "inputs": {}
        }