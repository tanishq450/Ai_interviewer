from agents.state import InterviewState


def _generate_text(llm, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        return str(llm.invoke(prompt))

    if hasattr(llm, "complete"):
        result = llm.complete(prompt)
        return str(getattr(result, "text", result))

    if hasattr(llm, "chat"):
        result = llm.chat([{"role": "user", "content": prompt}])
        return str(getattr(getattr(result, "message", None), "content", result))

    raise RuntimeError("LLM does not support invoke/complete/chat APIs")


class RagQuestionAgent:

    def __init__(self, resume_embedder, question_embedder, llm):
        self.resume_embedder = resume_embedder
        self.question_embedder = question_embedder
        self.llm = llm

    async def run(self, state: InterviewState):

        # -------- Build Query --------
        query = f"{state.topic} {state.difficulty} interview question"

        # -------- Retrieve from Qdrant --------
        retrieved_questions = await self.question_embedder.search(
            domain=state.domain,
            topic=state.topic,
            difficulty=state.difficulty,
            query=query
        )

        # fallback safety
        if not retrieved_questions:
            retrieved_questions = ["Explain this concept in detail."]

        # -------- Context Creation --------
        context = "\n".join(
            [f"- {q}" for q in retrieved_questions[:5]]
        )

        # -------- Resume Context (optional) --------
        resume_docs = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        resume_context = "\n".join(resume_docs) if resume_docs else ""

        # -------- History --------
        history = state.history[-2:] if state.history else []

        # -------- Prompt --------
        prompt = f"""
        You are an expert interviewer.

        Domain: {state.domain}
        Topic: {state.topic}
        Difficulty: {state.difficulty}

        Reference Questions:
        {context}

        Candidate Background:
        {resume_context}

        Conversation History:
        {history}

        Generate ONE new interview question.
        - Do not repeat previous questions
        - Match difficulty level
        - Keep it natural and conversational
        """

        # -------- Generate --------
        question = _generate_text(self.llm, prompt)

        # -------- Update State --------
        state.current_question = question
        state.step = "question"
        state.question_count += 1

        return {
            "goto": "await_user_answer",
            "state": state,
            "inputs": {}
        }

class LLMQuestionAgent:

    def __init__(self, resume_embedder, llm):
        self.resume_embedder = resume_embedder
        self.llm = llm

    async def run(self, state: InterviewState):

        # -------- Retrieve from DB --------
        resume_chunks = await self.resume_embedder.search(
            state.user_id,
            state.topic
        )

        resume_context = "\n".join(resume_chunks) if resume_chunks else ""

        history = state.history[-2:] if state.history else []

        prompt = f"""
        You are an interviewer.

        Topic: {state.topic}
        Difficulty: {state.difficulty}

        Candidate Background:
        {resume_context}

        Conversation history:
        {history}

        Generate ONE new question relevant to their background.
        Do not repeat.
        """

        question = _generate_text(self.llm, prompt)

        state.current_question = question
        state.step = "question"
        state.question_count += 1

        return {"goto": "await_user_answer", "state": state, "inputs": {}}