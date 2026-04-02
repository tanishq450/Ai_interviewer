class FeedbackAgent:

    def __init__(self, llm):
        self.llm = llm

    def run(self, state, inputs):

        last_eval = inputs.get("last_eval", {})

        question = last_eval.get("question", "")
        answer = last_eval.get("answer", "")
        score = last_eval.get("score", 0.0)

        # -------- Prompt --------
        prompt = f"""
        You are an expert interviewer.

        Question:
        {question}

        Candidate Answer:
        {answer}

        Score: {score}

        Provide structured feedback:
        - Strengths
        - Weaknesses
        - One improvement suggestion

        Keep it concise.
        """

        if hasattr(self.llm, "invoke"):
            feedback = str(self.llm.invoke(prompt))
        elif hasattr(self.llm, "complete"):
            result = self.llm.complete(prompt)
            feedback = str(getattr(result, "text", result))
        else:
            result = self.llm.chat([{"role": "user", "content": prompt}])
            feedback = str(getattr(getattr(result, "message", None), "content", result))

        # -------- Update state --------
        state.feedback.append(feedback)

        # Optional: track weak areas (simple heuristic)
        if score < 0.5:
            if state.topic:
                state.weak_areas.append(state.topic)

        # -------- Reset for next round --------
        state.current_question = None
        state.step = "question"

        return {
            "goto": "supervisor",
            "state": state,
            "inputs": {}
        }
        