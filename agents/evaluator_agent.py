class EvaluatorAgent:

    def run(self, state, inputs):

        question = inputs["question"]
        answer = inputs["answer"]

        score = min(len(answer) / 50, 1.0)

        state.scores.append(score)

        state.history.append({
            "question": question,
            "answer": answer,
            "score": score
        })

        # reset answer (IMPORTANT FIX)
        state.last_answer = None

        state.step = "feedback"

        return {"goto": "supervisor", "state": state, "inputs": {}}