from transformers import pipeline


class LocalSTTService:
    def __init__(self, model_name: str = "distil-whisper/distil-small.en"):
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
            )

    def transcribe(self, audio_path: str) -> str:
        self._load()
        result = self._pipe(audio_path)
        return (result.get("text") or "").strip()
