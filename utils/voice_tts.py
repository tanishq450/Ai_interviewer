import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoTokenizer, VitsModel


class LocalTTSService:
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
        output_dir: str = "tmp_audio",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._audio_index: Dict[str, str] = {}

    def _load(self):
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = VitsModel.from_pretrained(self.model_name)
            self._model.eval()

    def synthesize(self, text: str) -> str:
        self._load()
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "Please continue."

        inputs = self._tokenizer(cleaned, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs).waveform

        waveform = output.squeeze().cpu().numpy()
        waveform = np.clip(waveform, -1.0, 1.0)
        sample_rate = int(self._model.config.sampling_rate)
        audio = (waveform * 32767).astype(np.int16)

        audio_id = str(uuid.uuid4())
        audio_path = self.output_dir / f"{audio_id}.wav"
        write_wav(str(audio_path), sample_rate, audio)
        self._audio_index[audio_id] = str(audio_path)
        return audio_id

    def get_audio_path(self, audio_id: str) -> str:
        return self._audio_index.get(audio_id, "")
