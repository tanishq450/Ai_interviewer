import hashlib
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import redis
import torch
from loguru import logger
from scipy.io.wavfile import write as write_wav
from transformers import AutoTokenizer, VitsModel


class LocalTTSService:
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
        output_dir: str = "tmp_audio",
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        redis_db: int = 0,
        cache_ttl: int = 86400,  # 24 hours
        redis_password: str = "",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._audio_index: Dict[str, str] = {}

        # Redis caching
        self._redis: Optional[redis.Redis] = None
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._cache_ttl = cache_ttl
        self._redis_connected = False
        self._redis_password = redis_password
        self._connect_redis()

    def _connect_redis(self):
        
        try:
            self._redis = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                db=self._redis_db,
                decode_responses=False,
                socket_connect_timeout=3,
                socket_timeout=3,
                password=self._redis_password,
            )
            self._redis.ping()
            self._redis_connected = True
            logger.info(
                f"TTS cache connected to Redis at {self._redis_host}:{self._redis_port}"
            )
        except Exception as e:
            self._redis_connected = False
            logger.warning(f"TTS Redis cache unavailable ({e}), running without cache")

    @staticmethod
    def _cache_key(text: str) -> str:
        """Deterministic key from input text."""
        return f"tts:{hashlib.sha256(text.encode()).hexdigest()}"

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

        # --- Check Redis cache ---
        cache_key = self._cache_key(cleaned)
        if self._redis_connected:
            try:
                cached = self._redis.get(cache_key)
                if cached is not None:
                    audio_id = str(uuid.uuid4())
                    audio_path = self.output_dir / f"{audio_id}.wav"
                    audio_path.write_bytes(cached)
                    self._audio_index[audio_id] = str(audio_path)
                    logger.debug(f"TTS cache HIT for key={cache_key[:16]}...")
                    return audio_id
            except Exception as e:
                logger.warning(f"TTS cache read failed: {e}")

        # --- Run model inference ---
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

        
        if self._redis_connected:
            try:
                wav_bytes = audio_path.read_bytes()
                self._redis.setex(cache_key, self._cache_ttl, wav_bytes)
                logger.debug(f"TTS cache MISS, stored key={cache_key[:16]}... ({len(wav_bytes)} bytes)")
            except Exception as e:
                logger.warning(f"TTS cache write failed: {e}")

        return audio_id

    def get_audio_path(self, audio_id: str) -> str:
        return self._audio_index.get(audio_id, "")