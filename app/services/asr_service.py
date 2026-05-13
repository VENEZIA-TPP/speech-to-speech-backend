import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRResult:
    text: str
    detected_language: Optional[str]
    confidence: Optional[float]
    processing_time_ms: int


class ASRService:
    def __init__(self, model_name: str = "stub", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # TODO: load real model

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_language: Optional[str] = None,
    ) -> ASRResult:
        #TODO: Transcribe a WAV audio chunk.

        start = time.monotonic()
        text, detected_language, confidence = await self._transcribe(audio_bytes, source_language)
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return ASRResult(
            text=text,
            detected_language=detected_language or source_language,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

    async def _transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str],
    ) -> tuple[str, Optional[str], Optional[float]]:
        # TODO: Internal transcription - replace with real inference.

        return "[ASR stub] transcription placeholder", language or "en", 0.95
