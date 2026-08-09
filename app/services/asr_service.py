import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRResult:
    text: str
    detected_language: Optional[str]
    confidence: Optional[float]
    processing_time_ms: int


@dataclass(frozen=True)
class ASRService:
    """Immutable and shared process-wide across every session.

    frozen + a hand-written __slots__ means `engine.buffer = ...` raises
    FrozenInstanceError for declared fields AND for new names alike, so
    per-session state has nowhere to hide (ADR 0003, barrier #1). slots=True
    would raise an unreadable TypeError for new names instead.

    No field defaults: __slots__ and class-level defaults are mutually
    exclusive (ValueError at import time). Callers pass both values.

    A real backend subclasses this as a @dataclass(frozen=True) with
    __slots__ = () - a plain subclass gets a __dict__ back and loses the
    barrier.
    """

    __slots__ = ("model_name", "device")
    model_name: str
    device: str

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_language: Optional[str] = None,
    ) -> ASRResult:
        # TODO: Transcribe a WAV audio chunk.

        start = time.monotonic()
        text, detected_language, confidence = await self._transcribe(
            audio_bytes, source_language
        )
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
