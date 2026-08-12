import time
from dataclasses import dataclass
from typing import Optional

from anyio import CapacityLimiter, to_thread

from app.pipeline.contracts import ASRState

# One token per stage: inference must not land in the framework's default
# thread pool (40 tokens, shared with its sync dependencies), where 40
# inferences could run at once against a single GPU. Module level because the
# engine is a frozen dataclass with no __dict__, and there is one engine per
# stage per process anyway.
_LIMITER = CapacityLimiter(1)


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
    per-session state has nowhere to hide. slots=True would raise an
    unreadable TypeError for new names instead.

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
        state: ASRState,
        audio_bytes: bytes,
        source_language: Optional[str] = None,
    ) -> ASRResult:
        # `state` is first on purpose: the signature is what tells whoever
        # integrates a real streaming ASR where per-session memory belongs.
        # The engine is frozen, so self is not an option.
        start = time.monotonic()
        text, detected_language, confidence = await to_thread.run_sync(
            self._transcribe, state, audio_bytes, source_language, limiter=_LIMITER
        )
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return ASRResult(
            text=text,
            detected_language=detected_language or source_language,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

    def _transcribe(
        self,
        state: ASRState,
        audio_bytes: bytes,
        language: Optional[str],
    ) -> tuple[str, Optional[str], Optional[float]]:
        # Synchronous on purpose: this is the replacement point for a real
        # model, and real inference blocks. transcribe() dispatches it to a
        # thread. Writing it as `async def` is the trap: run_sync would hand
        # back a coroutine it never runs, so the inference would silently
        # never happen.
        # TODO: Internal transcription - replace with real inference. Buffer,
        # previous prompt and decoder cache go on `state`, never on self.
        # Stub doesn't need the audio bytes, so it doesn't buffer
        # them - state.buffer stays declared for the real streaming backend.
        state.chunks_seen += 1
        return (
            f"[ASR stub] transcription placeholder (chunk {state.chunks_seen})",
            language or "en",
            0.95,
        )
