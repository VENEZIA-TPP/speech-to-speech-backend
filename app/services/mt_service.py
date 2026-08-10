import time
from dataclasses import dataclass

from app.pipeline.contracts import MTState


@dataclass
class MTResult:
    translated_text: str
    source_language: str
    target_language: str
    processing_time_ms: int


@dataclass(frozen=True)
class MTService:
    """Immutable and shared process-wide. See ASRService for the full rationale
    (ADR 0003, barrier #1); the per-language-pair cache PR 10 adds lives here
    as a read-only dict, touched by the MT stage only.
    """

    __slots__ = ("model_name", "device")
    model_name: str
    device: str

    async def translate(
        self,
        state: MTState,
        text: str,
        source_language: str,
        target_language: str,
    ) -> MTResult:
        # TODO: Translate text from source_language to target_language.
        start = time.monotonic()
        translated = await self._translate(
            state, text, source_language, target_language
        )
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return MTResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            processing_time_ms=processing_time_ms,
        )

    async def _translate(
        self,
        state: MTState,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        # TODO: Internal translation - replace with real inference. Per-session
        # context goes on `state`; the per-language-pair model cache belongs on
        # the engine and is read-only at runtime (ADR 0003).
        state.segments_seen += 1
        return f"[MT stub] {source_language}->{target_language}: {text}"
