import time
from dataclasses import dataclass

import anyio
from anyio import CapacityLimiter

from app.pipeline.contracts import MTState

# One token per stage. Without this limiter the dispatch would land in the
# default thread pool - 40 tokens, shared with the framework's own sync
# dependencies - and up to 40 inferences could run in parallel against a
# single GPU. The limiter lives on the module, not on the instance: the engine
# is a frozen dataclass with no __dict__, and there is exactly one engine per
# stage per process anyway, which is the same cardinality.
_LIMITER = CapacityLimiter(1)


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
        translated = await anyio.to_thread.run_sync(
            self._translate,
            state,
            text,
            source_language,
            target_language,
            limiter=_LIMITER,
        )
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return MTResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            processing_time_ms=processing_time_ms,
        )

    def _translate(
        self,
        state: MTState,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        # Synchronous on purpose: this is the replacement point for a real
        # model, and real inference blocks. translate() dispatches it to a
        # thread, so an `await` in here would have no loop to run on.
        # TODO: Internal translation - replace with real inference. Per-session
        # context goes on `state`; the per-language-pair model cache belongs on
        # the engine and is read-only at runtime (ADR 0003).
        state.segments_seen += 1
        return f"[MT stub] {source_language}->{target_language}: {text}"
