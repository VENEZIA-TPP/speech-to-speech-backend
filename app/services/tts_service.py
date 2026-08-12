import io
import time
import wave
from dataclasses import dataclass

from anyio import CapacityLimiter, to_thread

from app.pipeline.contracts import RawAudio, TTSState, WatermarkedAudio

_STUB_SAMPLE_RATE = 16000
_STUB_DURATION_MS = 300

# One token per stage: inference must not land in the framework's default
# thread pool (40 tokens, shared with its sync dependencies), where 40
# inferences could run at once against a single GPU. Module level because the
# engine is a frozen dataclass with no __dict__, and there is one engine per
# stage per process anyway.
_LIMITER = CapacityLimiter(1)


@dataclass
class TTSResult:
    # One field, not five: TTSResult(audio_bytes=..., watermarked=False,
    # watermark_method=None) used to be constructible, and the system would
    # have emitted it. The type now carries the guarantee.
    audio: WatermarkedAudio
    processing_time_ms: int


@dataclass(frozen=True)
class TTSService:
    """Immutable and shared process-wide. See ASRService for the full rationale."""

    __slots__ = ("model_name", "device")
    model_name: str
    device: str

    async def synthesize(
        self,
        state: TTSState,
        text: str,
        language: str,
    ) -> TTSResult:
        # No speaker parameter: the voice sample is state.speaker, sealed once
        # per session by the segmenter. Passing the current audio chunk as a
        # speaker reference - which this pipeline once did, cloning whoever
        # spoke last instead of a stable voice - is no longer expressible.
        start = time.monotonic()
        raw = await to_thread.run_sync(
            self._synthesize, state, text, language, limiter=_LIMITER
        )
        # The watermark stays on the loop: today it only builds a dataclass. It
        # moves into the thread alongside synthesis once the marking is done in
        # the audio domain.
        audio = self._apply_watermark(raw)
        # Rule #6, enforced by the type rather than by line order: _synthesize()
        # can only produce RawAudio, and only this hook produces the
        # WatermarkedAudio the pipeline accepts. The isinstance check is the one
        # place all synthesized audio passes through, so an overridden hook that
        # hands the raw audio back fails closed here instead of reaching a socket.
        if not isinstance(audio, WatermarkedAudio):
            raise TypeError(
                f"{type(self).__name__}._apply_watermark() must return "
                f"WatermarkedAudio, got {type(audio).__name__}"
            )
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return TTSResult(audio=audio, processing_time_ms=processing_time_ms)

    def _synthesize(
        self,
        state: TTSState,
        text: str,
        language: str,
    ) -> RawAudio:
        # Synchronous on purpose: this is the replacement point for a real
        # model, and real inference blocks. synthesize() dispatches it to a
        # thread. Writing it as `async def` is the trap: run_sync would hand
        # back a coroutine it never runs, so the inference would silently
        # never happen.
        # TODO: Internal synthesis - replace with real inference. Heavy
        # inference must run in an executor/worker, never inline on the event
        # loop. Clone from state.speaker when it is sealed; None means the
        # default voice - explicit degradation, never silent cloning.

        n_frames = int(_STUB_SAMPLE_RATE * _STUB_DURATION_MS / 1000)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_STUB_SAMPLE_RATE)
            wav.writeframes(b"\x00\x00" * n_frames)
        return RawAudio(
            data=buffer.getvalue(),
            sample_rate=_STUB_SAMPLE_RATE,
            duration_ms=_STUB_DURATION_MS,
        )

    def _apply_watermark(self, raw: RawAudio) -> WatermarkedAudio:
        # Watermark/tagging hook. Stub tags metadata only; real audio-domain
        # watermarking replaces this body and must still return
        # WatermarkedAudio - it never bypasses the type.
        return WatermarkedAudio(
            data=raw.data,
            sample_rate=raw.sample_rate,
            duration_ms=raw.duration_ms,
            method="stub-metadata-tag",
        )
