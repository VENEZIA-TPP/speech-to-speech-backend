import io
import time
import wave
from dataclasses import dataclass
from typing import Optional

_STUB_SAMPLE_RATE = 16000
_STUB_DURATION_MS = 300


@dataclass
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    duration_ms: int
    watermarked: bool
    watermark_method: Optional[str]
    processing_time_ms: int


class TTSService:
    def __init__(self, model_name: str = "stub", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # TODO: load real model

    async def synthesize(
        self,
        text: str,
        language: str,
        speaker_reference: Optional[bytes] = None,
    ) -> TTSResult:
        # TODO: Synthesize speech for a translated segment. speaker_reference is
        # the voice sample to clone once the real model lands; the stub ignores it.
        start = time.monotonic()
        audio_bytes, sample_rate, duration_ms = await self._synthesize(
            text, language, speaker_reference
        )
        # Watermark applied here so a real _synthesize() cannot skip it.
        audio_bytes, watermarked, watermark_method = self._apply_watermark(audio_bytes)
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            watermarked=watermarked,
            watermark_method=watermark_method,
            processing_time_ms=processing_time_ms,
        )

    async def _synthesize(
        self,
        text: str,
        language: str,
        speaker_reference: Optional[bytes],
    ) -> tuple[bytes, int, int]:
        # TODO: Internal synthesis - replace with real inference. Heavy inference
        # must run in an executor/worker, never inline on the event loop.

        n_frames = int(_STUB_SAMPLE_RATE * _STUB_DURATION_MS / 1000)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_STUB_SAMPLE_RATE)
            wav.writeframes(b"\x00\x00" * n_frames)
        return buffer.getvalue(), _STUB_SAMPLE_RATE, _STUB_DURATION_MS

    def _apply_watermark(self, audio_bytes: bytes) -> tuple[bytes, bool, Optional[str]]:
        # Watermark/tagging hook. Stub tags metadata only;
        # real audio-domain watermarking replaces this body.
        return audio_bytes, True, "stub-metadata-tag"
