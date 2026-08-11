"""Typed events for the streaming WebSocket protocol (server -> client).

Every server frame carries a `type`. A segment is identified by
`segment_index` and is NOT the same thing as an audio chunk the client sent:
the client must not assume one segment per chunk, nor that a segment's events
arrive before it may send the next chunk. Today the pipeline happens to emit
one segment per chunk, but nothing in this protocol says so.

Binary frames carry synthesized audio and never ride inside the JSON: each one
is announced by the `audio.delta` immediately before it.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class SessionCreated(BaseModel):
    """First frame after the handshake is accepted."""

    type: Literal["session.created"] = "session.created"
    session_id: int


class TranscriptionCompleted(BaseModel):
    type: Literal["transcription.completed"] = "transcription.completed"
    segment_index: int
    transcript: str
    language_code: Optional[str] = None


class TranslationCompleted(BaseModel):
    type: Literal["translation.completed"] = "translation.completed"
    segment_index: int
    text: str
    target_language: str


class AudioDelta(BaseModel):
    """Announces the binary frame that follows it. One per binary frame."""

    type: Literal["audio.delta"] = "audio.delta"
    segment_index: int
    seq: int
    size_bytes: int


class AudioDone(BaseModel):
    """No more audio for this segment."""

    type: Literal["audio.done"] = "audio.done"
    segment_index: int
    watermarked: bool
    watermark_method: Optional[str] = None


class SegmentMetrics(BaseModel):
    """Per-stage timings, named after what is actually measured.

    `e2e_ms` covers the whole server-side window for the segment: from the
    moment the handler has the audio in hand to the moment the segment's
    whole synthesized audio frame has been handed to the transport. It
    includes the persistence writes, the serialization of the events that
    precede the audio, and the send itself, so it is always
    >= asr_ms + mt_ms + tts_ms, as long as each stage keeps measuring its own
    call rather than reporting a model's self-timed figure. Two things it
    deliberately does not cover: how long the frame sat unread in the socket
    buffer (the ASGI interface never reports when it arrived, so the clock
    starts at "the server has the audio", not at "the speaker stopped
    talking"), and the events that follow the audio frame. It is None for a
    segment that carried no audio, since there is no audio frame to stop it
    on.

    The TTS figure is the full synthesis time, not a time-to-first-byte,
    because the synthesis is not streamed yet.
    """

    type: Literal["segment.metrics"] = "segment.metrics"
    segment_index: int
    asr_ms: Optional[int] = None
    mt_ms: Optional[int] = None
    tts_ms: Optional[int] = None
    e2e_ms: Optional[int] = None


class SessionCompleted(BaseModel):
    type: Literal["session.completed"] = "session.completed"
    session_id: int
    total_segments: int


class ErrorEvent(BaseModel):
    """`segment_index` is None for errors that belong to no segment."""

    type: Literal["error"] = "error"
    code: str
    message: str
    segment_index: Optional[int] = None
