"""Typed contracts shared by the pipeline stages.

Imports nothing from FastAPI or app.models on purpose: contract tests must not
need the app, a database or an event loop.

Two kinds of object live here, with deliberately opposite properties:

  - Frozen values (SpeakerProfile, RawAudio, WatermarkedAudio): shared freely,
    safe to hand across sessions.
  - Per-session state (ASRState, MTState, TTSState, SessionState): mutable,
    exactly one per WebSocket connection, and never an attribute of an engine.

See docs/adr/0003-workers-persistentes-y-estado-por-sesion.md.
"""

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeakerProfile:
    """A sealed voice sample for cloning. Sealed = no later chunk overwrites it.

    The segmenter captures it in PR 12; PR 4 only defines the type, so that raw
    chunk bytes stop being a valid speaker reference anywhere in the pipeline.
    """

    __slots__ = ("pcm", "sample_rate", "duration_ms")
    pcm: bytes
    sample_rate: int
    duration_ms: int

    def __post_init__(self) -> None:
        if not self.pcm:
            raise ValueError("SpeakerProfile.pcm must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("SpeakerProfile.sample_rate must be positive")


@dataclass(frozen=True)
class RawAudio:
    """Synthesized audio before the watermark hook. The pipeline never sends it."""

    __slots__ = ("data", "sample_rate", "duration_ms")
    data: bytes
    sample_rate: int
    duration_ms: int


@dataclass(frozen=True)
class WatermarkedAudio:
    """The only audio the pipeline accepts, produced by TTSService._apply_watermark().

    A real _synthesize() cannot skip the hook because it cannot produce this
    type: the guarantee is structural, not a matter of two lines staying in
    order.
    """

    __slots__ = ("data", "sample_rate", "duration_ms", "method")
    data: bytes
    sample_rate: int
    duration_ms: int
    method: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("WatermarkedAudio.data must not be empty")
        if not self.method:
            raise ValueError("WatermarkedAudio.method must not be empty")

    @property
    def sha256(self) -> str:
        """Derived, never a field: a field can be filled with the hash of some
        other audio, a property cannot. PR 12 persists this to prove after the
        fact that a given output was tagged.
        """
        # ponytail: recomputed per call - hashing 300 ms of PCM is microseconds.
        # Cache it only if a profile ever says to.
        return hashlib.sha256(self.data).hexdigest()


# --- Per-session state. Mutable on purpose; one instance per connection. -----


@dataclass
class ASRState:
    """Where a streaming ASR keeps its memory between chunks.

    Buffer, previous prompt and decoder cache go here - never on the engine,
    which is frozen precisely so this is the only place left.
    """

    buffer: bytearray = field(default_factory=bytearray)
    chunks_seen: int = 0


@dataclass
class MTState:
    """Nearly empty today, and that is the point: the signature tells whoever
    integrates a real MT backend (PR 10) where per-session context goes, so the
    per-process cache stops looking like the natural home.
    """

    segments_seen: int = 0


@dataclass
class TTSState:
    """speaker is None until the segmenter seals a sample (PR 12). None means
    the default voice: explicit degradation, never silent cloning.
    """

    speaker: SpeakerProfile | None = None


@dataclass
class SessionState:
    """Built by the WebSocket handler right after accept(), dies with the
    coroutine. There is no global session_id -> state map to index wrong.
    """

    asr: ASRState = field(default_factory=ASRState)
    mt: MTState = field(default_factory=MTState)
    tts: TTSState = field(default_factory=TTSState)
