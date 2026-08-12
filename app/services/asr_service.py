import time
from dataclasses import dataclass
from typing import Any, Optional

from anyio import CapacityLimiter, to_thread

from app.pipeline.contracts import ASRState
from app.pipeline.vad import pcm_from_wav

# One token per stage: inference must not land in the framework's default
# thread pool (40 tokens, shared with its sync dependencies), where 40
# inferences could run at once against a single GPU. Module level because the
# engine is a frozen dataclass with no __dict__, and there is one engine per
# stage per process anyway.
_LIMITER = CapacityLimiter(1)

# Every value ASR_MODEL accepts. Anything else raises at construction: a typo
# like "parakeet-v3" would otherwise fall through to the stub in silence, and
# the system would transcribe with placeholders in production without one line
# of output saying so.
_BACKENDS = ("stub", "parakeet")

_PIP_INSTALL = "pip install onnx-asr"

# Third-party ONNX export of NVIDIA's Parakeet TDT 0.6B v3 (cc-by-4.0, same
# license as the upstream weights). NVIDIA's own repository holds the NeMo
# checkpoint, which only the NeMo toolkit - and therefore torch - can load;
# this conversion is the artifact onnx-asr actually runs, so it is the one
# that gets pinned.
#
# Pinned by commit because it is a third-party repository: an unpinned
# revision hands someone else the ability to change what this process
# transcribes with, between one deploy and the next, with no diff here.
_PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
_PARAKEET_REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"

# The int8 set (~650 MB). fp32 would cost ~2.4 GB - the encoder outgrows the
# 2 GB protobuf limit and ships a sidecar .data file - for a quality delta
# nothing here can measure until a recorded corpus exists. Revisit against
# fp32 on the target GPU, with numbers, if quality ever looks ASR-bound.
#
# nemo128.onnx is the mel preprocessor: on CPU onnx-asr computes features in
# NumPy and never opens it, but with a CUDA provider it runs the ONNX graph
# instead, so the snapshot must carry it for a GPU process to start offline.
_PARAKEET_FILES = (
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
    "config.json",
)


@dataclass
class ASRResult:
    text: str
    detected_language: Optional[str]
    confidence: Optional[float]
    processing_time_ms: int


def _load_recognizer(device: str) -> Any:
    """Fetch (or hit the local cache for) the pinned export and open it.

    The heavy imports live in here rather than at module level so
    ASR_MODEL=stub pays nothing for a backend it does not use: the development
    machine has no GPU and should not import an inference runtime to run the
    test suite.
    """
    try:
        import onnx_asr
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            f"ASR_MODEL=parakeet needs the recognition runtime, which is not "
            f"installed: {_PIP_INSTALL}"
        ) from exc

    path = snapshot_download(
        _PARAKEET_REPO,
        revision=_PARAKEET_REVISION,
        allow_patterns=list(_PARAKEET_FILES),
    )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    # Default session threading, unlike the VAD's single-thread session: this
    # encoder is the heavy compute and it runs on a worker thread, so giving
    # it the cores is the point, while the VAD costs ~0.07 ms per frame on the
    # event loop and survives the contention. load_model() accepts
    # sess_options to cap this session too - reach for it only with a
    # measurement showing loop latency degrading during inference.
    return onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        path,
        quantization="int8",
        providers=providers,
    )


def _load_backend(model_name: str, device: str) -> Any:
    if model_name not in _BACKENDS:
        raise ValueError(
            f"unknown ASR_MODEL {model_name!r}; valid values: "
            f"{', '.join(repr(name) for name in _BACKENDS)}"
        )
    if model_name == "stub":
        return None
    return _load_recognizer(device)


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

    `_recognizer` is in __slots__ but deliberately NOT annotated, so it is a
    slot and not a dataclass field: it stays out of the generated __init__,
    and every existing ASRService("stub", "cpu") call site keeps working
    unchanged. __post_init__ fills it with object.__setattr__, the one way
    past a frozen dataclass.
    """

    __slots__ = ("model_name", "device", "_recognizer")
    model_name: str
    device: str

    def __post_init__(self) -> None:
        # Loaded here, eagerly, rather than on first use - for the same
        # reasons the MT models are: it fails fast at startup when weights
        # are missing (the contract the lifespan relies on), and it leaves
        # nothing to check-then-set at runtime.
        object.__setattr__(
            self, "_recognizer", _load_backend(self.model_name, self.device)
        )

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
        state.chunks_seen += 1
        if self._recognizer is None:
            # Stub doesn't need the audio bytes, so it doesn't buffer them -
            # state.buffer stays declared for a real streaming backend.
            # Confidence is None, not an invented score: the old hardcoded
            # 0.95 was persisted to the database as if it measured something.
            return (
                f"[ASR stub] transcription placeholder (chunk {state.chunks_seen})",
                language or "en",
                None,
            )

        # The wire protocol and the segmenter both fix 16 kHz mono s16le, and
        # pcm_from_wav() rejects anything that is not mono 16-bit.
        waveform = pcm_from_wav(audio_bytes)
        text = self._recognizer.recognize(waveform, sample_rate=16000)
        # Both None, not invented: the runtime returns bare text for this
        # model - no language-identification output and no usable score.
        # transcribe() then falls back to the session's declared language,
        # which session creation validated against the configured language
        # pairs, so a detected-vs-declared mismatch naming an unloaded MT
        # pair cannot arise from this backend. If a future backend does
        # report detection, the fail-or-fall-back choice must be made then,
        # when a real value exists to disagree with the declaration.
        return (text, None, None)
