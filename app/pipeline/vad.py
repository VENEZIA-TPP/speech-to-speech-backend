"""Voice-driven segmentation: where one segment of speech ends.

Imports neither FastAPI nor app.models, for the same reason contracts.py does
not: a segmentation test must not need the app, a database or an event loop.

This replaces cutting the stream by the clock. Under fixed chunking the capture
term entered the latency budget whole and always - three seconds of it - no
matter how long the speaker actually spoke, which put the end-to-end target out
of reach by protocol design rather than by model choice. Under voice-driven
segmentation the capture term stops being a function of chunk size and becomes
the silence the speaker leaves at the end of a phrase.

The state machine is transferred from huggingface/speech-to-speech's
VADIterator, minus every conversational concept in it (turn reopening,
speculative turns): this pipeline translates what was already said, so there is
no generative answer whose commitment could be deferred, and nothing to reopen.

Four behaviours were taken from that reference deliberately, and each is easy to
get subtly wrong on a rewrite:

  - Onset requires the full threshold, but staying in speech only requires
    `threshold - 0.15`. That asymmetry is the hysteresis. A single symmetric
    threshold makes the machine chatter across the boundary and shreds a phrase
    into fragments.
  - Every frame is appended to the open segment BEFORE the silence test, so a
    segment keeps the low-confidence tail that was already spoken while the
    machine was still deciding the phrase had ended.
  - The pre-roll pad applies to the head of a segment only, never the tail. It
    costs no latency because it is audio already captured, and it keeps the
    attack of the first syllable.
  - The model's own recurrent state is reset once per session, never between
    segments. Resetting per segment is the intuitive move and it is wrong: it
    throws away the acoustic context that makes the next decision accurate.

The ceiling is this project's own, and the reference has nothing to copy for it:
there, a segment over the limit is discarded whole, and only after a natural
silence already closed it - so continuous speech never trips it at all. Dropping
a long sentence is not an option here, so the ceiling cuts and carries the
remainder instead.
"""

import io
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.pipeline.contracts import VADState

# Vendored rather than downloaded: see app/pipeline/weights/README.md for why
# the PyPI package is not used and why this is the 16 kHz-only build.
_WEIGHTS_PATH = Path(__file__).parent / "weights" / "silero_vad_16k_op15.onnx"

# Silero consumes exactly 512 samples per step at 16 kHz, and the ONNX build
# additionally wants the previous 64 samples prepended as context - 576 values
# per call. The input dimension is dynamic, so feeding it a bare 512 does NOT
# raise: it returns a near-zero probability for every frame, and the VAD simply
# never fires. Verified against the reference implementation's own wrapper.
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
# The hysteresis gap. Onset needs the full threshold; staying triggered needs
# this much less.
HYSTERESIS = 0.15
_SAMPLE_BYTES = 2  # pcm_s16le
_INT16_FULL_SCALE = 32768.0


@dataclass(frozen=True)
class SpeechSegment:
    """One utterance, as decided by the VAD rather than by the clock.

    `t_capture` is the anchor the end-to-end latency clock starts from. It
    travels with the segment instead of with the frame that happened to carry
    it, which is what makes it survive a chunk that closes two segments, a
    chunk that closes none, and a segment closed by the end-of-session flush -
    none of which a per-frame timestamp can express.
    """

    __slots__ = ("pcm", "duration_ms", "t_capture", "reason")
    pcm: bytes
    duration_ms: int
    t_capture: float
    reason: str  # "silence" | "ceiling" | "flush"


def pcm_from_wav(data: bytes) -> np.ndarray:
    """Decode one WAV frame into mono float32 in [-1, 1].

    The wire carries a self-contained 16 kHz mono pcm_s16le WAV per binary
    frame, so the header is re-parsed per chunk. Raises ValueError on anything
    else, which the WebSocket handler already turns into an error event and a
    failed session.
    """
    try:
        with wave.open(io.BytesIO(data)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != _SAMPLE_BYTES:
                raise ValueError(
                    "audio must be mono 16-bit PCM, got "
                    f"{wav.getnchannels()} channels / {wav.getsampwidth() * 8}-bit"
                )
            raw = wav.readframes(wav.getnframes())
    except wave.Error as exc:
        raise ValueError(f"audio frame is not a readable WAV: {exc}") from exc
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE


@dataclass(frozen=True)
class VoiceSegmenter:
    """Frozen and shared process-wide, like the three inference engines.

    Built once in the lifespan. The ONNX session is thread-safe and stateless
    across calls: everything that varies per session lives on VADState, passed
    as the first argument of every method - the same rule the engines follow.
    """

    __slots__ = (
        "session",
        "sample_rate",
        "threshold",
        "min_silence_frames",
        "min_speech_frames",
        "pad_frames",
        "max_frames",
        "lookback_frames",
    )
    session: ort.InferenceSession
    sample_rate: int
    threshold: float
    min_silence_frames: int
    min_speech_frames: int
    pad_frames: int
    max_frames: int
    lookback_frames: int

    @property
    def _floor(self) -> float:
        return self.threshold - HYSTERESIS

    def _speech_prob(self, state: VADState, frame: np.ndarray) -> float:
        if state.model_state is None:
            state.model_state = np.zeros((2, 1, 128), dtype=np.float32)
            state.context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        window = np.concatenate([state.context, frame.reshape(1, -1)], axis=1)
        out, state.model_state = self.session.run(
            None,
            {
                "input": window,
                "state": state.model_state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        state.context = window[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])

    def feed(self, state: VADState, wav_bytes: bytes) -> list[SpeechSegment]:
        """One arriving chunk in, however many segments it closed out.

        Zero, one or several - a chunk is not a segment. Runs inline on the
        event loop: measured at 0.066 ms per 32 ms frame on one CPU thread,
        so a 3 s chunk costs about 6 ms, against inference stages that are
        orders of magnitude larger. Move it behind to_thread.run_sync only if
        that ratio stops holding; a thread hop would cost more than it saves
        today.
        """
        samples = pcm_from_wav(wav_bytes)
        if state.residue is not None and len(state.residue):
            samples = np.concatenate([state.residue, samples])

        segments = []
        offset = 0
        while offset + FRAME_SAMPLES <= len(samples):
            segment = self._feed_frame(state, samples[offset : offset + FRAME_SAMPLES])
            if segment is not None:
                segments.append(segment)
            offset += FRAME_SAMPLES

        # Samples that did not fill a frame are carried, not dropped: at 3 s
        # chunks that is under 32 ms each time, but discarding it would punch a
        # gap into the audio at every chunk boundary.
        state.residue = samples[offset:]
        return segments

    def _feed_frame(self, state: VADState, frame: np.ndarray) -> SpeechSegment | None:
        prob = self._speech_prob(state, frame)

        if not state.triggered:
            state.preroll.append(frame)
            if len(state.preroll) > self.pad_frames:
                state.preroll.pop(0)
            if prob >= self.threshold:
                state.triggered = True
                state.segment = [*state.preroll, frame]
                # The pre-roll is audio from before the trigger, so it is not
                # speech by this machine's own decision and must not count
                # toward the minimum speech duration.
                state.probs = [0.0] * len(state.preroll) + [prob]
                state.preroll = []
                state.active_frames = 1
                state.silence_frames = 0
            return None

        state.segment.append(frame)
        state.probs.append(prob)
        if prob >= self._floor:
            state.active_frames += 1
            state.silence_frames = 0
        else:
            state.silence_frames += 1
            if state.silence_frames >= self.min_silence_frames:
                return self._close(state, "silence")

        if len(state.segment) >= self.max_frames:
            return self._close(state, "ceiling")
        return None

    def flush(self, state: VADState) -> SpeechSegment | None:
        """Close whatever is still open, at end of session.

        Without this the last utterance of every session is lost - the classic
        bug of this layer, and silent: the session simply ends one phrase short
        and nothing reports it.
        """
        if not state.triggered:
            return None
        return self._close(state, "flush")

    def _close(self, state: VADState, reason: str) -> SpeechSegment | None:
        frames, probs = state.segment, state.probs
        active = state.active_frames
        carry: list[np.ndarray] = []
        carry_probs: list[float] = []

        if reason == "ceiling" and self.lookback_frames:
            # The speaker has not stopped; the ceiling did. Cut at the quietest
            # frame in the tail so the split lands in a gap between words
            # rather than mid-syllable - this reduces the risk, it does not
            # remove it - and carry the remainder into the next segment. The
            # remainder is real speech: dropping it would lose what was said.
            window = probs[-self.lookback_frames :]
            cut = len(probs) - len(window) + int(np.argmin(window))
            carry, carry_probs = frames[cut:], probs[cut:]
            frames, probs = frames[:cut], probs[:cut]
            active = sum(1 for p in probs if p >= self._floor)

        state.triggered = False
        state.silence_frames = 0
        state.active_frames = 0
        state.segment = []
        state.probs = []
        state.preroll = []

        if carry:
            state.triggered = True
            state.segment, state.probs = carry, carry_probs
            state.active_frames = sum(1 for p in carry_probs if p >= self._floor)

        if active < self.min_speech_frames:
            # A burst too short to be speech. Discarding it here is what keeps
            # a cough or a door from reaching ASR at all.
            return None

        pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        return SpeechSegment(
            pcm=_wav_bytes(pcm, self.sample_rate),
            duration_ms=int(len(pcm) * 1000 / self.sample_rate),
            # Taken at the close, not at the arrival of the frame that closed
            # it: this is the anchor the segment carries downstream.
            t_capture=time.monotonic(),
            reason=reason,
        )


def build_segmenter() -> VoiceSegmenter:
    """Load the vendored weights and turn the millisecond settings into frames.

    Called once from the lifespan, like the three inference engines: if the
    weights are missing the exception propagates and the process refuses to
    start, rather than failing on some session's first chunk.

    The durations are converted here, once, so the hot path compares integers
    instead of recomputing the same division per frame.
    """
    options = ort.SessionOptions()
    # One thread: the work is 0.066 ms per frame, so the cost of handing it to
    # a thread pool is larger than the work itself, and a pool here would
    # compete with the inference stages for the same cores.
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(_WEIGHTS_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    sample_rate = settings.AUDIO_SAMPLE_RATE
    frame_ms = FRAME_SAMPLES * 1000 / sample_rate

    def frames(ms: int) -> int:
        return round(ms / frame_ms)

    return VoiceSegmenter(
        session=session,
        sample_rate=sample_rate,
        threshold=settings.VAD_THRESHOLD,
        min_silence_frames=frames(settings.VAD_MIN_SILENCE_MS),
        min_speech_frames=frames(settings.VAD_MIN_SPEECH_MS),
        pad_frames=frames(settings.VAD_SPEECH_PAD_MS),
        max_frames=frames(settings.VAD_MAX_SPEECH_MS),
        lookback_frames=frames(settings.VAD_CEILING_LOOKBACK_MS),
    )


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Re-wrap float32 samples as a 16 kHz mono pcm_s16le WAV.

    The segment goes back out as a WAV because that is what the ASR stage
    receives today; when a real backend wants raw tensors this is the single
    place that changes.
    """
    pcm16 = np.clip(samples * _INT16_FULL_SCALE, -_INT16_FULL_SCALE, 32767).astype(
        "<i2"
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(_SAMPLE_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buffer.getvalue()
