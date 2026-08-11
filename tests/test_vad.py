"""Segmentation tests, against the real Silero weights.

No mock of the model anywhere in this file. It is 1.3 MB, runs on CPU in
production too, and is deterministic to the bit - so a double would only be
faster, and would stop testing the thing that actually decides where a segment
ends.

Every case is built from one recorded utterance plus exactly generated silence
(see tests/fixtures/README.md). Silero does not fire on synthetic audio, so the
speech has to be real; the gaps between speech do not, and generating them makes
each duration exact to the sample.
"""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.pipeline.contracts import VADState
from app.pipeline.vad import build_segmenter, pcm_from_wav

SR = settings.AUDIO_SAMPLE_RATE
FIXTURE = Path(__file__).parent / "fixtures" / "es_sistema.wav"


@pytest.fixture(scope="module")
def segmenter():
    return build_segmenter()


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    return pcm_from_wav(FIXTURE.read_bytes())


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def wav(*parts: np.ndarray) -> bytes:
    """Wrap concatenated samples as one WAV frame, the way a client sends it."""
    samples = np.concatenate(parts) if len(parts) > 1 else parts[0]
    pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def duration_ms(segment_wav: bytes) -> int:
    with wave.open(io.BytesIO(segment_wav)) as w:
        return int(w.getnframes() * 1000 / w.getframerate())


# --- the boundary: what closes a segment and what does not ------------------


@pytest.mark.parametrize("gap_ms", [0, 100, 200, 250])
def test_gap_below_min_silence_does_not_split(segmenter, speech, gap_ms):
    """A pause shorter than the threshold is inside a phrase, not the end of one.

    This is the half of the dial that protects translation quality: cutting here
    would hand MT a syntactically broken fragment with no context to translate it.
    """
    state = VADState()
    segments = segmenter.feed(state, wav(speech, silence(gap_ms), speech, silence(600)))

    assert len(segments) == 1


@pytest.mark.parametrize("gap_ms", [300, 400, 500, 800])
def test_gap_above_min_silence_splits(segmenter, speech, gap_ms):
    state = VADState()
    segments = segmenter.feed(state, wav(speech, silence(gap_ms), speech, silence(600)))

    assert len(segments) == 2
    assert all(s.reason == "silence" for s in segments)


def test_min_silence_is_the_dial_that_decides(segmenter, speech):
    """The same 300 ms pause splits or not depending only on the setting.

    Guards against the split being driven by something else - the chunk
    boundary, the utterance length - which would still produce two segments here
    and would look identical from the outside.
    """
    audio = wav(speech, silence(300), speech, silence(700))
    counts = {}
    for min_silence_ms in (200, 400):
        tuned = build_segmenter()
        object.__setattr__(
            tuned, "min_silence_frames", round(min_silence_ms * SR / 1000 / 512)
        )
        counts[min_silence_ms] = len(tuned.feed(VADState(), audio))

    assert counts == {200: 2, 400: 1}


# --- what never reaches ASR -------------------------------------------------


def test_silence_produces_no_segment(segmenter):
    """The short-circuit: no segment means no ASR call and no database row.

    Under fixed chunking every chunk of silence still ran the whole pipeline and
    persisted an empty transcription.
    """
    assert segmenter.feed(VADState(), wav(silence(3000))) == []


def test_burst_shorter_than_min_speech_is_discarded(segmenter, speech):
    """A cough-length burst is noise, and noise must not become a segment."""
    burst = speech[: int(SR * 0.2)]  # 200 ms, below VAD_MIN_SPEECH_MS

    assert segmenter.feed(VADState(), wav(silence(300), burst, silence(600))) == []


def test_trailing_silence_cannot_pad_a_burst_over_the_bar(segmenter, speech):
    """min_speech counts frames that cleared the hysteresis floor, not length.

    A segment always carries its pre-roll and up to min_silence_ms of tail, so
    measuring the minimum against total length would let a 200 ms burst pass as
    ~800 ms of "speech".
    """
    burst = speech[: int(SR * 0.2)]
    segments = segmenter.feed(VADState(), wav(silence(400), burst, silence(900)))

    assert segments == []


# --- the ceiling ------------------------------------------------------------


def test_max_speech_forces_a_close(segmenter, speech):
    """Continuous speech past the ceiling is cut rather than left to grow.

    Workers are shared between sessions, so an uncapped segment is an uncapped
    blocking call: one speaker who never pauses would stall everyone else.
    """
    continuous = np.concatenate([speech] * 6)  # ~10.9 s, no pause anywhere
    segments = segmenter.feed(VADState(), wav(continuous, silence(600)))

    assert len(segments) == 2
    assert segments[0].reason == "ceiling"
    assert duration_ms(segments[0].pcm) <= settings.VAD_MAX_SPEECH_MS


def test_ceiling_carries_the_remainder_instead_of_dropping_it(speech):
    """The speaker did not stop, so the audio after the cut is still speech.

    The reference implementation discards an over-long segment whole; here that
    would silently lose the middle of a long sentence.

    The look-back is widened past the pre-roll on purpose. At the shipped
    settings this test cannot fail: the cut is 400 ms back, the pre-roll holds
    300 ms, so simply dropping the tail still re-captures almost all of it and
    the coverage assertion passes either way - verified by mutation. Widening
    the look-back to 900 ms separates the two, which is the point: the ceiling
    must not depend on speech_pad_ms happening to be large enough to paper over
    the gap.
    """
    wide = build_segmenter()
    object.__setattr__(wide, "lookback_frames", round(900 * SR / 1000 / 512))

    continuous = np.concatenate([speech] * 6)
    spoken_ms = int(len(continuous) * 1000 / SR)
    segments = wide.feed(VADState(), wav(continuous, silence(600)))

    covered = sum(duration_ms(s.pcm) for s in segments)
    assert covered >= spoken_ms


# --- the hysteresis ---------------------------------------------------------
#
# These two drive the state machine with injected probabilities instead of the
# model. Not to avoid the weights - every other test here uses them - but
# because the recorded fixture is clean speech that scores ~1.0 or ~0.0, so it
# never visits the band between the floor and the threshold where the
# hysteresis is the only thing deciding anything. Setting HYSTERESIS to 0
# leaves the rest of this file green (verified by mutation); it fails here.


def _drive(segmenter, probs, monkeypatch):
    """Feed one frame per probability, bypassing the model."""
    supplied = iter(probs)
    monkeypatch.setattr(
        type(segmenter), "_speech_prob", lambda self, state, frame: next(supplied)
    )
    state = VADState()
    frame = np.zeros(512, dtype=np.float32)
    return [
        segment
        for segment in (segmenter._feed_frame(state, frame) for _ in probs)
        if segment is not None
    ], state


def test_dip_into_the_hysteresis_band_does_not_end_the_segment(segmenter, monkeypatch):
    """Staying in speech needs less confidence than entering it.

    A quiet consonant that dips to 0.4 - below the 0.5 that would have started
    the segment, above the 0.35 floor that ends it - is still the same phrase.
    Without the asymmetry the machine chatters across the boundary and shreds
    one sentence into fragments.
    """
    speech_frames = [0.9] * 20
    dip = [0.4] * 30  # far longer than min_silence, but inside the band
    segments, state = _drive(segmenter, speech_frames + dip, monkeypatch)

    assert segments == []
    assert state.triggered is True


def test_dip_below_the_floor_does_end_the_segment(segmenter, monkeypatch):
    """The other side of the same asymmetry, so the test above cannot pass by
    the machine simply never closing anything."""
    segments, state = _drive(segmenter, [0.9] * 20 + [0.2] * 30, monkeypatch)

    assert len(segments) == 1
    assert state.triggered is False


# --- end of session ---------------------------------------------------------


def test_flush_closes_the_open_segment(segmenter, speech):
    """Without this the last utterance of every session is lost, silently.

    The session ends one phrase short and nothing reports it: no error, no empty
    segment, just a translation that never arrives.
    """
    state = VADState()
    assert segmenter.feed(state, wav(speech, silence(100))) == []

    flushed = segmenter.flush(state)

    assert flushed is not None
    assert flushed.reason == "flush"
    assert duration_ms(flushed.pcm) > 1000


def test_flush_with_nothing_open_produces_nothing(segmenter):
    """A session that ended on silence must not emit a phantom segment."""
    state = VADState()
    segmenter.feed(state, wav(silence(1000)))

    assert segmenter.flush(state) is None


def test_flush_is_idempotent(segmenter, speech):
    """Nothing in the handler guarantees flush runs once; nothing should break."""
    state = VADState()
    segmenter.feed(state, wav(speech, silence(100)))

    assert segmenter.flush(state) is not None
    assert segmenter.flush(state) is None


# --- the anchor -------------------------------------------------------------


def test_each_segment_carries_its_own_capture_time(segmenter, speech):
    """Two segments from one chunk must not share one timestamp.

    The anchor used to be a local taken when the binary frame arrived, so both
    segments in a chunk like this one got the same value and the second
    segment's end-to-end figure absorbed the entire processing of the first.
    """
    segments = segmenter.feed(
        VADState(), wav(speech, silence(500), speech, silence(600))
    )

    assert len(segments) == 2
    assert segments[0].t_capture != segments[1].t_capture
    assert segments[0].t_capture < segments[1].t_capture


def test_flushed_segment_has_an_anchor(segmenter, speech):
    """The path with no binary frame in its iteration at all.

    A frame-arrival anchor has nothing to read here: the flush happens on a text
    frame, where the old local was either stale or never assigned.
    """
    state = VADState()
    segmenter.feed(state, wav(speech, silence(100)))

    assert segmenter.flush(state).t_capture > 0


# --- streaming across chunk boundaries --------------------------------------


def test_segment_spans_chunks(segmenter, speech):
    """N chunks in, M segments out, with M != N.

    Speech split across three chunks with the pause in the middle chunk: two of
    them close nothing at all, and the segment that comes out is longer than any
    single chunk that carried it.
    """
    first, second = speech[: len(speech) // 2], speech[len(speech) // 2 :]
    state = VADState()

    assert segmenter.feed(state, wav(first)) == []
    assert segmenter.feed(state, wav(second)) == []
    segments = segmenter.feed(state, wav(silence(600)))

    assert len(segments) == 1
    assert duration_ms(segments[0].pcm) > 1000


def test_partial_frames_are_carried_across_chunks(segmenter, speech):
    """Chunks rarely divide evenly into 512-sample frames.

    Dropping the remainder would punch a hole into the audio at every chunk
    boundary; the damage is inaudible per chunk and cumulative. Feeding the same
    speech in ragged pieces must produce the same segment as feeding it whole.
    """
    whole = segmenter.feed(VADState(), wav(speech, silence(600)))

    state = VADState()
    stream = np.concatenate([speech, silence(600)])
    ragged = []
    for start in range(0, len(stream), 777):  # not a multiple of 512
        ragged += segmenter.feed(state, wav(stream[start : start + 777]))

    assert len(whole) == len(ragged) == 1
    assert abs(duration_ms(whole[0].pcm) - duration_ms(ragged[0].pcm)) <= 64


# --- the trap that makes all of the above possible --------------------------


def test_model_needs_the_context_window(segmenter, speech):
    """Feeding Silero 512 samples without the 64-sample context returns ~0.

    The ONNX input dimension is dynamic, so the wrong length does not raise - it
    quietly reports "no speech" for every frame and the VAD never fires. This
    pins the contract that makes the difference.
    """
    state = VADState()
    frame = speech[512:1024]  # squarely inside the speech
    with_context = segmenter._speech_prob(state, frame)

    assert with_context > 0.5

    bare = segmenter.session.run(
        None,
        {
            "input": frame.reshape(1, -1),
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "sr": np.array(SR, dtype=np.int64),
        },
    )[0][0][0]

    assert bare < 0.1


def test_non_wav_frame_is_rejected(segmenter):
    """Garbage bytes are a client protocol error, not silence."""
    with pytest.raises(ValueError, match="not a readable WAV"):
        segmenter.feed(VADState(), b"fake_audio_chunk")
