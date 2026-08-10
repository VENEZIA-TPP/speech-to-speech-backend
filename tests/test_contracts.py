"""Contract tests - no app, no DB, no event loop.

app/pipeline/contracts.py imports nothing from FastAPI or app.models on
purpose, and these tests are what keeps it that way.
"""

import hashlib

import pytest

from app.pipeline.contracts import (
    ASRState,
    MTState,
    RawAudio,
    SessionState,
    SpeakerProfile,
    TTSState,
    WatermarkedAudio,
)


def test_watermarked_audio_rejects_empty_method():
    """A method nobody ran is not a watermark."""
    with pytest.raises(ValueError):
        WatermarkedAudio(
            data=b"RIFFdata", sample_rate=16000, duration_ms=300, method=""
        )


def test_watermarked_audio_rejects_empty_data():
    with pytest.raises(ValueError):
        WatermarkedAudio(data=b"", sample_rate=16000, duration_ms=300, method="stub")


def test_watermarked_audio_sha256_is_derived_from_its_bytes():
    """sha256 is a property, not a field: it cannot be declared, only computed.

    PR 12 persists this value and asserts it against the bytes the client
    received; that assertion is only worth anything if the hash cannot lie.
    """
    audio = WatermarkedAudio(
        data=b"RIFFdata", sample_rate=16000, duration_ms=300, method="stub"
    )
    assert audio.sha256 == hashlib.sha256(b"RIFFdata").hexdigest()


def test_speaker_profile_rejects_empty_pcm():
    with pytest.raises(ValueError):
        SpeakerProfile(pcm=b"", sample_rate=16000, duration_ms=5000)


def test_speaker_profile_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        SpeakerProfile(pcm=b"\x00\x00", sample_rate=0, duration_ms=5000)


def test_raw_audio_is_not_a_watermarked_audio():
    """The whole type lock rests on these two being unrelated types."""
    raw = RawAudio(data=b"RIFFdata", sample_rate=16000, duration_ms=300)
    assert not isinstance(raw, WatermarkedAudio)


def test_session_state_substates_are_not_shared_between_sessions():
    """Guards the classic mutable-default bug: without default_factory both
    sessions would write into the same bytearray."""
    a = SessionState()
    b = SessionState()

    a.asr.chunks_seen += 1
    a.asr.buffer.extend(b"chunk")
    a.mt.segments_seen += 1

    assert b.asr.chunks_seen == 0
    assert b.asr.buffer == bytearray()
    assert b.mt.segments_seen == 0


def test_tts_state_starts_without_a_speaker():
    """No sealed sample yet means the default voice - explicit degradation.
    The capture itself lands in PR 12."""
    assert TTSState().speaker is None


def test_states_are_mutable_on_purpose():
    """The engines are frozen; the state is where writing is allowed."""
    asr, mt = ASRState(), MTState()
    asr.chunks_seen = 7
    mt.segments_seen = 7
    assert (asr.chunks_seen, mt.segments_seen) == (7, 7)
