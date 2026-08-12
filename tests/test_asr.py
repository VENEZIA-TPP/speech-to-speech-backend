"""Real transcription, and the numbers it deliberately does not invent.

The parakeet backend returns None for both detected language and confidence:
the runtime hands back bare text for this model, with no language
identification and no usable score. A value invented here would be persisted
and later read as if it measured something - which is exactly what the stub's
old hardcoded 0.95 did. These tests pin the None contract and the fallback it
triggers: with no detection to disagree with it, the session's declared
language is what reaches MT, so a detected-vs-declared mismatch naming an
unloaded language pair cannot arise from this backend.
"""

import os
from pathlib import Path

import pytest

from app.pipeline.contracts import ASRState
from app.services import asr_service as asr_module
from app.services.asr_service import ASRService

_FIXTURE_WAV = Path(__file__).parent / "fixtures" / "es_sistema.wav"


def _recording_loader(loaded):
    """Stands in for _load_recognizer, and records what it was asked to load.

    The object it returns describes the waveform it was given, so an
    assertion can tell "the audio went through the model" apart from "the
    stub text came back anyway".
    """

    class Fake:
        def recognize(self, waveform, sample_rate=16000):
            return f"recognized {len(waveform)} samples at {sample_rate} Hz"

    def load(device):
        loaded.append(device)
        return Fake()

    return load


async def test_asr_confidence_is_none_when_unavailable(monkeypatch):
    """No score is invented for a backend that does not report one."""
    loaded = []
    monkeypatch.setattr(asr_module, "_load_recognizer", _recording_loader(loaded))
    engine = ASRService("parakeet", "cpu")

    # One eager load at construction, with the configured device.
    assert loaded == ["cpu"]

    result = await engine.transcribe(
        ASRState(), _FIXTURE_WAV.read_bytes(), source_language="es"
    )

    # The audio reached the model - this is not the stub answering.
    assert result.text.startswith("recognized")
    assert "16000 Hz" in result.text
    assert result.confidence is None


async def test_asr_detected_language_falls_back_to_declared(monkeypatch):
    """The backend detects nothing, so the declared language is what rides on.

    The raw hook reports None; the public wrapper falls back to the language
    the caller declared, which is what the pipeline hands to MT. Session
    creation validates that language against the configured pairs, so this
    fallback is what keeps an unloaded-pair error unreachable at runtime.
    """
    monkeypatch.setattr(asr_module, "_load_recognizer", _recording_loader([]))
    engine = ASRService("parakeet", "cpu")

    text, detected, confidence = engine._transcribe(
        ASRState(), _FIXTURE_WAV.read_bytes(), "es"
    )
    assert detected is None
    assert confidence is None

    result = await engine.transcribe(
        ASRState(), _FIXTURE_WAV.read_bytes(), source_language="es"
    )
    assert result.detected_language == "es"


def test_unknown_asr_model_raises_listing_options():
    """A typo must fail at startup, not transcribe with placeholders forever."""
    with pytest.raises(ValueError) as exc:
        ASRService("parakeet-v3", "cpu")

    message = str(exc.value)
    assert "parakeet-v3" in message
    for backend in ("stub", "parakeet"):
        assert backend in message, f"the error does not offer {backend!r}"


def test_stub_backend_loads_nothing(monkeypatch):
    """ASR_MODEL=stub must not touch the loader.

    Without this, a misplaced gate would keep every other test green while
    making the whole suite download 650 MB of weights.
    """

    def explode(device):
        raise AssertionError("the stub backend loaded a model")

    monkeypatch.setattr(asr_module, "_load_recognizer", explode)

    engine = ASRService("stub", "cpu")
    text, _, confidence = engine._transcribe(ASRState(), b"ignored", "es")
    assert "[ASR stub]" in text
    assert confidence is None


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not os.getenv("ASR_REAL_MODEL"),
    reason="downloads ~650 MB of weights; run with ASR_REAL_MODEL=1",
)
async def test_asr_parakeet_transcribes_real_speech():
    """The real backend, against the real pinned artifact.

    Opt-in because it needs the network and 650 MB of disk. It is the only
    check that the third-party export is faithful.

    Both assertions were chosen by feeding the model deliberately broken
    input and reading what came back, rather than by guessing at a failure
    mode:

      - Garbage samples (the file's bytes reinterpreted as float32) return
        120 repeated <unk> tokens with no separators. The content words are
        what catch that; a word count does not, because the whole run is a
        single whitespace-free token.
      - The character bound catches the opposite shape - a decode that starts
        correct and then runs away repeating, which is how the MT stage fails
        when its terminator is missing.

    What does NOT need guarding, measured: amplitude scaling. Feeding raw
    int16 values without dividing by full scale returns the identical
    transcript, because the mel preprocessor normalizes the difference away.
    It only leaves a RuntimeWarning about an overflowing cast.
    """
    reference = "el sistema funciona correctamente"
    engine = ASRService("parakeet", "cpu")

    result = await engine.transcribe(
        ASRState(), _FIXTURE_WAV.read_bytes(), source_language="es"
    )

    text = result.text.lower()
    assert "sistema" in text, text
    assert "funciona" in text, text
    assert len(text) < 4 * len(reference), text
    assert result.confidence is None
    assert result.detected_language == "es"
