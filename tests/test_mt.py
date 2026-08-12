"""Real translation, and the two failures that look like success.

The first is a code trap recorded before the backend existed: a constructor
that loads es->en while _translate() ignores its arguments. Everything passes,
the demo works, and the system translates es->en no matter what pair it is
asked for - discovered the day someone creates an en->es session.

The second was found while integrating: the source token list needs a trailing
</s>, and omitting it does not raise. The decoder simply never emits EOS and
degenerates into repeated fragments after a correct-looking first clause. An
assertion that a content word appears in the output passes on that version,
which is why the real-model test also bounds the output length.
"""

import os

import pytest

from app.core.config import settings
from app.pipeline.contracts import MTState
from app.services import mt_service as mt_module
from app.services.mt_service import MTService


def _recording_loader(loaded):
    """Stands in for _load_translator, and records what it was asked to load.

    The object it returns reports the pair it belongs to, so a translation can
    be traced back to the model that produced it.
    """

    class Fake:
        def __init__(self, pair):
            self.pair = pair

        def translate(self, text):
            return f"{self.pair[0]}->{self.pair[1]}:{text}"

    def load(pair, device):
        loaded.append(pair)
        return Fake(pair)

    return load


def test_mt_caches_by_language_pair(monkeypatch):
    """One model per configured pair, loaded once, and each pair used for itself.

    The pairs are pinned here rather than read from settings on purpose:
    SUPPORTED_LANGUAGE_PAIRS is a dial every developer is expected to change in
    their own .env, which is gitignored. A test that reads it goes red on one
    machine and green on another. Same reason pinned_segmenter() exists.
    """
    monkeypatch.setattr(settings, "SUPPORTED_LANGUAGE_PAIRS", "es-en,en-es")
    loaded = []
    monkeypatch.setattr(mt_module, "_load_translator", _recording_loader(loaded))

    engine = MTService("opus-mt", "cpu")

    # One load per configured pair, for exactly the configured pairs - which is
    # also the assertion that the settings -> object wiring is real.
    assert sorted(loaded) == [("en", "es"), ("es", "en")]

    state = MTState()
    to_english = engine._translate(state, "hola", "es", "en")
    to_spanish = engine._translate(state, "hello", "en", "es")
    engine._translate(state, "hola de nuevo", "es", "en")
    engine._translate(state, "hello again", "en", "es")

    # Four translations later, still two loads: the cache is a cache.
    assert sorted(loaded) == [("en", "es"), ("es", "en")]

    # And this is the assertion that kills the trap. A _translate() that
    # ignored its arguments would route both calls to whichever model was
    # loaded last and still satisfy every other assertion in this test.
    assert to_english == "es->en:hola"
    assert to_spanish == "en->es:hello"


def test_unknown_mt_model_raises_listing_options():
    """A typo must fail at startup, not translate with placeholders forever."""
    with pytest.raises(ValueError) as exc:
        MTService("opusmt", "cpu")

    message = str(exc.value)
    assert "opusmt" in message
    for backend in ("stub", "opus-mt"):
        assert backend in message, f"the error does not offer {backend!r}"


def test_unknown_language_pair_is_explicit(monkeypatch):
    """A pair with no model names itself and the configured list.

    This is what a session whose detected language differs from its declared
    one will hit once ASR is real. It has to arrive as a sentence someone can
    act on, not as a bare KeyError out of a dict lookup.
    """
    monkeypatch.setattr(settings, "SUPPORTED_LANGUAGE_PAIRS", "es-en")
    monkeypatch.setattr(mt_module, "_load_translator", _recording_loader([]))
    engine = MTService("opus-mt", "cpu")

    with pytest.raises(ValueError) as exc:
        engine._translate(MTState(), "bonjour", "fr", "en")

    message = str(exc.value)
    assert "fr->en" in message
    assert "es->en" in message, "the error does not say what IS loaded"


def test_stub_backend_loads_nothing(monkeypatch):
    """MT_MODEL=stub must not touch the loader, whatever the pairs say.

    Without this, moving the stub check after the loop would keep every other
    test green while making the whole suite download 311 MB of weights.
    """

    def explode(pair, device):
        raise AssertionError(f"the stub backend loaded a model for {pair}")

    monkeypatch.setattr(settings, "SUPPORTED_LANGUAGE_PAIRS", "es-en,en-es")
    monkeypatch.setattr(mt_module, "_load_translator", explode)

    engine = MTService("stub", "cpu")
    assert "[MT stub]" in engine._translate(MTState(), "hola", "es", "en")


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not os.getenv("MT_REAL_MODEL"),
    reason="downloads ~311 MB of weights; run with MT_REAL_MODEL=1",
)
def test_mt_opus_translates_both_directions(monkeypatch):
    """The real backend, against the real artifacts, in both directions.

    Opt-in because it needs the network and 311 MB of disk. It is the only
    check that the third-party conversion is faithful, and the only one that
    would catch a missing </s>: the length bound is what separates a
    translation from a decoder that never stopped.
    """
    monkeypatch.setattr(settings, "SUPPORTED_LANGUAGE_PAIRS", "es-en,en-es")
    engine = MTService("opus-mt", "cpu")
    state = MTState()

    source_es = "el sistema de traduccion funciona correctamente"
    to_english = engine._translate(state, source_es, "es", "en").lower()
    assert "translation" in to_english
    assert "works" in to_english or "working" in to_english
    assert len(to_english.split()) < 2 * len(source_es.split()), to_english

    source_en = "I need you to send me the report before Friday"
    to_spanish = engine._translate(state, source_en, "en", "es").lower()
    assert "informe" in to_spanish
    assert "viernes" in to_spanish
    assert len(to_spanish.split()) < 2 * len(source_en.split()), to_spanish

    assert state.segments_seen == 2
