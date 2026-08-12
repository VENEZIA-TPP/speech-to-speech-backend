"""Engines are built once, by the lifespan, and never by a getter.

These tests need neither the DB nor the test client fixtures: TestClient(app)
is used only as a way to run the lifespan.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import dependencies as deps
from app.core.config import settings
from app.main import app
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService

THREADS = 16


@pytest.fixture(autouse=True)
def restore_engine_globals(monkeypatch):
    """ws_client runs the lifespan too, so these globals are process-wide state.

    monkeypatch.setattr snapshots them and puts them back after each test.
    """
    monkeypatch.setattr(deps, "_asr_service", deps._asr_service)
    monkeypatch.setattr(deps, "_mt_service", deps._mt_service)
    monkeypatch.setattr(deps, "_tts_service", deps._tts_service)


def _counting(cls, calls, key):
    """A plain subclass is fine for a test double: it only counts __init__.

    A real backend must be a @dataclass(frozen=True) with __slots__ = ()
    instead, because a plain subclass gets a __dict__ back and loses the
    barrier against per-session state on the engine. A double that only
    overrides methods never stores anything, so it does not need it.
    """

    class Counting(cls):
        def __init__(self, *args, **kwargs):
            calls[key] += 1
            super().__init__(*args, **kwargs)

    return Counting


def _patch_counting_engines(monkeypatch):
    calls = {"asr": 0, "mt": 0, "tts": 0}
    monkeypatch.setattr(deps, "ASRService", _counting(ASRService, calls, "asr"))
    monkeypatch.setattr(deps, "MTService", _counting(MTService, calls, "mt"))
    monkeypatch.setattr(deps, "TTSService", _counting(TTSService, calls, "tts"))
    return calls


def _call_from_threads(fn):
    """Call fn() from THREADS threads at once; return results or exceptions."""
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(fn) for _ in range(THREADS)]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as exc:
                out.append(exc)
        return out


def test_engines_built_once(monkeypatch):
    """One lifespan, N concurrent readers, one construction per engine.

    The threads are the C3 scenario, not the assertion: today the getters
    are pure reads, so this cannot race. It stays as the reproducer for
    the day someone puts construction back in a getter.
    """
    calls = _patch_counting_engines(monkeypatch)
    # Distinct per-service values: with config.py's shared "stub" default,
    # a mis-paired init_engines() (e.g. MT built with ASR_MODEL) would pass
    # the model_name assertions below undetected.
    #
    # ASR and MT have to be names their backend selectors accept: both
    # validate model_name at construction, so an invented value kills the
    # lifespan before this test gets to assert anything. With the two of
    # them pinned to "stub", telling an ASR/MT swap apart rides on the
    # per-service devices below; "tts-z" stays invented (and distinct) until
    # the TTS backend validates too.
    monkeypatch.setattr(settings, "ASR_MODEL", "stub")
    monkeypatch.setattr(settings, "MT_MODEL", "stub")
    monkeypatch.setattr(settings, "TTS_MODEL", "tts-z")
    # Distinct per-service devices too: ASR_DEVICE/MT_DEVICE/TTS_DEVICE all
    # default to "cpu", so a mis-paired init_engines() (e.g. MT built with
    # TTS_DEVICE) would pass the model_name-only assertions undetected.
    monkeypatch.setattr(settings, "ASR_DEVICE", "dev-a")
    monkeypatch.setattr(settings, "MT_DEVICE", "dev-b")
    monkeypatch.setattr(settings, "TTS_DEVICE", "dev-c")

    with TestClient(app):
        assert calls == {"asr": 1, "mt": 1, "tts": 1}
        assert deps.get_asr_service().model_name == settings.ASR_MODEL
        assert deps.get_mt_service().model_name == settings.MT_MODEL
        assert deps.get_tts_service().model_name == settings.TTS_MODEL
        assert deps.get_asr_service().device == settings.ASR_DEVICE
        assert deps.get_mt_service().device == settings.MT_DEVICE
        assert deps.get_tts_service().device == settings.TTS_DEVICE

        for getter in (
            deps.get_asr_service,
            deps.get_mt_service,
            deps.get_tts_service,
        ):
            results = _call_from_threads(getter)
            assert len(results) == THREADS
            # Same object for everyone: no thread built anything of its own.
            assert all(r is results[0] for r in results)

    assert calls == {"asr": 1, "mt": 1, "tts": 1}


def test_getters_do_not_construct_without_lifespan(monkeypatch):
    """Without the lifespan there is nothing to read, and the getter says so.

    This is the assertion that kills a reintroduced lazy `if is None: build`:
    test_engines_built_once alone would stay green with one, because the
    lifespan would still be the one doing the single construction.
    """
    calls = _patch_counting_engines(monkeypatch)
    monkeypatch.setattr(deps, "_asr_service", None)
    monkeypatch.setattr(deps, "_mt_service", None)
    monkeypatch.setattr(deps, "_tts_service", None)
    # The segmenter is built by the same lifespan and read by the same kind of
    # getter, so it is subject to the same rule: read, never construct.
    monkeypatch.setattr(deps, "_segmenter", None)

    for getter in (
        deps.get_asr_service,
        deps.get_mt_service,
        deps.get_tts_service,
        deps.get_segmenter,
    ):
        results = _call_from_threads(getter)
        assert len(results) == THREADS
        for r in results:
            assert isinstance(r, RuntimeError)
            assert "lifespan" in str(r)

    assert calls == {"asr": 0, "mt": 0, "tts": 0}


def test_startup_fails_fast_when_an_engine_cannot_be_built(monkeypatch):
    """Missing weights must kill the process at boot, not the first request."""

    class BrokenTTSService(TTSService):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights missing")

    monkeypatch.setattr(deps, "TTSService", BrokenTTSService)

    entered = False
    with pytest.raises(RuntimeError, match="weights missing"):
        with TestClient(app):
            entered = True
    assert not entered, "startup must fail before the app body runs"


def test_startup_fails_fast_when_the_vad_weights_are_missing(monkeypatch):
    """Same rule for the segmenter, and a likelier failure than a missing GPU.

    Its weights are a file vendored in the tree, so this breaks on a bad
    checkout or a packaging step that drops non-Python files - and it must
    break at boot, not on the first chunk of some user's session.
    """

    def missing():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(deps, "build_segmenter", missing)

    entered = False
    with pytest.raises(RuntimeError, match="weights missing"):
        with TestClient(app):
            entered = True
    assert not entered, "startup must fail before the app body runs"
