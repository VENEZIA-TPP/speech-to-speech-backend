"""Engines are built once, by the lifespan, and never by a getter (PR 5, C3/C8).

These tests need neither the DB nor the test client fixtures: TestClient(app)
is used only as a way to run the lifespan.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import dependencies as deps
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
    monkeypatch.setattr(deps, "_asr_service", deps._asr_service, raising=False)
    monkeypatch.setattr(deps, "_mt_service", deps._mt_service, raising=False)
    monkeypatch.setattr(deps, "_tts_service", deps._tts_service, raising=False)


def _counting(cls, calls, key):
    """A plain subclass is fine for a test double: it only counts __init__.

    (CLAUDE.md: a real backend must be @dataclass(frozen=True) with
    __slots__ = (), but a plain subclass is the documented shape for doubles.)
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
            except Exception as exc:  # noqa: BLE001 - the test inspects the type
                out.append(exc)
        return out


def test_engines_built_once(monkeypatch):
    """One lifespan, N concurrent readers, one construction per engine."""
    calls = _patch_counting_engines(monkeypatch)

    with TestClient(app):
        assert calls == {"asr": 1, "mt": 1, "tts": 1}

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

    for getter in (deps.get_asr_service, deps.get_mt_service, deps.get_tts_service):
        results = _call_from_threads(getter)
        assert len(results) == THREADS
        for r in results:
            assert isinstance(r, RuntimeError)
            assert "lifespan" in str(r)

    assert calls == {"asr": 0, "mt": 0, "tts": 0}
