import io
import wave
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from app.main import app
from app.db.base import Base
from app.db.session import get_session
from app.dependencies import get_segmenter
from app.pipeline.vad import (
    FRAME_SAMPLES,
    VoiceSegmenter,
    build_segmenter,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# StaticPool
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionLocal = async_sessionmaker(
    test_engine, expire_on_commit=False, class_=AsyncSession
)


async def override_get_session():
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def cleanup_engine():
    yield
    import asyncio

    asyncio.run(test_engine.dispose())


@pytest_asyncio.fixture(scope="function")
async def db_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_segmenter] = pinned_segmenter

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


_FIXTURE_WAV = Path(__file__).parent / "fixtures" / "es_sistema.wav"

# Segmentación con parámetros fijos para todo lo que corra contra la app.
#
# `build_segmenter()` lee la configuración, y `VAD_MIN_SILENCE_MS` es un dial
# que se espera que cada quien ajuste en su propio `.env` - que está
# gitignoreado y es local. Sin esto, subirlo a 500 no rompe la suite: la
# **cuelga**. Los tests de WebSocket esperan un segmento que nunca llega, y no
# hay pytest-timeout configurado, así que el bloqueo es para siempre y no dice
# de qué se trata. Cambiar un valor soportado tiene que ser inofensivo para los
# tests.
_TEST_VAD = dict(
    threshold=0.5,
    min_silence_ms=300,
    min_speech_ms=384,
    speech_pad_ms=300,
    max_speech_ms=8000,
    lookback_ms=400,
)


def pinned_segmenter() -> VoiceSegmenter:
    base = build_segmenter()
    rate = base.sample_rate

    def frames(ms: int) -> int:
        return round(ms * rate / 1000 / FRAME_SAMPLES)

    return VoiceSegmenter(
        session=base.session,
        sample_rate=rate,
        threshold=_TEST_VAD["threshold"],
        min_silence_frames=frames(_TEST_VAD["min_silence_ms"]),
        min_speech_frames=frames(_TEST_VAD["min_speech_ms"]),
        pad_frames=frames(_TEST_VAD["speech_pad_ms"]),
        max_frames=frames(_TEST_VAD["max_speech_ms"]),
        lookback_frames=frames(_TEST_VAD["lookback_ms"]),
    )


def speech_chunk(trailing_silence_ms: int = 600) -> bytes:
    """One chunk of real speech that closes exactly one segment.

    Tests used to send b"fake_audio_chunk", which worked while every chunk
    produced a segment regardless of its content. The pipeline now runs on what
    the VAD decides is speech, and the VAD does not fire on synthetic audio -
    tones and noise score below 0.03 against a threshold of 0.5 - so a chunk
    that is not real speech produces no segment and nothing to assert on.

    The trailing silence is what closes the segment: without it the speech is
    still open when the test reads, and only the end-of-session flush would
    emit it. It defaults to comfortably more than VAD_MIN_SILENCE_MS.
    """
    with wave.open(str(_FIXTURE_WAV), "rb") as source:
        frames = source.readframes(source.getnframes())
        rate = source.getframerate()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)
        out.writeframes(b"\x00\x00" * int(rate * trailing_silence_ms / 1000))
    return buffer.getvalue()


@pytest.fixture(scope="session")
def speech() -> bytes:
    return speech_chunk()


async def _run_ddl(fn):
    async with test_engine.begin() as conn:
        await conn.run_sync(fn)


@pytest.fixture(scope="function")
def ws_client():
    """Sync TestClient for WebSocket tests (httpx AsyncClient can't speak WS)."""
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_segmenter] = pinned_segmenter

    try:
        with TestClient(app) as tc:
            # El DDL corre en el portal del TestClient, no en un asyncio.run()
            # aparte: un asyncio.run() abre y cierra su propio loop, y create_all/
            # drop_all terminaban pisando la conexion que el handler de WS seguia
            # usando en el loop del TestClient - de ahi el "no active connection"
            # intermitente. Con tc.portal.call() todo el DDL y todo el trafico de
            # la sesion viven en el mismo loop.
            tc.portal.call(_run_ddl, Base.metadata.create_all)
            try:
                yield tc
            finally:
                tc.portal.call(_run_ddl, Base.metadata.drop_all)
                # Belt and braces: dejar el engine sin conexion cacheada para que el
                # proximo test arranque un checkout limpio. No es lo que arregla el
                # "no active connection" intermitente - eso lo arreglan el DDL sobre
                # el portal y el cierre explicito del WS del lado del cliente - pero
                # tampoco cuesta nada y acota el estado que un test le deja al
                # siguiente.
                tc.portal.call(test_engine.dispose)
    finally:
        app.dependency_overrides.clear()
