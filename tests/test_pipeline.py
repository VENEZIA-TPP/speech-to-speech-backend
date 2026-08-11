"""
Pipeline tests - covers stub ASR/MT/TTS services and the full pipeline flow.

When real models are integrated:
  - Replace stub assertions with actual expected outputs.
  - Add latency threshold assertions (e.g. total_processing_time_ms < 5000).
  - Add BLEU score evaluation against a reference translation corpus.
"""

import hashlib
import io
import wave

import pytest
from httpx import AsyncClient

from app.pipeline.contracts import ASRState, MTState, SessionState, TTSState
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService


# Unit tests - stub services
async def test_asr_stub_returns_result():
    asr = ASRService("stub", "cpu")
    result = await asr.transcribe(ASRState(), b"fake_audio", source_language="en")

    assert result.text
    assert result.detected_language == "en"
    assert result.confidence is not None
    assert result.processing_time_ms >= 0


async def test_asr_stub_auto_detects_language():
    asr = ASRService("stub", "cpu")
    result = await asr.transcribe(ASRState(), b"fake_audio")

    assert result.detected_language is not None


async def test_mt_stub_returns_result():
    mt = MTService("stub", "cpu")
    result = await mt.translate(MTState(), "Hello world", "en", "es")

    assert result.translated_text
    assert result.source_language == "en"
    assert result.target_language == "es"
    assert result.processing_time_ms >= 0


async def test_mt_stub_preserves_language_codes():
    mt = MTService("stub", "cpu")
    result = await mt.translate(MTState(), "Hola mundo", "es", "en")

    assert result.source_language == "es"
    assert result.target_language == "en"


async def test_tts_stub_returns_result():
    tts = TTSService("stub", "cpu")
    result = await tts.synthesize(TTSState(), "Hola mundo", "es")

    assert result.audio.data
    assert result.audio.data[:4] == b"RIFF"
    with wave.open(io.BytesIO(result.audio.data)) as wav:
        assert wav.getframerate() == result.audio.sample_rate
    assert result.audio.duration_ms > 0
    assert result.processing_time_ms >= 0


async def test_tts_stub_output_is_watermark_tagged():
    """Synthesized output must carry the watermark/tag."""
    tts = TTSService("stub", "cpu")
    result = await tts.synthesize(TTSState(), "Hola mundo", "es")

    assert result.audio.method
    assert result.audio.sha256 == hashlib.sha256(result.audio.data).hexdigest()


# Integration tests - pipeline via HTTP + DB (stub services)
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "asr" in data["services"]
    assert "mt" in data["services"]
    assert "tts" in data["services"]


async def test_pipeline_processes_chunk(db_session):
    from app.services.translation_pipeline_service import TranslationPipelineService
    from app.repositories.translation_session_repository import (
        SQLAlchemyTranslationSessionRepository,
    )
    from app.repositories.transcription_repository import (
        SQLAlchemyTranscriptionRepository,
    )
    from app.repositories.translation_repository import SQLAlchemyTranslationRepository
    from app.schemas.translation_session import TranslationSessionCreate

    session_repo = SQLAlchemyTranslationSessionRepository(db_session)
    transcription_repo = SQLAlchemyTranscriptionRepository(db_session)
    translation_repo = SQLAlchemyTranslationRepository(db_session)

    session = await session_repo.create(
        TranslationSessionCreate(source_language="en", target_language="es")
    )

    pipeline = TranslationPipelineService(
        session_repo=session_repo,
        transcription_repo=transcription_repo,
        translation_repo=translation_repo,
        asr_service=ASRService("stub", "cpu"),
        mt_service=MTService("stub", "cpu"),
        tts_service=TTSService("stub", "cpu"),
    )

    result = await pipeline.process_audio_chunk(
        SessionState(),
        session_id=session.id,
        audio_bytes=b"fake_audio_chunk",
        chunk_index=0,
    )

    assert result.chunk_index == 0
    assert result.original_text
    assert result.translated_text
    assert result.source_language == "en"
    assert result.target_language == "es"
    assert (
        result.total_processing_time_ms is not None
        and result.total_processing_time_ms >= 0
    )
    assert (
        result.tts_processing_time_ms is not None and result.tts_processing_time_ms >= 0
    )
    assert result.watermarked is True
    assert result.watermark_method
    assert result.synthesized_audio[:4] == b"RIFF"
    assert result.synthesized_audio_size_bytes == len(result.synthesized_audio)

    transcriptions = await transcription_repo.get_by_session(session.id)
    assert len(transcriptions) == 1
    assert transcriptions[0].chunk_index == 0

    translations = await translation_repo.get_by_session(session.id)
    assert len(translations) == 1


# E2E - WebSocket protocol (full stub pipeline)
def test_ws_pipeline_full_stub_flow(ws_client):
    """Also proves SessionState survives across chunks within one connection.

    Passes falsely if `state = SessionState()` moves inside the receive loop
    in app/api/controller/pipeline.py - the second chunk's ASR counter would
    reset to 1 instead of advancing to 2.
    """
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert created.status_code == 201
    session_id = created.json()["id"]

    token = created.json()["ws_token"]
    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")

        msg = ws.receive_json()
        assert msg["chunk_index"] == 0
        assert msg["original_text"]
        assert "chunk 1" in msg["original_text"]
        assert msg["translated_text"]
        assert msg["asr_processing_time_ms"] >= 0
        assert msg["mt_processing_time_ms"] >= 0
        assert msg["tts_processing_time_ms"] >= 0
        assert msg["watermarked"] is True
        assert msg["synthesized_audio_size_bytes"] > 0
        assert "synthesized_audio" not in msg

        audio = ws.receive_bytes()
        assert len(audio) == msg["synthesized_audio_size_bytes"]
        assert audio[:4] == b"RIFF"

        ws.send_bytes(b"fake_audio_chunk_2")

        msg2 = ws.receive_json()
        assert msg2["chunk_index"] == 1
        assert "chunk 2" in msg2["original_text"]

        audio2 = ws.receive_bytes()
        assert len(audio2) == msg2["synthesized_audio_size_bytes"]

        ws.send_text("END")
        done = ws.receive_json()
        assert done["status"] == "completed"
        assert done["total_chunks"] == 2


def test_ws_unknown_session_is_rejected(ws_client):
    """Una sesión inexistente se rechaza en la autorización, antes del pipeline.

    Manda un frame primero: si el chequeo de auth se regresiona, el servidor
    responde con algo (en vez de bloquear en receive()) y el test falla limpio
    en vez de colgarse - no hay pytest-timeout configurado.
    """
    with ws_client.websocket_connect(
        "/pipeline/ws/99999", subprotocols=["whatever"]
    ) as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401


def test_ws_pipeline_error_marks_session_failed(ws_client):
    """Mid-pipeline failure: one JSON error frame, no binary frame, session FAILED."""
    from app.dependencies import get_tts_service
    from app.main import app

    class BrokenTTSService(TTSService):
        def _synthesize(self, state, text, language):
            raise RuntimeError("boom")

    app.dependency_overrides[get_tts_service] = lambda: BrokenTTSService("stub", "cpu")

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert "Pipeline error" in msg["error"]
        assert ws.receive()["type"] == "websocket.close"

    assert (
        ws_client.get(f"/sessions/{session_id}", headers=headers).json()["status"]
        == "failed"
    )


def test_raw_audio_never_reaches_socket(ws_client):
    """A TTS whose watermark hook hands back the raw audio must fail closed.

    Rule #6 used to depend on the order of two lines inside synthesize(). Now
    it depends on the type: _synthesize() can only produce RawAudio, and the
    pipeline only accepts WatermarkedAudio, so unmarked audio cannot reach the
    socket even if the hook is overridden to do nothing. The client gets one
    JSON error frame, NO binary frame, and the session lands FAILED.
    """
    from app.dependencies import get_tts_service
    from app.main import app

    class UnmarkedTTSService(TTSService):
        def _apply_watermark(self, raw):
            return raw  # RawAudio, not WatermarkedAudio

    app.dependency_overrides[get_tts_service] = lambda: UnmarkedTTSService(
        "stub", "cpu"
    )

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()
    session_id = created["id"]
    token = created["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert "Pipeline error" in msg["error"]
        assert "WatermarkedAudio" in msg["error"]
        # The next frame is the close, never audio.
        assert ws.receive()["type"] == "websocket.close"

    assert (
        ws_client.get(f"/sessions/{session_id}", headers=headers).json()["status"]
        == "failed"
    )


async def test_tts_result_carries_a_watermarked_audio():
    """TTSResult(watermarked=False) with audio inside used to be constructible."""
    from app.pipeline.contracts import TTSState, WatermarkedAudio

    tts = TTSService("stub", "cpu")
    result = await tts.synthesize(TTSState(), "Hola mundo", "es")

    assert isinstance(result.audio, WatermarkedAudio)
    assert result.audio.method
    assert result.audio.data[:4] == b"RIFF"


def test_ws_abrupt_disconnect_does_not_leave_session_active(ws_client):
    """Cerrar el socket sin mandar "END" no puede dejar la fila en ACTIVE."""
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert created.status_code == 201
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        ws.receive_json()
        ws.receive_bytes()
        # Close from the client side ourselves, then wait for the server's own
        # close frame. This forces the test to observe the disconnect branch
        # actually running server-side: letting the `with` block's own
        # teardown do the closing races the handler's pending receive() against
        # an unconditional cancel and passes whether or not that branch exists
        # (verified via mutation - see test-1-report.md).
        ws.close()
        assert ws.receive()["type"] == "websocket.close"

    assert (
        ws_client.get(f"/sessions/{session_id}", headers=headers).json()["status"]
        != "active"
    )


def test_ws_error_then_abrupt_disconnect_keeps_failed(ws_client):
    """Un fallo del pipeline marca FAILED, y la desconexión no puede pisarlo con COMPLETED."""
    from app.dependencies import get_tts_service
    from app.main import app

    class BrokenTTSService(TTSService):
        def _synthesize(self, state, text, language):
            raise RuntimeError("boom")

    app.dependency_overrides[get_tts_service] = lambda: BrokenTTSService("stub", "cpu")

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        assert "Pipeline error" in ws.receive_json()["error"]
        # A diferencia de test_ws_pipeline_error_marks_session_failed, NO se lee
        # el frame de cierre: se sale del bloque directamente.

    assert (
        ws_client.get(f"/sessions/{session_id}", headers=headers).json()["status"]
        == "failed"
    )


def test_ws_rejects_missing_token(ws_client):
    """Manda un frame primero: si el chequeo de auth se regresiona, el servidor
    responde (en vez de bloquear en receive()) y el test falla limpio en vez de
    colgarse - no hay pytest-timeout configurado."""
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]

    with ws_client.websocket_connect(f"/pipeline/ws/{session_id}") as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401


def test_ws_rejects_foreign_session_token(ws_client):
    """El token de la sesión A no abre la sesión B."""
    a = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()
    b = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{b['id']}", subprotocols=[a["ws_token"]]
    ) as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401

    b_headers = {"Authorization": f"Bearer {b['ws_token']}"}
    assert (
        ws_client.get(f"/sessions/{b['id']}", headers=b_headers).json()["status"]
        == "active"
    )


def test_ws_accepts_own_token(ws_client):
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        # A browser that offers a subprotocol and doesn't get one echoed back
        # in the handshake response treats the connection as failed to open -
        # this is the regression guard for that (see fix round 1 report).
        assert ws.accepted_subprotocol == created["ws_token"]
        ws.send_bytes(b"fake_audio_chunk")
        assert ws.receive_json()["chunk_index"] == 0
        ws.receive_bytes()
        ws.send_text("END")
        assert ws.receive_json()["status"] == "completed"


def test_ws_rejects_oversized_frame(ws_client):
    from app.core.config import settings

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        ws.send_bytes(b"\x00" * (settings.MAX_AUDIO_FRAME_BYTES + 1))
        assert ws.receive()["code"] == 1009  # RFC 6455: message too big


def test_ws_duplicate_chunk_mid_pipeline_does_not_leave_session_stuck():
    """A UniqueConstraint violation mid-chunk (e.g. a client reconnecting and
    resending chunk_index=0) must not poison the shared AsyncSession into
    PendingRollbackError for the rest of the request. Without the fix,
    fail_session()/complete_session() both reuse that poisoned session and
    raise in turn, so the handler never decides a terminal status and the
    session is stuck at whatever it was (the exact bug this branch's first
    commit fixed, reopened via a different trigger).

    Seeds the colliding chunk_index=0 row directly via the repositories,
    before the TestClient/WS portal exists, rather than via the ws_client
    fixture + a second real WS connection: StaticPool's single aiosqlite
    connection does not survive a websocket_connect() context's teardown
    cancelling an in-flight statement on it, so driving this through two
    sequential real WS connections is flaky at the test-harness level and
    unrelated to the bug under test.
    """
    import asyncio

    from fastapi.testclient import TestClient

    from app.db.base import Base
    from app.db.session import get_session
    from app.main import app
    from app.repositories.transcription_repository import (
        SQLAlchemyTranscriptionRepository,
    )
    from app.repositories.translation_session_repository import (
        SQLAlchemyTranslationSessionRepository,
    )
    from app.schemas.translation_session import TranslationSessionCreate
    from tests.conftest import TestSessionLocal, _run_ddl, override_get_session

    async def setup():
        await _run_ddl(Base.metadata.create_all)
        async with TestSessionLocal() as session:
            session_repo = SQLAlchemyTranslationSessionRepository(session)
            created = await session_repo.create(
                TranslationSessionCreate(source_language="en", target_language="es")
            )
            transcription_repo = SQLAlchemyTranscriptionRepository(session)
            await transcription_repo.create(
                session_id=created.id,
                chunk_index=0,
                original_text="existing",
                detected_language="en",
                confidence=None,
                asr_processing_time_ms=1,
            )
            return created.id, created.ws_token

    session_id, token = asyncio.run(setup())

    app.dependency_overrides[get_session] = override_get_session
    try:
        with TestClient(app) as tc:
            headers = {"Authorization": f"Bearer {token}"}
            with tc.websocket_connect(
                f"/pipeline/ws/{session_id}", subprotocols=[token]
            ) as ws:
                ws.send_bytes(b"fake_audio_chunk")
                msg = ws.receive_json()
                assert "error" in msg
                assert ws.receive()["type"] == "websocket.close"

            assert (
                tc.get(f"/sessions/{session_id}", headers=headers).json()["status"]
                == "failed"
            )
    finally:
        app.dependency_overrides.clear()
        asyncio.run(_run_ddl(Base.metadata.drop_all))


async def test_authorize_session_token_rejects_missing_and_wrong_and_none(db_session):
    from app.repositories.translation_session_repository import (
        SQLAlchemyTranslationSessionRepository,
    )
    from app.schemas.translation_session import TranslationSessionCreate
    from app.services.session_auth import authorize_session_token

    session_repo = SQLAlchemyTranslationSessionRepository(db_session)
    session = await session_repo.create(
        TranslationSessionCreate(source_language="en", target_language="es")
    )

    assert (
        await authorize_session_token(session_repo, session.id, session.ws_token)
        is True
    )
    assert (
        await authorize_session_token(session_repo, session.id, "wrong-token") is False
    )
    assert await authorize_session_token(session_repo, session.id, None) is False
    assert (
        await authorize_session_token(session_repo, 999999, session.ws_token) is False
    )
    # Non-ASCII must not crash the constant-time compare (both operands are
    # bytes via .encode(), which never raises on non-ASCII input).
    assert await authorize_session_token(session_repo, session.id, "ñ") is False


async def test_duplicate_chunk_index_rejected(db_session):
    from sqlalchemy.exc import IntegrityError

    from app.repositories.translation_session_repository import (
        SQLAlchemyTranslationSessionRepository,
    )
    from app.repositories.transcription_repository import (
        SQLAlchemyTranscriptionRepository,
    )
    from app.schemas.translation_session import TranslationSessionCreate

    session_repo = SQLAlchemyTranslationSessionRepository(db_session)
    transcription_repo = SQLAlchemyTranscriptionRepository(db_session)

    session = await session_repo.create(
        TranslationSessionCreate(source_language="en", target_language="es")
    )

    await transcription_repo.create(
        session_id=session.id,
        chunk_index=0,
        original_text="first",
        detected_language="en",
        confidence=None,
        asr_processing_time_ms=1,
    )

    with pytest.raises(IntegrityError):
        await transcription_repo.create(
            session_id=session.id,
            chunk_index=0,
            original_text="duplicate",
            detected_language="en",
            confidence=None,
            asr_processing_time_ms=1,
        )

    # The failed INSERT leaves the transaction poisoned; without this rollback
    # the fixture's drop_all fails and the error surfaces as a teardown cascade
    # in unrelated tests.
    await db_session.rollback()
