"""
Pipeline tests - covers stub ASR/MT/TTS services and the full pipeline flow.

When real models are integrated:
  - Replace stub assertions with actual expected outputs.
  - Add latency threshold assertions (e.g. total_processing_time_ms < 5000).
  - Add BLEU score evaluation against a reference translation corpus.
"""

import hashlib
import io
import json
import wave

import anyio
import pytest
from httpx import AsyncClient

from app.api.controller import pipeline as pipeline_controller
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


def _read_segment(ws) -> dict:
    """Read one segment's burst of events, keyed by event type.

    Reads by type instead of by position on purpose: the protocol no longer
    promises a fixed number of frames per chunk the client sent. The binary
    frame is read right after the `audio.delta` that announces it and lands
    under the "audio" key. Returns on `segment.metrics` (the segment's last
    event) or on an `error`, so a broken segment fails the test instead of
    hanging it - there is no pytest-timeout configured.
    """
    segment: dict = {}
    while True:
        event = ws.receive_json()
        segment[event["type"]] = event
        if event["type"] == "audio.delta":
            segment["audio"] = ws.receive_bytes()
        if event["type"] in ("segment.metrics", "error"):
            return segment


# E2E - WebSocket protocol (full stub pipeline)
def test_ws_pipeline_full_stub_flow(ws_client):
    """Two chunks in, two segments out, with no lock-step in between.

    Both chunks are sent before a single reply is read: nothing in the
    protocol lets the client assume a response arrives before it may send
    again. Also proves SessionState survives across chunks within one
    connection - it fails if `state = SessionState()` moves inside the
    receive loop in app/api/controller/pipeline.py, because the second
    segment's ASR counter would reset to 1 instead of advancing to 2.
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
        assert ws.receive_json() == {
            "type": "session.created",
            "session_id": session_id,
        }

        ws.send_bytes(b"fake_audio_chunk")
        ws.send_bytes(b"fake_audio_chunk_2")

        first = _read_segment(ws)
        second = _read_segment(ws)

        transcription = first["transcription.completed"]
        assert transcription["segment_index"] == 0
        assert "chunk 1" in transcription["transcript"]
        assert first["translation.completed"]["segment_index"] == 0
        assert first["translation.completed"]["text"]
        assert first["translation.completed"]["target_language"] == "es"

        assert first["audio.delta"]["segment_index"] == 0
        assert first["audio.delta"]["seq"] == 0
        assert first["audio"][:4] == b"RIFF"
        assert first["audio.delta"]["size_bytes"] == len(first["audio"])
        assert first["audio.done"] == {
            "type": "audio.done",
            "segment_index": 0,
            "watermarked": True,
            "watermark_method": first["audio.done"]["watermark_method"],
        }
        assert first["audio.done"]["watermark_method"]

        metrics = first["segment.metrics"]
        assert metrics["segment_index"] == 0
        assert metrics["asr_ms"] >= 0
        assert metrics["mt_ms"] >= 0
        assert metrics["tts_ms"] >= 0
        assert metrics["e2e_ms"] >= 0

        # Raw audio rides in its own frame, never inside any JSON event.
        assert all(
            "synthesized_audio" not in event
            for event in first.values()
            if isinstance(event, dict)
        )

        assert "chunk 2" in second["transcription.completed"]["transcript"]
        assert second["segment.metrics"]["segment_index"] == 1
        assert second["audio.delta"]["size_bytes"] == len(second["audio"])

        ws.send_text(json.dumps({"type": "input_audio.commit"}))
        assert ws.receive_json() == {
            "type": "session.completed",
            "session_id": session_id,
            "total_segments": 2,
        }


def test_ws_invalid_control_frame_does_not_kill_stream(ws_client):
    """Un frame de control mal formado se responde, no se descarta ni cierra."""
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"

        ws.send_text("not json at all")
        assert ws.receive_json() == {
            "type": "error",
            "code": "invalid_event",
            "message": "expected a JSON object with a `type` field",
            "segment_index": None,
        }

        ws.send_text(json.dumps({"type": "session.update"}))
        unknown = ws.receive_json()
        assert unknown["code"] == "invalid_event"
        assert "session.update" in unknown["message"]

        # The audio stream survives two bad control frames.
        ws.send_bytes(b"fake_audio_chunk")
        segment = _read_segment(ws)
        assert segment["audio.done"]["watermarked"] is True

        # Drive the session to a decided terminal status before the `with`
        # block exits: complete_session() writes before session.completed is
        # sent, so reading that event is a genuine sync point, unlike letting
        # the harness's own teardown race the connection closed.
        ws.send_text(json.dumps({"type": "input_audio.commit"}))
        assert ws.receive_json()["type"] == "session.completed"


def test_ws_audio_done_no_watermark_claim_without_audio(ws_client):
    """A segment with no audio must not claim a watermark either.

    TTSResult.audio is normally a WatermarkedAudio, whose __post_init__
    rejects empty data - the stub can never produce this case by itself.
    This overrides synthesize() (not just the _synthesize()/_apply_watermark()
    hooks that type guarantee runs through) with a fake audio object to
    simulate what a TTS backend returning nothing would otherwise let slip
    through as an unearned watermark claim, and checks the controller does
    not repeat that claim over the wire - audio.done still closes the
    segment, but with no watermark and no binary frame behind it.
    """
    from app.dependencies import get_tts_service
    from app.main import app
    from app.services.tts_service import TTSResult

    class _EmptyAudio:
        data = b""
        method = "stub-metadata-tag"

    class SilentTTSService(TTSService):
        async def synthesize(self, state, text, language):
            return TTSResult(audio=_EmptyAudio(), processing_time_ms=0)

    app.dependency_overrides[get_tts_service] = lambda: SilentTTSService("stub", "cpu")

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        segment = _read_segment(ws)

        assert "audio.delta" not in segment
        assert "audio" not in segment
        assert segment["audio.done"] == {
            "type": "audio.done",
            "segment_index": 0,
            "watermarked": False,
            "watermark_method": None,
        }

        # Same sync reasoning as test_ws_invalid_control_frame_does_not_kill_stream:
        # drive to a decided terminal status instead of letting the harness
        # teardown race the still-unwritten completion status.
        ws.send_text(json.dumps({"type": "input_audio.commit"}))
        assert ws.receive_json()["type"] == "session.completed"


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
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "pipeline_failed"
        assert msg["segment_index"] == 0
        assert "Pipeline error" in msg["message"]
        assert ws.receive()["type"] == "websocket.close"

    assert (
        ws_client.get(f"/sessions/{session_id}", headers=headers).json()["status"]
        == "failed"
    )


def test_ws_pipeline_value_error_marks_session_failed(ws_client):
    """A ValueError out of process_audio_chunk() must fail the session too.

    Same wire label (`pipeline_failed`) as the RuntimeError path covered by
    test_ws_pipeline_error_marks_session_failed, but ValueError is caught by
    its own except clause - this guards that clause marks the session FAILED
    just like its sibling, instead of leaving it for the finally block to
    mark COMPLETED.
    """
    from app.dependencies import get_asr_service
    from app.main import app

    class BrokenASRService(ASRService):
        def _transcribe(self, state, audio_bytes, language):
            raise ValueError("boom")

    app.dependency_overrides[get_asr_service] = lambda: BrokenASRService("stub", "cpu")

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "pipeline_failed"
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
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Pipeline error" in msg["message"]
        assert "WatermarkedAudio" in msg["message"]
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
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        _read_segment(ws)
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
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        assert "Pipeline error" in ws.receive_json()["message"]
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
        # this is the regression guard for that.
        assert ws.accepted_subprotocol == created["ws_token"]
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"fake_audio_chunk")
        assert _read_segment(ws)["segment.metrics"]["segment_index"] == 0
        ws.send_text(json.dumps({"type": "input_audio.commit"}))
        assert ws.receive_json()["type"] == "session.completed"


def test_ws_rejects_oversized_frame(ws_client):
    from app.core.config import settings

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()
    headers = {"Authorization": f"Bearer {created['ws_token']}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_bytes(b"\x00" * (settings.MAX_AUDIO_FRAME_BYTES + 1))
        assert ws.receive()["code"] == 1009  # RFC 6455: message too big

    # Regression guard for the status write ordering: this must land
    # COMPLETED, not be left ACTIVE for the finally block to race against a
    # concurrent shutdown.
    assert (
        ws_client.get(f"/sessions/{created['id']}", headers=headers).json()["status"]
        == "completed"
    )


def test_ws_rejects_oversized_text_frame(ws_client):
    """A text control frame has the same size cap as a binary chunk.

    Unlike the old protocol, which only ever compared a text frame to a
    constant, this one parses it as JSON - an unbounded text frame would
    reach json.loads() bounded only by uvicorn's much larger default.
    """
    from app.core.config import settings

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()
    headers = {"Authorization": f"Bearer {created['ws_token']}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_text("x" * (settings.MAX_AUDIO_FRAME_BYTES + 1))
        assert ws.receive()["code"] == 1009  # RFC 6455: message too big

    assert (
        ws_client.get(f"/sessions/{created['id']}", headers=headers).json()["status"]
        == "completed"
    )


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
                assert ws.receive_json()["type"] == "session.created"
                ws.send_bytes(b"fake_audio_chunk")
                msg = ws.receive_json()
                assert msg["type"] == "error"
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


def test_ws_event_wire_shape():
    """El `type` de cada evento es parte del contrato, no un detalle interno."""
    from app.schemas.events import (
        AudioDelta,
        AudioDone,
        ErrorEvent,
        SegmentMetrics,
        SessionCompleted,
        SessionCreated,
        TranscriptionCompleted,
        TranslationCompleted,
    )

    assert SessionCreated(session_id=7).model_dump() == {
        "type": "session.created",
        "session_id": 7,
    }
    assert SessionCompleted(session_id=7, total_segments=2).model_dump() == {
        "type": "session.completed",
        "session_id": 7,
        "total_segments": 2,
    }
    assert AudioDelta(segment_index=0, seq=0, size_bytes=44).model_dump() == {
        "type": "audio.delta",
        "segment_index": 0,
        "seq": 0,
        "size_bytes": 44,
    }
    assert ErrorEvent(code="invalid_event", message="boom").model_dump() == {
        "type": "error",
        "code": "invalid_event",
        "message": "boom",
        "segment_index": None,
    }
    assert [
        e(segment_index=0, **kwargs).model_dump()["type"]
        for e, kwargs in [
            (TranscriptionCompleted, {"transcript": "hola"}),
            (TranslationCompleted, {"text": "hi", "target_language": "en"}),
            (AudioDone, {"watermarked": True}),
            (SegmentMetrics, {}),
        ]
    ] == [
        "transcription.completed",
        "translation.completed",
        "audio.done",
        "segment.metrics",
    ]


def test_metrics_cover_full_window(ws_client, monkeypatch):
    """e2e_ms cubre la ventana entera del segmento, no solo el pipeline.

    La asercion obvia -- e2e_ms >= asr+mt+tts -- no prueba nada por si sola:
    el numero viejo se medía adentro de process_audio_chunk(), que ya envuelve
    a las tres etapas, asi que pasaba igual antes del cambio. Lo que discrimina
    es meter demora en la parte de la ventana que solo cubre la medicion nueva:
    entre el return del pipeline y el send_bytes del audio. Con _send() dormido,
    tres envios caen adentro del reloj (transcription.completed,
    translation.completed y el audio.delta que anuncia el binario); con la
    medicion vieja el reloj ya habia cortado antes de los tres y e2e_ms se
    quedaba en un digito de milisegundos.
    """
    delay_s = 0.04
    delay_ms = int(delay_s * 1000)
    original_send = pipeline_controller._send

    async def slow_send(websocket, event):
        await anyio.sleep(delay_s)
        await original_send(websocket, event)

    monkeypatch.setattr(pipeline_controller, "_send", slow_send)

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert created.status_code == 201
    session_id = created.json()["id"]
    token = created.json()["ws_token"]

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", subprotocols=[token]
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"

        ws.send_bytes(b"fake_audio_chunk")
        segment = _read_segment(ws)

        # Cierre desde el cliente y espera del close del servidor: el with no
        # sirve para eso, manda su propio close seguido de un cancel casi
        # inmediato.
        ws.close()
        assert ws.receive()["type"] == "websocket.close"

    metrics = segment["segment.metrics"]
    stages = metrics["asr_ms"] + metrics["mt_ms"] + metrics["tts_ms"]
    assert metrics["e2e_ms"] >= stages
    # Tres _send() demorados caen adentro de la ventana; se exigen dos para no
    # atarse a la precision del timer del loop. Con la medicion vieja e2e_ms
    # era de un digito, asi que dos ya discrimina de sobra.
    assert metrics["e2e_ms"] >= 2 * delay_ms
