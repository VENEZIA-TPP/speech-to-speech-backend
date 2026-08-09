"""
Pipeline tests - covers stub ASR/MT/TTS services and the full pipeline flow.

When real models are integrated:
  - Replace stub assertions with actual expected outputs.
  - Add latency threshold assertions (e.g. total_processing_time_ms < 5000).
  - Add BLEU score evaluation against a reference translation corpus.
"""

import io
import wave

from httpx import AsyncClient

from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService


# Unit tests - stub services
async def test_asr_stub_returns_result():
    asr = ASRService()
    result = await asr.transcribe(b"fake_audio", source_language="en")

    assert result.text
    assert result.detected_language == "en"
    assert result.confidence is not None
    assert result.processing_time_ms >= 0


async def test_asr_stub_auto_detects_language():
    asr = ASRService()
    result = await asr.transcribe(b"fake_audio")

    assert result.detected_language is not None


async def test_mt_stub_returns_result():
    mt = MTService()
    result = await mt.translate("Hello world", "en", "es")

    assert result.translated_text
    assert result.source_language == "en"
    assert result.target_language == "es"
    assert result.processing_time_ms >= 0


async def test_mt_stub_preserves_language_codes():
    mt = MTService()
    result = await mt.translate("Hola mundo", "es", "en")

    assert result.source_language == "es"
    assert result.target_language == "en"


async def test_tts_stub_returns_result():
    tts = TTSService()
    result = await tts.synthesize("Hola mundo", "es")

    assert result.audio_bytes
    assert result.audio_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(result.audio_bytes)) as wav:
        assert wav.getframerate() == result.sample_rate
    assert result.duration_ms > 0
    assert result.processing_time_ms >= 0


async def test_tts_stub_output_is_watermark_tagged():
    """Synthesized output must carry the watermark/tag."""
    tts = TTSService()
    result = await tts.synthesize("Hola mundo", "es")

    assert result.watermarked is True
    assert result.watermark_method


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
        asr_service=ASRService(),
        mt_service=MTService(),
        tts_service=TTSService(),
    )

    result = await pipeline.process_audio_chunk(
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
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    assert created.status_code == 201
    session_id = created.json()["id"]

    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}
    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", headers=headers
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")

        msg = ws.receive_json()
        assert msg["chunk_index"] == 0
        assert msg["original_text"]
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

        ws.send_text("END")
        done = ws.receive_json()
        assert done["status"] == "completed"
        assert done["total_chunks"] == 1


def test_ws_unknown_session_is_rejected(ws_client):
    """Una sesión inexistente se rechaza en la autorización, antes del pipeline.

    Manda un frame primero: si el chequeo de auth se regresiona, el servidor
    responde con algo (en vez de bloquear en receive()) y el test falla limpio
    en vez de colgarse - no hay pytest-timeout configurado.
    """
    headers = {"Authorization": "Bearer whatever"}
    with ws_client.websocket_connect("/pipeline/ws/99999", headers=headers) as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401


def test_ws_pipeline_error_marks_session_failed(ws_client):
    """Mid-pipeline failure: one JSON error frame, no binary frame, session FAILED."""
    from app.dependencies import get_tts_service
    from app.main import app

    class BrokenTTSService(TTSService):
        async def _synthesize(self, text, language, speaker_reference):
            raise RuntimeError("boom")

    app.dependency_overrides[get_tts_service] = lambda: BrokenTTSService()

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", headers=headers
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        msg = ws.receive_json()
        assert "Pipeline error" in msg["error"]
        assert ws.receive()["type"] == "websocket.close"

    assert ws_client.get(f"/sessions/{session_id}").json()["status"] == "failed"


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
        f"/pipeline/ws/{session_id}", headers=headers
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

    assert ws_client.get(f"/sessions/{session_id}").json()["status"] != "active"


def test_ws_error_then_abrupt_disconnect_keeps_failed(ws_client):
    """Un fallo del pipeline marca FAILED, y la desconexión no puede pisarlo con COMPLETED."""
    from app.dependencies import get_tts_service
    from app.main import app

    class BrokenTTSService(TTSService):
        async def _synthesize(self, text, language, speaker_reference):
            raise RuntimeError("boom")

    app.dependency_overrides[get_tts_service] = lambda: BrokenTTSService()

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    )
    session_id = created.json()["id"]
    token = created.json()["ws_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with ws_client.websocket_connect(
        f"/pipeline/ws/{session_id}", headers=headers
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        assert "Pipeline error" in ws.receive_json()["error"]
        # A diferencia de test_ws_pipeline_error_marks_session_failed, NO se lee
        # el frame de cierre: se sale del bloque directamente.

    assert ws_client.get(f"/sessions/{session_id}").json()["status"] == "failed"


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

    headers = {"Authorization": f"Bearer {a['ws_token']}"}
    with ws_client.websocket_connect(f"/pipeline/ws/{b['id']}", headers=headers) as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401

    assert ws_client.get(f"/sessions/{b['id']}").json()["status"] == "active"


def test_ws_rejects_non_ascii_token(ws_client):
    """secrets.compare_digest lanza TypeError con str no-ASCII; encode() lo evita.

    Sin el .encode(), esto explota en un 500/crash para una sesión existente en
    vez del mismo 4401 limpio que un token cualquiera equivocado - reabriendo el
    oráculo de enumeración que esta task existe para cerrar.
    """
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    # HTTP header values are transmitted as raw bytes (ASGI decodes them as
    # latin-1) - httpx's str path rejects non-ASCII outright, so send the
    # header value pre-encoded to reach the server at all.
    headers = {"Authorization": "Bearer ñ".encode("utf-8")}
    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", headers=headers
    ) as ws:
        ws.send_bytes(b"x")
        assert ws.receive()["code"] == 4401


def test_ws_accepts_own_token(ws_client):
    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    headers = {"Authorization": f"Bearer {created['ws_token']}"}
    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", headers=headers
    ) as ws:
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

    headers = {"Authorization": f"Bearer {created['ws_token']}"}
    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", headers=headers
    ) as ws:
        ws.send_bytes(b"\x00" * (settings.MAX_AUDIO_FRAME_BYTES + 1))
        assert ws.receive()["code"] == 1009  # RFC 6455: message too big
