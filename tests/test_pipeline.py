"""
Pipeline tests - covers stub ASR/MT services and the full pipeline flow.

When real models are integrated:
  - Replace stub assertions with actual expected outputs.
  - Add latency threshold assertions (e.g. total_processing_time_ms < 5000).
  - Add BLEU score evaluation against a reference translation corpus.
"""
import pytest
from httpx import AsyncClient

from app.services.asr_service import ASRService
from app.services.mt_service import MTService

pytestmark = pytest.mark.asyncio


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


# Integration tests - pipeline via HTTP + DB (stub services)
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "asr" in data["services"]
    assert "mt" in data["services"]


async def test_pipeline_processes_chunk(db_session):
    from app.services.translation_pipeline_service import TranslationPipelineService
    from app.repositories.translation_session_repository import SQLAlchemyTranslationSessionRepository
    from app.repositories.transcription_repository import SQLAlchemyTranscriptionRepository
    from app.repositories.translation_repository import SQLAlchemyTranslationRepository
    from app.schemas.translation_session import TranslationSessionCreate

    session_repo = SQLAlchemyTranslationSessionRepository(db_session)
    transcription_repo = SQLAlchemyTranscriptionRepository(db_session)
    translation_repo = SQLAlchemyTranslationRepository(db_session)

    session = await session_repo.create(TranslationSessionCreate(source_language="en", target_language="es"))

    pipeline = TranslationPipelineService(
        session_repo=session_repo,
        transcription_repo=transcription_repo,
        translation_repo=translation_repo,
        asr_service=ASRService(),
        mt_service=MTService(),
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
    assert result.total_processing_time_ms is not None and result.total_processing_time_ms >= 0

    transcriptions = await transcription_repo.get_by_session(session.id)
    assert len(transcriptions) == 1
    assert transcriptions[0].chunk_index == 0

    translations = await translation_repo.get_by_session(session.id)
    assert len(translations) == 1
