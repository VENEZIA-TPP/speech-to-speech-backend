from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.repositories.interfaces.translation_session_repository import (
    ITranslationSessionRepository,
)
from app.repositories.interfaces.transcription_repository import (
    ITranscriptionRepository,
)
from app.repositories.interfaces.translation_repository import ITranslationRepository
from app.repositories.translation_session_repository import (
    SQLAlchemyTranslationSessionRepository,
)
from app.repositories.transcription_repository import SQLAlchemyTranscriptionRepository
from app.repositories.translation_repository import SQLAlchemyTranslationRepository
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.session_service import SessionService
from app.services.translation_pipeline_service import TranslationPipelineService
from app.services.tts_service import TTSService


# Repositories
def get_session_repository(
    db: AsyncSession = Depends(get_session),
) -> ITranslationSessionRepository:
    return SQLAlchemyTranslationSessionRepository(db)


def get_transcription_repository(
    db: AsyncSession = Depends(get_session),
) -> ITranscriptionRepository:
    return SQLAlchemyTranscriptionRepository(db)


def get_translation_repository(
    db: AsyncSession = Depends(get_session),
) -> ITranslationRepository:
    return SQLAlchemyTranslationRepository(db)


# AI services - singletons (loaded once at startup)
_asr_service: ASRService | None = None
_mt_service: MTService | None = None
_tts_service: TTSService | None = None


def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService(
            model_name=settings.ASR_MODEL, device=settings.ASR_DEVICE
        )
    return _asr_service


def get_mt_service() -> MTService:
    global _mt_service
    if _mt_service is None:
        _mt_service = MTService(model_name=settings.MT_MODEL, device=settings.MT_DEVICE)
    return _mt_service


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(
            model_name=settings.TTS_MODEL, device=settings.TTS_DEVICE
        )
    return _tts_service


# Application services
def get_session_service(
    session_repo: ITranslationSessionRepository = Depends(get_session_repository),
    transcription_repo: ITranscriptionRepository = Depends(
        get_transcription_repository
    ),
    translation_repo: ITranslationRepository = Depends(get_translation_repository),
) -> SessionService:
    return SessionService(session_repo, transcription_repo, translation_repo)


async def require_session_token(
    session_id: int,
    authorization: str | None = Header(default=None),
    service: SessionService = Depends(get_session_service),
) -> None:
    token = (
        authorization[7:]
        if authorization and authorization.lower().startswith("bearer ")
        else None
    )
    if not await service.authorize(session_id, token):
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_pipeline_service(
    session_repo: ITranslationSessionRepository = Depends(get_session_repository),
    transcription_repo: ITranscriptionRepository = Depends(
        get_transcription_repository
    ),
    translation_repo: ITranslationRepository = Depends(get_translation_repository),
    asr_service: ASRService = Depends(get_asr_service),
    mt_service: MTService = Depends(get_mt_service),
    tts_service: TTSService = Depends(get_tts_service),
) -> TranslationPipelineService:
    return TranslationPipelineService(
        session_repo=session_repo,
        transcription_repo=transcription_repo,
        translation_repo=translation_repo,
        asr_service=asr_service,
        mt_service=mt_service,
        tts_service=tts_service,
    )
