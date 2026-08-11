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
from app.pipeline.vad import VoiceSegmenter, build_segmenter
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


# AI services - built once by the lifespan (app/main.py), read from here.
# Deliberately without a lock: a getter that never constructs has no
# check-then-set to protect.
_asr_service: ASRService | None = None
_mt_service: MTService | None = None
_tts_service: TTSService | None = None
_segmenter: VoiceSegmenter | None = None

_NOT_BUILT = "engines not built: the lifespan did not run"


def init_engines() -> None:
    """Build the three engines once, at startup, before any traffic.

    Called from the lifespan. If a backend cannot load its weights, the
    exception propagates out of the lifespan and the process refuses to
    start - instead of failing on some user's first request.

    Runs synchronously inside an async lifespan. Harmless with
    stubs (two string assignments per engine), but real weights are
    30-90s of blocking CUDA/ONNX loading that stalls SIGTERM/startup
    probes. Upgrade path if that bites: wrap the call at the lifespan
    call site with `await anyio.to_thread.run_sync(init_engines)` -
    fail-fast still holds, the exception still propagates out.
    """
    global _asr_service, _mt_service, _tts_service, _segmenter
    _asr_service = ASRService(model_name=settings.ASR_MODEL, device=settings.ASR_DEVICE)
    _mt_service = MTService(model_name=settings.MT_MODEL, device=settings.MT_DEVICE)
    _tts_service = TTSService(model_name=settings.TTS_MODEL, device=settings.TTS_DEVICE)
    _segmenter = build_segmenter()


def get_asr_service() -> ASRService:
    if _asr_service is None:
        raise RuntimeError(_NOT_BUILT)
    return _asr_service


def get_mt_service() -> MTService:
    if _mt_service is None:
        raise RuntimeError(_NOT_BUILT)
    return _mt_service


def get_tts_service() -> TTSService:
    if _tts_service is None:
        raise RuntimeError(_NOT_BUILT)
    return _tts_service


def get_segmenter() -> VoiceSegmenter:
    if _segmenter is None:
        raise RuntimeError(_NOT_BUILT)
    return _segmenter


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
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_pipeline_service(
    session_repo: ITranslationSessionRepository = Depends(get_session_repository),
    transcription_repo: ITranscriptionRepository = Depends(
        get_transcription_repository
    ),
    translation_repo: ITranslationRepository = Depends(get_translation_repository),
    asr_service: ASRService = Depends(get_asr_service),
    mt_service: MTService = Depends(get_mt_service),
    tts_service: TTSService = Depends(get_tts_service),
    segmenter: VoiceSegmenter = Depends(get_segmenter),
) -> TranslationPipelineService:
    return TranslationPipelineService(
        session_repo=session_repo,
        transcription_repo=transcription_repo,
        translation_repo=translation_repo,
        asr_service=asr_service,
        mt_service=mt_service,
        tts_service=tts_service,
        segmenter=segmenter,
    )
