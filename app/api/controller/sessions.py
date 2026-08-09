from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_session_service, require_session_token
from app.schemas.transcription import TranscriptionRead
from app.schemas.translation import TranslationRead
from app.schemas.translation_session import (
    TranslationSessionCreate,
    TranslationSessionCreated,
    TranslationSessionRead,
)
from app.services.session_service import SessionService

router = APIRouter()


@router.post("/", response_model=TranslationSessionCreated, status_code=201)
async def create_session(
    session_in: TranslationSessionCreate,
    service: SessionService = Depends(get_session_service),
):
    return await service.create_session(session_in)


@router.get("/{session_id}", response_model=TranslationSessionRead)
async def get_session(
    session_id: int,
    service: SessionService = Depends(get_session_service),
    _: None = Depends(require_session_token),
):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}/complete", response_model=TranslationSessionRead)
async def complete_session(
    session_id: int,
    service: SessionService = Depends(get_session_service),
    _: None = Depends(require_session_token),
):
    session = await service.complete_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    service: SessionService = Depends(get_session_service),
    _: None = Depends(require_session_token),
):
    deleted = await service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/transcriptions", response_model=List[TranscriptionRead])
async def get_session_transcriptions(
    session_id: int,
    service: SessionService = Depends(get_session_service),
    _: None = Depends(require_session_token),
):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await service.get_transcriptions(session_id)


@router.get("/{session_id}/translations", response_model=List[TranslationRead])
async def get_session_translations(
    session_id: int,
    service: SessionService = Depends(get_session_service),
    _: None = Depends(require_session_token),
):
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await service.get_translations(session_id)
