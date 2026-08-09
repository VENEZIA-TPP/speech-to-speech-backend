from typing import List, Optional

from app.models.translation_session import TranslationSession, SessionStatus
from app.models.transcription import Transcription
from app.models.translation import Translation
from app.repositories.interfaces.translation_session_repository import (
    ITranslationSessionRepository,
)
from app.repositories.interfaces.transcription_repository import (
    ITranscriptionRepository,
)
from app.repositories.interfaces.translation_repository import ITranslationRepository
from app.schemas.translation_session import TranslationSessionCreate
from app.services.session_auth import authorize_session_token


class SessionService:
    def __init__(
        self,
        session_repo: ITranslationSessionRepository,
        transcription_repo: ITranscriptionRepository,
        translation_repo: ITranslationRepository,
    ):
        self.session_repo = session_repo
        self.transcription_repo = transcription_repo
        self.translation_repo = translation_repo

    async def create_session(
        self, session_in: TranslationSessionCreate
    ) -> TranslationSession:
        return await self.session_repo.create(session_in)

    async def get_session(self, session_id: int) -> Optional[TranslationSession]:
        return await self.session_repo.get_by_id(session_id)

    async def complete_session(self, session_id: int) -> Optional[TranslationSession]:
        return await self.session_repo.update_status(
            session_id, SessionStatus.COMPLETED
        )

    async def fail_session(self, session_id: int) -> Optional[TranslationSession]:
        return await self.session_repo.update_status(session_id, SessionStatus.FAILED)

    async def delete_session(self, session_id: int) -> bool:
        return await self.session_repo.delete(session_id)

    async def get_transcriptions(self, session_id: int) -> List[Transcription]:
        return await self.transcription_repo.get_by_session(session_id)

    async def get_translations(self, session_id: int) -> List[Translation]:
        return await self.translation_repo.get_by_session(session_id)

    async def authorize(self, session_id: int, token: str | None) -> bool:
        return await authorize_session_token(self.session_repo, session_id, token)
