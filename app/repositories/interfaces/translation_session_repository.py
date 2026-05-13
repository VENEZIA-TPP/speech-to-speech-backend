from abc import ABC, abstractmethod

from app.models.translation_session import TranslationSession, SessionStatus
from app.schemas.translation_session import TranslationSessionCreate


class ITranslationSessionRepository(ABC):
    @abstractmethod
    async def create(self, session_in: TranslationSessionCreate) -> TranslationSession: ...

    @abstractmethod
    async def get_by_id(self, session_id: int) -> TranslationSession | None: ...

    @abstractmethod
    async def update_status(self, session_id: int, status: SessionStatus) -> TranslationSession | None: ...

    @abstractmethod
    async def delete(self, session_id: int) -> bool: ...
