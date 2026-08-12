from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.translation import Translation


class ITranslationRepository(ABC):
    @abstractmethod
    async def create(
        self,
        transcription_id: int,
        translated_text: str,
        source_language: str,
        target_language: str,
        mt_processing_time_ms: Optional[int],
    ) -> Translation: ...

    @abstractmethod
    async def get_by_session(self, session_id: int) -> List[Translation]: ...

    @abstractmethod
    async def get_by_transcription(
        self, transcription_id: int
    ) -> Translation | None: ...
