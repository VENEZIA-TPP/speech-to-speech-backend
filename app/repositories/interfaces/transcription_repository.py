from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.transcription import Transcription


class ITranscriptionRepository(ABC):
    @abstractmethod
    async def create(
        self,
        session_id: int,
        chunk_index: int,
        original_text: str,
        detected_language: Optional[str],
        confidence: Optional[float],
        asr_processing_time_ms: Optional[int],
    ) -> Transcription: ...

    @abstractmethod
    async def get_by_session(self, session_id: int) -> List[Transcription]: ...

    @abstractmethod
    async def get_by_id(self, transcription_id: int) -> Transcription | None: ...
