from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.transcription import Transcription
from app.repositories.interfaces.transcription_repository import (
    ITranscriptionRepository,
)


class SQLAlchemyTranscriptionRepository(ITranscriptionRepository):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        session_id: int,
        chunk_index: int,
        original_text: str,
        detected_language: Optional[str],
        confidence: Optional[float],
        asr_processing_time_ms: Optional[int],
    ) -> Transcription:
        transcription = Transcription(
            session_id=session_id,
            chunk_index=chunk_index,
            original_text=original_text,
            detected_language=detected_language,
            confidence=confidence,
            asr_processing_time_ms=asr_processing_time_ms,
        )
        self.db.add(transcription)
        await self.db.commit()
        await self.db.refresh(transcription)
        return transcription

    async def get_by_session(self, session_id: int) -> List[Transcription]:
        result = await self.db.execute(
            select(Transcription)
            .where(Transcription.session_id == session_id)
            .order_by(Transcription.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_id(self, transcription_id: int) -> Transcription | None:
        result = await self.db.execute(
            select(Transcription).where(Transcription.id == transcription_id)
        )
        return result.scalar_one_or_none()
