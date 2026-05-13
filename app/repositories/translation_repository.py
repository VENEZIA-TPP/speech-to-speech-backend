from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.translation import Translation
from app.models.transcription import Transcription
from app.repositories.interfaces.translation_repository import ITranslationRepository


class SQLAlchemyTranslationRepository(ITranslationRepository):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        transcription_id: int,
        translated_text: str,
        source_language: str,
        target_language: str,
        mt_processing_time_ms: Optional[int],
    ) -> Translation:
        translation = Translation(
            transcription_id=transcription_id,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            mt_processing_time_ms=mt_processing_time_ms,
        )
        self.db.add(translation)
        await self.db.commit()
        await self.db.refresh(translation)
        return translation

    async def get_by_session(self, session_id: int) -> List[Translation]:
        result = await self.db.execute(
            select(Translation)
            .join(Transcription)
            .where(Transcription.session_id == session_id)
            .order_by(Transcription.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_transcription(self, transcription_id: int) -> Translation | None:
        result = await self.db.execute(
            select(Translation).where(Translation.transcription_id == transcription_id)
        )
        return result.scalar_one_or_none()
