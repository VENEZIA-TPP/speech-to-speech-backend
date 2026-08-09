from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.translation_session import TranslationSession, SessionStatus
from app.schemas.translation_session import TranslationSessionCreate
from app.repositories.interfaces.translation_session_repository import (
    ITranslationSessionRepository,
)


class SQLAlchemyTranslationSessionRepository(ITranslationSessionRepository):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, session_in: TranslationSessionCreate) -> TranslationSession:
        session = TranslationSession(
            source_language=session_in.source_language,
            target_language=session_in.target_language,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_by_id(self, session_id: int) -> TranslationSession | None:
        result = await self.db.execute(
            select(TranslationSession).where(TranslationSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self, session_id: int, status: SessionStatus
    ) -> TranslationSession | None:
        # The shared AsyncSession may already be poisoned by a failed write
        # elsewhere in this request (e.g. a UniqueConstraint violation) - every
        # call on it raises PendingRollbackError until it's rolled back. This
        # is a no-op when the session isn't in that state, so always run it.
        # It also unconditionally discards any uncommitted work on the shared
        # session - harmless today since every write in this codebase commits
        # immediately, but worth knowing if a unit-of-work / deferred-commit
        # pattern is ever introduced later.
        await self.db.rollback()
        session = await self.get_by_id(session_id)
        if session is None:
            return None
        session.status = status
        session.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def delete(self, session_id: int) -> bool:
        session = await self.get_by_id(session_id)
        if session is None:
            return False
        await self.db.delete(session)
        await self.db.commit()
        return True
