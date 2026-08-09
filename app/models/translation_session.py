import enum
import secrets
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Enum as SAEnum, DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationSession(Base):
    __tablename__ = "translation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    source_language = Column(String(10), nullable=False)  # ISO 639-1, e.g. "en"
    target_language = Column(String(10), nullable=False)  # ISO 639-1, e.g. "es"
    # Opaque bearer token for /pipeline/ws/{id}. Column default rather than
    # service-level generation so it cannot be forgotten by a new call site.
    ws_token = Column(
        String(64),
        default=lambda: secrets.token_urlsafe(32),
        nullable=False,
        index=True,
    )
    status = Column(SAEnum(SessionStatus), default=SessionStatus.ACTIVE, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    transcriptions = relationship(
        "Transcription",
        back_populates="session",
        cascade="all, delete-orphan",
    )
