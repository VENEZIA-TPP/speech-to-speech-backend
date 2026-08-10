from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.base import Base


class Transcription(Base):
    __tablename__ = "transcriptions"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "chunk_index", name="uq_transcription_session_chunk"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer, ForeignKey("translation_sessions.id"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)
    original_text = Column(String, nullable=False)
    detected_language = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)
    asr_processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session = relationship("TranslationSession", back_populates="transcriptions")
    translation = relationship(
        "Translation",
        back_populates="transcription",
        uselist=False,
        cascade="all, delete-orphan",
    )
