from datetime import datetime
from pydantic import BaseModel

from app.models.translation_session import SessionStatus


class TranslationSessionCreate(BaseModel):
    source_language: str = "en"
    target_language: str = "es"


class TranslationSessionUpdate(BaseModel):
    status: SessionStatus


class TranslationSessionRead(BaseModel):
    id: int
    source_language: str
    target_language: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
