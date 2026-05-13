from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TranscriptionRead(BaseModel):
    id: int
    session_id: int
    chunk_index: int
    original_text: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    asr_processing_time_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}
