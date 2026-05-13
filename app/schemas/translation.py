from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TranslationRead(BaseModel):
    id: int
    transcription_id: int
    translated_text: str
    source_language: str
    target_language: str
    mt_processing_time_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PipelineResult(BaseModel):
    chunk_index: int
    original_text: str
    translated_text: str
    detected_language: Optional[str] = None
    source_language: str
    target_language: str
    asr_processing_time_ms: Optional[int] = None
    mt_processing_time_ms: Optional[int] = None
    total_processing_time_ms: Optional[int] = None
