from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


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
    tts_processing_time_ms: Optional[int] = None
    total_processing_time_ms: Optional[int] = None
    synthesized_audio_size_bytes: Optional[int] = None
    watermarked: Optional[bool] = None
    watermark_method: Optional[str] = None
    # Raw audio reaches the controller but never the JSON frame; the WS
    # protocol sends it as a separate binary frame right after the JSON.
    synthesized_audio: Optional[bytes] = Field(default=None, exclude=True)
