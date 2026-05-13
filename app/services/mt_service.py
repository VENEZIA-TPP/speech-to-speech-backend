import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class MTResult:
    translated_text: str
    source_language: str
    target_language: str
    processing_time_ms: int


class MTService:
    def __init__(self, model_name: str = "stub", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # TODO: load real model

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> MTResult:
        #TODO: Translate text from source_language to target_language.
        start = time.monotonic()
        translated = await self._translate(text, source_language, target_language)
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return MTResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            processing_time_ms=processing_time_ms,
        )

    async def _translate(self, text: str, source_language: str, target_language: str) -> str:
        #TODO: Internal translation  replace with real inference.

        return f"[MT stub] {source_language}->{target_language}: {text}"
