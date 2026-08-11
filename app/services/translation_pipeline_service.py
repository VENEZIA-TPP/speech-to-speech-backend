"""
Translation Pipeline Service - orchestrates the full ASR -> MT -> TTS pipeline.

Flow per audio chunk:
  1. Look up the active TranslationSession.
  2. Run ASRService.transcribe() -> get text + detected language + latency.
  3. Persist Transcription to DB.
  4. Run MTService.translate() -> get translated text + latency.
  5. Persist Translation to DB.
  6. Run TTSService.synthesize() -> synthesized audio + watermark tag + latency
     (TTS output is not persisted this delivery).
  7. Return PipelineResult with the per-stage metrics + synthesized audio.
     The end-to-end clock deliberately does NOT live here: measured from
     inside this method it would miss the event serialization and the send,
     which is how it came to underreport. It belongs to the WebSocket
     handler, the only place that can see the whole window.

Per-session streaming state (buffer, previous prompt, decoder cache) travels in
the SessionState the WebSocket handler builds at accept() time - never on an
engine, which is frozen and shared process-wide across every session.
"""

from app.models.translation_session import SessionStatus
from app.pipeline.contracts import SessionState
from app.repositories.interfaces.translation_session_repository import (
    ITranslationSessionRepository,
)
from app.repositories.interfaces.transcription_repository import (
    ITranscriptionRepository,
)
from app.repositories.interfaces.translation_repository import ITranslationRepository
from app.schemas.translation import PipelineResult
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.session_auth import authorize_session_token
from app.services.tts_service import TTSService


class TranslationPipelineService:
    def __init__(
        self,
        session_repo: ITranslationSessionRepository,
        transcription_repo: ITranscriptionRepository,
        translation_repo: ITranslationRepository,
        asr_service: ASRService,
        mt_service: MTService,
        tts_service: TTSService,
    ):
        self.session_repo = session_repo
        self.transcription_repo = transcription_repo
        self.translation_repo = translation_repo
        self.asr_service = asr_service
        self.mt_service = mt_service
        self.tts_service = tts_service

    async def process_audio_chunk(
        self,
        state: SessionState,
        session_id: int,
        audio_bytes: bytes,
        chunk_index: int,
    ) -> PipelineResult:
        session = await self.session_repo.get_by_id(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # ASR
        asr_result = await self.asr_service.transcribe(
            state.asr,
            audio_bytes=audio_bytes,
            source_language=session.source_language,
        )

        transcription = await self.transcription_repo.create(
            session_id=session_id,
            chunk_index=chunk_index,
            original_text=asr_result.text,
            detected_language=asr_result.detected_language,
            confidence=asr_result.confidence,
            asr_processing_time_ms=asr_result.processing_time_ms,
        )

        # MT
        mt_result = await self.mt_service.translate(
            state.mt,
            text=asr_result.text,
            source_language=asr_result.detected_language or session.source_language,
            target_language=session.target_language,
        )

        await self.translation_repo.create(
            transcription_id=transcription.id,
            translated_text=mt_result.translated_text,
            source_language=mt_result.source_language,
            target_language=mt_result.target_language,
            mt_processing_time_ms=mt_result.processing_time_ms,
        )

        # TTS (output not persisted this delivery). The voice sample rides on
        # state.tts; the raw chunk is not a speaker reference and can no longer
        # be passed as one.
        tts_result = await self.tts_service.synthesize(
            state.tts,
            text=mt_result.translated_text,
            language=mt_result.target_language,
        )

        audio = tts_result.audio

        return PipelineResult(
            chunk_index=chunk_index,
            original_text=asr_result.text,
            translated_text=mt_result.translated_text,
            detected_language=asr_result.detected_language,
            source_language=mt_result.source_language,
            target_language=mt_result.target_language,
            asr_processing_time_ms=asr_result.processing_time_ms,
            mt_processing_time_ms=mt_result.processing_time_ms,
            tts_processing_time_ms=tts_result.processing_time_ms,
            synthesized_audio_size_bytes=len(audio.data),
            # True by construction: WatermarkedAudio cannot exist with an empty
            # method, and it is the only audio type that gets this far.
            watermarked=True,
            watermark_method=audio.method,
            synthesized_audio=audio.data,
        )

    async def authorize(self, session_id: int, token: str | None) -> bool:
        """Constant-time check that `token` is this session's ws_token.

        Lives in the service, not the controller: it is an authorization rule
        and it needs the repository.
        """
        return await authorize_session_token(self.session_repo, session_id, token)

    async def complete_session(self, session_id: int) -> None:
        await self.session_repo.update_status(session_id, SessionStatus.COMPLETED)

    async def fail_session(self, session_id: int) -> None:
        await self.session_repo.update_status(session_id, SessionStatus.FAILED)
