"""
Translation Pipeline Service - orchestrates the full ASR -> MT -> TTS pipeline.

The unit of work is a segment, not a chunk. An arriving chunk is fed to the
voice segmenter, which decides where speech ends and hands back however many
segments closed - zero, one or several. Each of those runs the pipeline:

  1. Run ASRService.transcribe() -> text + detected language + latency.
  2. Persist Transcription.
  3. Run MTService.translate() -> translated text + latency.
  4. Persist Translation.
  5. Run TTSService.synthesize() -> synthesized audio + watermark tag + latency
     (TTS output is not persisted this delivery).
  6. Return a PipelineResult per segment, carrying the segment's own capture
     time. The end-to-end clock deliberately does NOT run here: measured from
     inside this service it would miss the event serialization and the send,
     which is how it came to underreport. This returns the anchor; the
     WebSocket handler, the only place that can see the whole window, does the
     subtraction.

A chunk that closes no segment returns an empty list before touching the
database or the ASR at all. That is the point of the whole change: silence used
to run the full pipeline and persist an empty transcription every time.

Per-session streaming state (segmenter memory, buffers, previous prompt,
decoder cache) travels in the SessionState the WebSocket handler builds at
accept() time - never on an engine, which is frozen and shared process-wide
across every session.
"""

from app.models.translation_session import SessionStatus
from app.pipeline.contracts import SessionState
from app.pipeline.vad import SpeechSegment, VoiceSegmenter
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
        segmenter: VoiceSegmenter,
    ):
        self.session_repo = session_repo
        self.transcription_repo = transcription_repo
        self.translation_repo = translation_repo
        self.asr_service = asr_service
        self.mt_service = mt_service
        self.tts_service = tts_service
        self.segmenter = segmenter

    async def process_audio_chunk(
        self,
        state: SessionState,
        session_id: int,
        audio_bytes: bytes,
    ) -> list[PipelineResult]:
        """One arriving chunk in, one result per segment it closed out.

        Returns an empty list for a chunk that closed nothing - silence, or
        speech still in progress. Segmentation runs first precisely so that
        case costs one VAD pass and nothing else: no session lookup, no
        inference, no row.
        """
        segments = self.segmenter.feed(state.vad, audio_bytes)
        if not segments:
            return []
        return [await self._run(state, session_id, s) for s in segments]

    async def flush(
        self, state: SessionState, session_id: int
    ) -> PipelineResult | None:
        """Close and process whatever speech was still open at session end.

        Without this the last utterance of every session is lost. It is not a
        rare path: it happens whenever a speaker stops talking and closes the
        stream before the silence threshold elapses, which is the normal way a
        session ends.
        """
        segment = self.segmenter.flush(state.vad)
        if segment is None:
            return None
        return await self._run(state, session_id, segment)

    async def _run(
        self, state: SessionState, session_id: int, segment: SpeechSegment
    ) -> PipelineResult:
        session = await self.session_repo.get_by_id(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # The index is taken from the session's own state rather than from a
        # counter the caller keeps: a chunk can close several segments, so a
        # caller-side counter and the one persisted here would be two counters
        # that only agree by luck.
        segment_index = state.segments_emitted
        state.segments_emitted += 1

        # ASR
        asr_result = await self.asr_service.transcribe(
            state.asr,
            audio_bytes=segment.pcm,
            source_language=session.source_language,
        )

        transcription = await self.transcription_repo.create(
            session_id=session_id,
            chunk_index=segment_index,
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
            chunk_index=segment_index,
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
            segment_started_at=segment.t_capture,
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
