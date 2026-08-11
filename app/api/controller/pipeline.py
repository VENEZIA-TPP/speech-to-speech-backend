import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from app.core.config import settings
from app.dependencies import get_pipeline_service
from app.pipeline.contracts import SessionState
from app.schemas.events import (
    AudioDelta,
    AudioDone,
    ErrorEvent,
    SegmentMetrics,
    SessionCompleted,
    SessionCreated,
    TranscriptionCompleted,
    TranslationCompleted,
)
from app.services.translation_pipeline_service import TranslationPipelineService

router = APIRouter()


async def _send(websocket: WebSocket, event: BaseModel) -> None:
    await websocket.send_json(event.model_dump())


@router.websocket("/ws/{session_id}")
async def pipeline_websocket(
    websocket: WebSocket,
    session_id: int,
    pipeline_service: TranslationPipelineService = Depends(get_pipeline_service),
):
    # Sec-WebSocket-Protocol, not a header or query string: it's settable from
    # a browser's native WebSocket API (new WebSocket(url, [token])), and like
    # a header it never lands in uvicorn's default access log (only the path +
    # query string do).
    subprotocols = websocket.scope.get("subprotocols", [])
    token = subprotocols[0] if subprotocols else None
    # Echoed into a response header below; gate to the token charset first so
    # unvalidated client bytes never reach the header encoder (worst case
    # without this is an ugly 500 from h11, not header injection, but it's
    # cheap insurance). str.isalnum() alone is Unicode-aware (accepts 'ñ',
    # '日本', '٣'), so isascii() is required too to actually restrict this to
    # ASCII letters/digits plus '-'/'_'.
    is_plausible_token = (
        token is not None
        and token.isascii()
        and token.replace("-", "").replace("_", "").isalnum()
    )
    echo_subprotocol = token if is_plausible_token else None
    if not await pipeline_service.authorize(session_id, token):
        # accept() before close() is required to deliver a custom close code:
        # an ASGI reject without accept surfaces to the client as HTTP 403,
        # not 4401. It must also echo back the subprotocol the client
        # offered: a real browser that offers a subprotocol and gets none
        # back in the handshake response treats that as a handshake
        # *failure* (onclose fires with 1006, our 4401 never arrives), not a
        # clean open-then-close - silently defeating this task's whole point.
        await websocket.accept(subprotocol=echo_subprotocol)
        await websocket.close(code=4401)
        return

    await websocket.accept(subprotocol=echo_subprotocol)
    await _send(websocket, SessionCreated(session_id=session_id))

    # Born with the connection, dies with this coroutine. There is no global
    # session_id -> state map to index wrong, and the engines are frozen, so
    # this is the only place per-session streaming state can live.
    state = SessionState()

    # A segment is what the client is told about; today the pipeline closes one
    # per chunk received, which is why this doubles as the chunk index the
    # pipeline persists. Voice-driven segmentation breaks that equality, and
    # the protocol already allows it to: nothing sent over the wire promises
    # one segment per chunk.
    segment_index = 0
    # True once this handler has decided the session's terminal status, so the
    # cleanup below cannot overwrite a FAILED session with COMPLETED.
    status_decided = False

    try:
        while True:
            data = await websocket.receive()

            # Raw receive() RETURNS the disconnect message instead of raising it;
            # without this branch the next receive() blows up with RuntimeError,
            # that escapes the handler, and the session stays ACTIVE forever.
            if data["type"] == "websocket.disconnect":
                break

            # Control message: a JSON object with a `type`.
            if "text" in data:
                try:
                    event_type = json.loads(data["text"])["type"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    await _send(
                        websocket,
                        ErrorEvent(
                            code="invalid_event",
                            message="expected a JSON object with a `type` field",
                        ),
                    )
                    continue

                if event_type == "input_audio.commit":
                    await pipeline_service.complete_session(session_id)
                    status_decided = True
                    await _send(
                        websocket,
                        SessionCompleted(
                            session_id=session_id, total_segments=segment_index
                        ),
                    )
                    break

                # An unrecognized control frame is answered, not swallowed, and
                # it does not tear down the audio stream over one bad frame.
                # The echo is truncated: it is client-controlled input.
                await _send(
                    websocket,
                    ErrorEvent(
                        code="invalid_event",
                        message=f"unknown event type: {str(event_type)[:64]}",
                    ),
                )
                continue

            # Audio chunk (binary)
            if "bytes" in data:
                if len(data["bytes"]) > settings.MAX_AUDIO_FRAME_BYTES:
                    # Status write happens-before the close signal, like every
                    # other terminal path below - a client that has observed
                    # the close already knows the session's status landed, so
                    # nothing is left in flight for a concurrent shutdown (the
                    # test harness tearing down its DB) to race against.
                    with contextlib.suppress(Exception):
                        await pipeline_service.complete_session(session_id)
                    status_decided = True
                    await websocket.close(code=1009)  # RFC 6455: message too big
                    break
                try:
                    result = await pipeline_service.process_audio_chunk(
                        state,
                        session_id=session_id,
                        audio_bytes=data["bytes"],
                        chunk_index=segment_index,
                    )
                except ValueError as exc:
                    # Same terminal status as the except Exception branch below,
                    # for the same reason: a session that failed mid-chunk must
                    # not be left for the finally block to mark COMPLETED. The
                    # suppress guards against fail_session()'s own write failing
                    # (a rollback trip, a DB blip) escaping as a second
                    # exception, which would skip the error event below.
                    with contextlib.suppress(Exception):
                        await pipeline_service.fail_session(session_id)
                    status_decided = True
                    await _send(
                        websocket,
                        ErrorEvent(
                            code="pipeline_failed",
                            segment_index=segment_index,
                            message=str(exc),
                        ),
                    )
                    break
                except Exception as exc:
                    # Mirrors the finally block's suppress on complete_session():
                    # if fail_session()'s own write fails (its rollback trips, a
                    # DB blip), that must not raise a second exception out of the
                    # handler - that would skip the error event below and, worse,
                    # leave status_decided False, making the finally block try
                    # (and very possibly also fail) to mark the session COMPLETED
                    # instead - the same stuck-ACTIVE failure shape already fixed
                    # twice (79d3547, d8cd711).
                    with contextlib.suppress(Exception):
                        await pipeline_service.fail_session(session_id)
                    status_decided = True
                    await _send(
                        websocket,
                        ErrorEvent(
                            code="pipeline_failed",
                            segment_index=segment_index,
                            message=f"Pipeline error: {exc}",
                        ),
                    )
                    break

                index = segment_index
                segment_index += 1

                # Send failures (client gone mid-burst) must not mark the
                # session FAILED - they fall through to the disconnect handling.
                await _send(
                    websocket,
                    TranscriptionCompleted(
                        segment_index=index,
                        transcript=result.original_text,
                        language_code=result.detected_language,
                    ),
                )
                await _send(
                    websocket,
                    TranslationCompleted(
                        segment_index=index,
                        text=result.translated_text,
                        target_language=result.target_language,
                    ),
                )
                if result.synthesized_audio:
                    await _send(
                        websocket,
                        AudioDelta(
                            segment_index=index,
                            seq=0,
                            size_bytes=len(result.synthesized_audio),
                        ),
                    )
                    await websocket.send_bytes(result.synthesized_audio)
                await _send(
                    websocket,
                    AudioDone(
                        segment_index=index,
                        watermarked=bool(result.watermarked),
                        watermark_method=result.watermark_method,
                    ),
                )
                await _send(
                    websocket,
                    SegmentMetrics(
                        segment_index=index,
                        asr_ms=result.asr_processing_time_ms,
                        mt_ms=result.mt_processing_time_ms,
                        tts_ms=result.tts_processing_time_ms,
                        e2e_ms=result.total_processing_time_ms,
                    ),
                )

    except WebSocketDisconnect:
        pass
    finally:
        # An abandoned session lands on COMPLETED rather than a new
        # ABANDONED status - that would cost an enum migration for a distinction
        # nothing reads yet. Add it when a report needs to tell them apart.
        if not status_decided:
            with contextlib.suppress(Exception):
                await pipeline_service.complete_session(session_id)
        # A failed close is never actionable: the socket is going away either way.
        with contextlib.suppress(Exception):
            await websocket.close()
