import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.core.config import settings
from app.dependencies import get_pipeline_service
from app.services.translation_pipeline_service import TranslationPipelineService

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def pipeline_websocket(
    websocket: WebSocket,
    session_id: int,
    pipeline_service: TranslationPipelineService = Depends(get_pipeline_service),
):
    # accept() before close() is required to deliver a custom close code: an
    # ASGI reject without accept surfaces to the client as HTTP 403, not 4401.
    await websocket.accept()
    # Query-string tokens land in uvicorn's access log and any proxy log on
    # every successful connection, not just rejections. A header does not.
    auth_header = websocket.headers.get("authorization", "")
    token = auth_header[7:] if auth_header.lower().startswith("bearer ") else None
    if not await pipeline_service.authorize(session_id, token):
        await websocket.close(code=4401)
        return

    chunk_index = 0
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

            # Control message
            if "text" in data:
                if data["text"] == "END":
                    await pipeline_service.complete_session(session_id)
                    status_decided = True
                    await websocket.send_json(
                        {
                            "status": "completed",
                            "session_id": session_id,
                            "total_chunks": chunk_index,
                        }
                    )
                    break
                continue  # ignore other text frames

            # Audio chunk (binary)
            if "bytes" in data:
                if len(data["bytes"]) > settings.MAX_AUDIO_FRAME_BYTES:
                    await websocket.close(code=1009)  # RFC 6455: message too big
                    break
                try:
                    result = await pipeline_service.process_audio_chunk(
                        session_id=session_id,
                        audio_bytes=data["bytes"],
                        chunk_index=chunk_index,
                    )
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
                    break
                except Exception as exc:
                    await pipeline_service.fail_session(session_id)
                    status_decided = True
                    await websocket.send_json({"error": f"Pipeline error: {exc}"})
                    break
                # Send failures (client gone mid-frame-pair) must not mark the
                # session FAILED - they fall through to the disconnect handling.
                chunk_index += 1
                await websocket.send_json(result.model_dump())
                if result.synthesized_audio:
                    await websocket.send_bytes(result.synthesized_audio)

    except WebSocketDisconnect:
        pass
    finally:
        # An abandoned session lands on COMPLETED rather than a new
        # ABANDONED status - that would cost an enum migration for a distinction
        # nothing reads yet. Add it when a report needs to tell them apart.
        if not status_decided:
            await pipeline_service.complete_session(session_id)
        # A failed close is never actionable: the socket is going away either way.
        with contextlib.suppress(Exception):
            await websocket.close()
