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
                    # Mirrors the finally block's suppress on complete_session():
                    # if fail_session()'s own write fails (its rollback trips, a
                    # DB blip), that must not raise a second exception out of the
                    # handler - that would skip the send_json below and, worse,
                    # leave status_decided False, making the finally block try
                    # (and very possibly also fail) to mark the session COMPLETED
                    # instead - the same stuck-ACTIVE failure shape already fixed
                    # twice (79d3547, d8cd711).
                    with contextlib.suppress(Exception):
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
            with contextlib.suppress(Exception):
                await pipeline_service.complete_session(session_id)
        # A failed close is never actionable: the socket is going away either way.
        with contextlib.suppress(Exception):
            await websocket.close()
