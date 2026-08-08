from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.dependencies import get_pipeline_service
from app.services.translation_pipeline_service import TranslationPipelineService

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def pipeline_websocket(
    websocket: WebSocket,
    session_id: int,
    pipeline_service: TranslationPipelineService = Depends(get_pipeline_service),
):
    await websocket.accept()
    chunk_index = 0

    try:
        while True:
            data = await websocket.receive()

            # Control message
            if "text" in data:
                if data["text"] == "END":
                    await pipeline_service.complete_session(session_id)
                    await websocket.send_json({
                        "status": "completed",
                        "session_id": session_id,
                        "total_chunks": chunk_index,
                    })
                    break
                continue  # ignore other text frames

            # Audio chunk (binary)
            if "bytes" in data:
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
                    await websocket.send_json({"error": f"Pipeline error: {exc}"})
                    break
                # Send failures (client gone mid-frame-pair) must not mark the
                # session FAILED - they fall through to the disconnect handling.
                chunk_index += 1
                await websocket.send_json(result.model_dump())
                if result.synthesized_audio:
                    await websocket.send_bytes(result.synthesized_audio)

        # Loop exits via break (END or error): close explicitly so clients
        # see a clean close frame instead of relying on server teardown.
        await websocket.close()

    except WebSocketDisconnect:
        await pipeline_service.complete_session(session_id)
