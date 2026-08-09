"""End-to-end smoke test against a running server (real Postgres, stub models).

Creates a session over REST, streams synthetic WAV chunks over the WebSocket,
and asserts the pipeline replies with a JSON result + a playable WAV frame per
chunk. Complements tests/ (which runs in-process against SQLite) by exercising
the real server, the real DB and a real WS client.

Usage (server must already be running):
    .venv/bin/python scripts/smoke_test.py

Env overrides: BASE_URL (default http://127.0.0.1:8000), CHUNKS (default 2).
"""

import asyncio
import io
import json
import os
import urllib.request
import wave

import websockets

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
CHUNKS = int(os.environ.get("CHUNKS", "2"))

SAMPLE_RATE = 16000
CHUNK_DURATION_S = 2


def make_wav() -> bytes:
    """Synthetic 16 kHz mono pcm_s16le silence - the format the pipeline expects."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * (SAMPLE_RATE * CHUNK_DURATION_S))
    return buffer.getvalue()


def create_session() -> tuple[int, str]:
    payload = json.dumps({"source_language": "es", "target_language": "en"}).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/sessions/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        session = json.load(response)
    print(f"session {session['id']} created ({session['status']})")
    return session["id"], session["ws_token"]


async def stream(session_id: int, token: str, audio: bytes) -> None:
    ws_url = f"{BASE_URL.replace('http', 'ws', 1)}/pipeline/ws/{session_id}"

    async with websockets.connect(ws_url, subprotocols=[token]) as ws:
        for i in range(CHUNKS):
            await ws.send(audio)

            result = json.loads(await ws.recv())
            assert "error" not in result, result
            assert result["chunk_index"] == i, result
            assert result["original_text"], result
            assert result["translated_text"], result
            # Rule #6: every synthesized output must carry a watermark tag.
            assert result["watermarked"] is True, result
            assert result["watermark_method"], result

            print(
                f"chunk {i}: {result['original_text']!r} -> "
                f"{result['translated_text']!r} "
                f"({result['total_processing_time_ms']} ms total)"
            )

            # Raw audio never rides inside the JSON - it comes as a separate frame.
            assert result["synthesized_audio_size_bytes"] > 0, result
            frame = await ws.recv()
            assert isinstance(frame, bytes), type(frame)
            assert len(frame) == result["synthesized_audio_size_bytes"]
            with wave.open(io.BytesIO(frame)) as out:
                print(
                    f"         audio {len(frame)} bytes, {out.getframerate()} Hz, "
                    f"{out.getnchannels()} ch, "
                    f"{out.getnframes() / out.getframerate():.2f}s"
                )

        await ws.send("END")
        done = json.loads(await ws.recv())
        assert done["status"] == "completed", done
        assert done["total_chunks"] == CHUNKS, done
        print(f"session {session_id} completed, {done['total_chunks']} chunks")


def main() -> None:
    session_id, token = create_session()
    asyncio.run(stream(session_id, token, make_wav()))
    print("OK")


if __name__ == "__main__":
    main()
