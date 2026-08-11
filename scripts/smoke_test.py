"""End-to-end smoke test against a running server (real Postgres, stub models).

Creates a session over REST, streams WAV chunks over the WebSocket, and reads
back the typed event stream: a transcription, a translation, a watermarked audio
frame and its metrics per segment. Sends every chunk before reading any reply,
because the protocol promises neither one segment per chunk nor a reply before
the next send. Complements tests/ (which runs in-process against SQLite) by
exercising the real server, the real DB and a real WS client.

It sends recorded speech rather than generated silence. The pipeline runs on
what the voice segmenter decides is speech, and the segmenter does not fire on
synthetic audio, so a run made of silence would now complete with zero segments
and assert nothing at all - passing while exercising none of the pipeline.

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
from pathlib import Path

import websockets

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
CHUNKS = int(os.environ.get("CHUNKS", "2"))

SAMPLE_RATE = 16000
# Enough silence after the speech to take the segmenter past its endpoint
# threshold, so each chunk closes its own segment instead of leaving one open
# for the end-of-session flush.
TRAILING_SILENCE_S = 0.6
SPEECH_WAV = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / ("es_sistema.wav")
)


def make_wav() -> bytes:
    """One chunk: recorded speech plus enough trailing silence to close it."""
    with wave.open(str(SPEECH_WAV), "rb") as source:
        frames = source.readframes(source.getnframes())

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)
        wav.writeframes(b"\x00\x00" * int(SAMPLE_RATE * TRAILING_SILENCE_S))
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
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created", created

        # Every chunk goes out before a single reply is read: the protocol
        # does not promise one segment per chunk, nor a reply before the next
        # send is allowed. The commit rides right behind them.
        for _ in range(CHUNKS):
            await ws.send(audio)
        await ws.send(json.dumps({"type": "input_audio.commit"}))

        segments: set[int] = set()
        pending_audio: tuple[int, int] | None = None

        while True:
            frame = await ws.recv()

            if isinstance(frame, bytes):
                assert pending_audio is not None, "binary frame with no audio.delta"
                index, size_bytes = pending_audio
                assert len(frame) == size_bytes, (len(frame), size_bytes)
                with wave.open(io.BytesIO(frame)) as out:
                    print(
                        f"         segment {index}: audio {len(frame)} bytes, "
                        f"{out.getframerate()} Hz, {out.getnchannels()} ch, "
                        f"{out.getnframes() / out.getframerate():.2f}s"
                    )
                pending_audio = None
                continue

            event = json.loads(frame)
            assert event["type"] != "error", event
            index = event.get("segment_index")

            if event["type"] == "transcription.completed":
                print(f"segment {index}: {event['transcript']!r}")
            elif event["type"] == "translation.completed":
                print(f"         -> {event['text']!r}")
            elif event["type"] == "audio.delta":
                pending_audio = (index, event["size_bytes"])
            elif event["type"] == "audio.done":
                # Rule #6: every synthesized output must carry a watermark tag.
                assert event["watermarked"] is True, event
                assert event["watermark_method"], event
                segments.add(index)
            elif event["type"] == "segment.metrics":
                print(f"         {event['e2e_ms']} ms total")
            elif event["type"] == "session.completed":
                assert event["total_segments"] == len(segments), (event, segments)
                # A run that produced nothing is a failed run, not a quiet one:
                # every assertion above lives inside a branch that only fires
                # when a segment comes back.
                assert segments, "no segment came back: the VAD found no speech"
                print(
                    f"session {session_id} completed, "
                    f"{event['total_segments']} segments"
                )
                break

        assert pending_audio is None, pending_audio


def main() -> None:
    session_id, token = create_session()
    asyncio.run(stream(session_id, token, make_wav()))
    print("OK")


if __name__ == "__main__":
    main()
