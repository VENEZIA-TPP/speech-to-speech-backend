"""Cost and behaviour of voice-driven segmentation, on CPU.

Why this one measurement counts as evidence and the rest of the local numbers
do not: Silero runs on a CPU thread in production too. There is no GPU in this
path and there never will be, so a figure measured on a development machine is
measured on the same kind of hardware that will serve it. Every other timing
this project can produce locally - ASR, MT, TTS - is a correctness check
standing in for hardware it will not run on.

Two things are reported:

  1. `frame_cost` - what the model costs per 32 ms of audio, which is what
     decides whether segmentation can stay on the event loop.
  2. `endpoint` - how a pause of a given length is classified, which is the
     latency/quality dial: too low and an intra-phrase breath ends the segment,
     handing the translator a fragment; too high and every phrase pays the
     difference in latency.

Run:  .venv/bin/python benchmarks/vad_endpoint.py
Writes benchmarks/results/vad_endpoint.json.

The audio is the recorded fixture the tests use (tests/fixtures/README.md).
It is text-to-speech, so these numbers describe segmentation timing and
segmentation behaviour. They are not evidence about transcription or
translation quality, which needs a human corpus.
"""

import io
import json
import platform
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Run as a script from anywhere, without a PYTHONPATH incantation. pytest.ini
# does this for the test suite; a benchmark is not collected by pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.pipeline.contracts import VADState  # noqa: E402
from app.pipeline.vad import (  # noqa: E402
    FRAME_SAMPLES,
    build_segmenter,
    pcm_from_wav,
)

ROOT = Path(__file__).resolve().parent.parent
SPEECH_WAV = ROOT / "tests" / "fixtures" / "es_sistema.wav"
RESULTS = Path(__file__).parent / "results" / "vad_endpoint.json"

SR = settings.AUDIO_SAMPLE_RATE
REPEATS = 5


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def as_wav(samples: np.ndarray) -> bytes:
    pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def measure_frame_cost(segmenter, speech) -> dict:
    """Per-frame model cost. Warmup is excluded and reported separately."""
    warmup_start = time.perf_counter()
    segmenter.feed(VADState(), as_wav(speech))
    warmup_ms = (time.perf_counter() - warmup_start) * 1000

    per_frame_ms = []
    state = VADState()
    frame = speech[FRAME_SAMPLES : 2 * FRAME_SAMPLES]
    for _ in range(REPEATS * 200):
        start = time.perf_counter()
        segmenter._speech_prob(state, frame)
        per_frame_ms.append((time.perf_counter() - start) * 1000)

    chunk_frames = settings.AUDIO_CHUNK_DURATION_MS * SR / 1000 / FRAME_SAMPLES
    return {
        "frame_ms": FRAME_SAMPLES * 1000 / SR,
        "warmup_ms": round(warmup_ms, 3),
        "samples": len(per_frame_ms),
        "mean_ms": round(statistics.mean(per_frame_ms), 4),
        "p50_ms": round(statistics.median(per_frame_ms), 4),
        "p95_ms": round(
            statistics.quantiles(per_frame_ms, n=20)[18],
            4,
        ),
        "max_ms": round(max(per_frame_ms), 4),
        # What one arriving chunk costs the event loop, which is the number
        # that decides whether this work needs a thread at all.
        "per_chunk_ms": round(statistics.mean(per_frame_ms) * chunk_frames, 3),
        "chunk_duration_ms": settings.AUDIO_CHUNK_DURATION_MS,
        "real_time_factor": round(
            statistics.mean(per_frame_ms) / (FRAME_SAMPLES * 1000 / SR), 6
        ),
    }


def measure_endpoint(segmenter, speech) -> list[dict]:
    """How a pause of each length gets classified, and what it costs.

    Two phrases separated by an exact gap. One segment means the gap was read
    as a pause inside a phrase; two means it was read as the end of one.
    """
    rows = []
    for gap_ms in (0, 100, 200, 250, 300, 400, 500, 800, 1200):
        audio = as_wav(np.concatenate([speech, silence(gap_ms), speech, silence(900)]))
        elapsed = []
        for _ in range(REPEATS):
            state = VADState()
            start = time.perf_counter()
            segments = segmenter.feed(state, audio)
            elapsed.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "gap_ms": gap_ms,
                "segments": len(segments),
                "split": len(segments) > 1,
                "segment_ms": [s.duration_ms for s in segments],
                "vad_pass_ms": round(statistics.median(elapsed), 3),
            }
        )
    return rows


def measure_ceiling(segmenter, speech) -> dict:
    """Continuous speech past the ceiling, the bounded worst case."""
    continuous = np.concatenate([speech] * 6)
    segments = segmenter.feed(
        VADState(), as_wav(np.concatenate([continuous, silence(900)]))
    )
    return {
        "spoken_ms": int(len(continuous) * 1000 / SR),
        "segments": len(segments),
        "segment_ms": [s.duration_ms for s in segments],
        "reasons": [s.reason for s in segments],
        "max_speech_ms": settings.VAD_MAX_SPEECH_MS,
    }


def main() -> None:
    segmenter = build_segmenter()
    speech = pcm_from_wav(SPEECH_WAV.read_bytes())

    payload = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": "Silero VAD segmentation on CPU",
        "evidence": (
            "CPU-bound in production too, so these timings transfer. "
            "Speech is text-to-speech, so this says nothing about WER or BLEU."
        ),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "settings": {
            "threshold": settings.VAD_THRESHOLD,
            "min_silence_ms": settings.VAD_MIN_SILENCE_MS,
            "min_speech_ms": settings.VAD_MIN_SPEECH_MS,
            "speech_pad_ms": settings.VAD_SPEECH_PAD_MS,
            "max_speech_ms": settings.VAD_MAX_SPEECH_MS,
            "sample_rate": SR,
        },
        "audio": {
            "file": str(SPEECH_WAV.relative_to(ROOT)),
            "duration_ms": int(len(speech) * 1000 / SR),
        },
        "frame_cost": measure_frame_cost(segmenter, speech),
        "endpoint": measure_endpoint(segmenter, speech),
        "ceiling": measure_ceiling(segmenter, speech),
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")

    cost = payload["frame_cost"]
    print(f"frame cost   {cost['mean_ms']:.4f} ms mean / {cost['p95_ms']:.4f} ms p95")
    print(
        f"per chunk    {cost['per_chunk_ms']:.2f} ms "
        f"for {cost['chunk_duration_ms']} ms of audio "
        f"(RTF {cost['real_time_factor']:.5f})"
    )
    print(f"warmup       {cost['warmup_ms']:.1f} ms, once")
    print()
    print("gap      segments  durations(ms)")
    for row in payload["endpoint"]:
        print(f"{row['gap_ms']:>5} ms  {row['segments']:>5}     {row['segment_ms']}")
    ceiling = payload["ceiling"]
    print()
    print(
        f"ceiling      {ceiling['spoken_ms']} ms continuous -> "
        f"{ceiling['segments']} segments {ceiling['segment_ms']} "
        f"({', '.join(ceiling['reasons'])})"
    )
    print(f"\nwritten to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
