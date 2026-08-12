"""OPUS-MT via CTranslate2 on CPU: a correctness check wearing a stopwatch.

**These timings are not evidence for this project.** MT is meant to run on a
cloud GPU; a number produced on a development laptop describes hardware that
will never serve a request. The result file says so in its own `evidence`
field, so it cannot be quoted out of context. Contrast with vad_endpoint.py,
which measures work that really is CPU-bound in production.

Three things are worth measuring anyway:

  1. `load` - what building one MTService costs. This is the figure the ASR
     replacement will need: the WebSocket test fixture is function-scoped and
     runs the whole lifespan per test, so whatever a model costs to open gets
     paid once per test, not once per suite. With stubs that is free and the
     issue is invisible.
  2. `translate` - per-segment wall clock, at the segment sizes voice-driven
     segmentation actually produces. A correctness check: it says the pipeline
     is not accidentally quadratic, and nothing about production latency.
  3. `missing_eos` - what happens when the source token list is not terminated
     with </s>. This is the one number here with lasting value, because the
     failure is silent: the decoder never emits EOS, runs to its decoding
     limit, and returns text that opens correctly and then degenerates into
     repeated fragments. Nothing raises. The cost ratio is what makes it
     obvious in a profile.

Run:  MT_MODEL=opus-mt .venv/bin/python benchmarks/mt_opus.py
Writes benchmarks/results/mt_opus_local.json. Needs the weights (~311 MB for
both directions), fetched to the huggingface_hub cache on first run.
"""

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.pipeline.contracts import MTState  # noqa: E402
from app.services.mt_service import MTService, _load_translator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).parent / "results" / "mt_opus_local.json"

REPEATS = 10

# Phrase-length inputs, because that is what the segmenter now hands over: one
# utterance ending where the speaker stopped, not a fixed three-second slice.
CORPUS = {
    ("es", "en"): [
        "el sistema de traduccion funciona correctamente",
        "buenos dias, me llamo Victor y estudio ingenieria",
        "necesito que me pases el informe antes del viernes",
        "no entiendo bien lo que me estas pidiendo ahora",
    ],
    ("en", "es"): [
        "the translation system works correctly",
        "good morning, my name is Victor and I study engineering",
        "I need you to send me the report before Friday",
        "I do not quite understand what you are asking me",
    ],
}

# How many times the lifespan actually runs in one suite pass, counted by
# wrapping init_engines() rather than by reading fixture signatures - the
# WebSocket canary is parametrised by stage, so the two numbers differ.
LIFESPAN_TESTS_PER_SUITE = 25


def measure_load(pair: tuple[str, str]) -> dict:
    """Cost of opening one pair, with the artifact already in the local cache.

    The download is deliberately excluded: it happens once per machine, while
    this cost is paid on every process start.
    """
    start = time.perf_counter()
    _load_translator(pair, settings.MT_DEVICE)
    return {
        "pair": f"{pair[0]}->{pair[1]}",
        "load_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def measure_translate(engine: MTService, pair: tuple[str, str]) -> dict:
    source, target = pair
    state = MTState()
    per_phrase = []

    for text in CORPUS[pair]:
        engine._translate(state, text, source, target)  # warm the path
        elapsed = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            engine._translate(state, text, source, target)
            elapsed.append((time.perf_counter() - start) * 1000)
        per_phrase.append(
            {
                "words_in": len(text.split()),
                "median_ms": round(statistics.median(elapsed), 1),
                "max_ms": round(max(elapsed), 1),
                "output": engine._translate(state, text, source, target),
            }
        )

    return {
        "pair": f"{source}->{target}",
        "median_ms": round(statistics.median(p["median_ms"] for p in per_phrase), 1),
        "phrases": per_phrase,
    }


def measure_missing_eos(engine: MTService, pair: tuple[str, str]) -> dict:
    """The silent failure, priced.

    Reaches past the public method on purpose: the point is to run the exact
    tokenization a caller would write from the CTranslate2 docs, whose own
    example omits the terminator.
    """
    translator = engine._translators[pair]
    text = CORPUS[pair][2]
    rows = {}

    for label, suffix in (("with_eos", ["</s>"]), ("without_eos", [])):
        tokens = translator.source_sp.encode(text, out_type=str) + suffix
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            result = translator.engine.translate_batch([tokens])
            elapsed.append((time.perf_counter() - start) * 1000)
        output = translator.target_sp.decode(result[0].hypotheses[0])
        rows[label] = {
            "median_ms": round(statistics.median(elapsed), 1),
            "words_out": len(output.split()),
            "output_head": output[:90],
        }

    rows["cost_ratio"] = round(
        rows["without_eos"]["median_ms"] / rows["with_eos"]["median_ms"], 1
    )
    rows["input"] = text
    return rows


def main() -> None:
    if settings.MT_MODEL != "opus-mt":
        sys.exit(
            f"MT_MODEL is {settings.MT_MODEL!r}; run with MT_MODEL=opus-mt "
            f"so this measures the real backend"
        )

    engine = MTService(settings.MT_MODEL, settings.MT_DEVICE)
    pairs = sorted(engine._translators)

    loads = [measure_load(pair) for pair in pairs]
    per_process_ms = round(sum(row["load_ms"] for row in loads), 1)

    payload = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": "OPUS-MT es<->en via CTranslate2, on CPU",
        "evidence": (
            "NOT evidence for this project. MT targets a cloud GPU, so these "
            "numbers describe hardware that will not serve a request: they are "
            "a correctness check, not a latency result. Do not quote them in "
            "the report. The one figure with a use beyond this file is "
            "load.per_process_ms, which the ASR replacement will pay per test."
        ),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "settings": {
            "mt_model": settings.MT_MODEL,
            "mt_device": settings.MT_DEVICE,
            "pairs": [f"{s}->{t}" for s, t in pairs],
            "compute_type": engine._translators[pairs[0]].engine.compute_type,
        },
        "load": {
            "note": (
                "Download excluded - paid once per machine. This is the cost "
                "of a process start, and the lifespan runs once per "
                "WebSocket test, not once per suite."
            ),
            "per_pair": loads,
            "per_process_ms": per_process_ms,
            "lifespan_runs_per_suite": LIFESPAN_TESTS_PER_SUITE,
            "projected_suite_overhead_ms": round(
                per_process_ms * LIFESPAN_TESTS_PER_SUITE, 1
            ),
        },
        "translate": [measure_translate(engine, pair) for pair in pairs],
        "missing_eos": {
            f"{s}->{t}": measure_missing_eos(engine, (s, t)) for s, t in pairs
        },
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"compute type  {payload['settings']['compute_type']}")
    for row in loads:
        print(f"load {row['pair']:>8}  {row['load_ms']:.1f} ms")
    print(
        f"per process   {per_process_ms:.1f} ms "
        f"-> {payload['load']['projected_suite_overhead_ms'] / 1000:.1f} s "
        f"per suite run at {LIFESPAN_TESTS_PER_SUITE} lifespans"
    )
    print()
    for row in payload["translate"]:
        print(f"{row['pair']}  median {row['median_ms']:.1f} ms/segment")
        for phrase in row["phrases"]:
            print(f"    {phrase['median_ms']:>6.1f} ms  {phrase['output']}")
    print()
    for pair, row in payload["missing_eos"].items():
        print(
            f"{pair} missing </s>: {row['cost_ratio']}x slower, "
            f"{row['without_eos']['words_out']} words out vs "
            f"{row['with_eos']['words_out']}"
        )
    print(f"\nwritten to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
