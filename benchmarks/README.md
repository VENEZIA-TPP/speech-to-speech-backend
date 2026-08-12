# Benchmarks

Reproducible measurements, with their results committed as JSON next to the
script that produced them. A performance claim in this project cites a file in
`results/` or it is not a claim.

```bash
.venv/bin/python benchmarks/vad_endpoint.py              # results/vad_endpoint.json
MT_MODEL=opus-mt .venv/bin/python benchmarks/mt_opus.py  # results/mt_opus_local.json
```

## What counts as evidence here, and what does not

Almost nothing measured on a development machine is evidence for this project:
ASR, MT and TTS are meant to run on a cloud GPU, so a local timing of them
describes hardware that will never serve a request. Those runs are correctness
checks wearing a stopwatch.

**Segmentation is the exception.** Silero VAD runs on a CPU thread in
production exactly as it does here, so `vad_endpoint.py` measures the same work
on the same kind of hardware. Its per-frame cost transfers.

The audio is `tests/fixtures/es_sistema.wav`, which is text-to-speech rather
than a recorded human. That makes these results evidence about **segmentation
timing and segmentation behaviour**, and no evidence at all about transcription
or translation quality — WER and BLEU need a human corpus and cannot be
substituted for.

## `vad_endpoint.py`

- **`frame_cost`** — what the model costs per 32 ms frame, and per arriving
  chunk. This is the number that decides whether segmentation can run on the
  event loop or has to be handed to a thread.
- **`endpoint`** — how a pause of each length is classified. This is the
  latency/quality dial in `VAD_MIN_SILENCE_MS`: the shortest gap that produces
  two segments is where the setting actually sits, as opposed to where the
  configuration says it sits.
- **`ceiling`** — continuous speech past `VAD_MAX_SPEECH_MS`, showing the
  bounded worst case rather than an unbounded one.

Warmup is measured separately and never mixed into the per-frame figures.

## `mt_opus.py` — not evidence, and its filename says so

Every timing in `results/mt_opus_local.json` is a correctness check. The
`_local` suffix and the file's own `evidence` field both say it, because a JSON
full of milliseconds is exactly the thing that gets quoted without its context.

Two results in it are worth reading anyway:

- **`load.per_process_ms`** — what opening the models costs at process start.
  It matters beyond this file because the lifespan runs **once per WebSocket
  test**, not once per suite: 25 times per pass, counted by instrumenting
  `init_engines()` rather than by reading fixture signatures. Whatever the next
  backend costs to open gets multiplied by that.
- **`missing_eos`** — the cost of forgetting the `</s>` that terminates the
  source token list. The decoder then never emits EOS and runs to its decoding
  limit, so the failure arrives as 256 words of degenerating text rather than
  as an exception. Measured at ~22× the wall clock of a correct call, which is
  the shape it takes in a profile.
