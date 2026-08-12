# Speech-to-Speech Translation - Backend

Real-time speech translation pipeline: **ASR -> MT -> TTS**. ASR and MT have
real backends behind env vars (`ASR_MODEL=parakeet`, `MT_MODEL=opus-mt`); every
stage keeps a `stub` backend, which is still the default for all three.

TPP - Ingeniería en Informática, FIUBA.  
Authors: Victor Cipriano (106593) · Ricardo Contreras (107239)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI (async) + WebSocket |
| Database | PostgreSQL 15 + SQLAlchemy 2 async + Alembic |
| ASR | Parakeet TDT 0.6B v3 via onnx-asr *(`stub` by default)* |
| MT | OPUS-MT es↔en via CTranslate2 *(`stub` by default)* |
| TTS | Chatterbox Multilingual planned *(stub for now)* |
| Testing | pytest + pytest-asyncio + SQLite in-memory |
| Container | Docker + Docker Compose |

---

## Project structure

```
app/
├── main.py
├── core/
│   ├── config.py                          # env vars + AI model settings
│   └── middleware.py                      # CORS, request logging
├── db/
│   ├── base.py
│   └── session.py
├── models/
│   ├── translation_session.py             # session (source/target lang, status)
│   ├── transcription.py                   # ASR output per chunk
│   └── translation.py                     # MT output per chunk
├── schemas/
│   ├── translation_session.py
│   ├── transcription.py
│   └── translation.py                     # includes PipelineResult
├── repositories/
│   ├── interfaces/                        # abstract interfaces (ABC)
│   │   ├── translation_session_repository.py
│   │   ├── transcription_repository.py
│   │   └── translation_repository.py
│   ├── translation_session_repository.py  # SQLAlchemy implementation
│   ├── transcription_repository.py
│   └── translation_repository.py
├── services/
│   ├── asr_service.py                     # Parakeet via onnx-asr (or stub)
│   ├── mt_service.py                      # OPUS-MT via CTranslate2 (or stub)
│   ├── session_service.py                 # session CRUD + history queries
│   ├── translation_pipeline_service.py   # ASR -> MT -> TTS orchestrator
│   └── tts_service.py                     # TTS stub -> replace with real model
├── api/
│   ├── api.py
│   └── controller/
│       ├── health.py                      # GET /health
│       ├── sessions.py                    # REST session management
│       └── pipeline.py                    # WS  /pipeline/ws/{session_id}
└── dependencies.py                        # DI wiring (FastAPI Depends)
tests/
├── conftest.py                            # SQLite in-memory fixtures
├── test_sessions.py                       # session CRUD tests
└── test_pipeline.py                       # stub services + pipeline + WS e2e tests
```

---

## API endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/` | Service info | none |
| GET | `/health/` | Health + model status | none |
| POST | `/sessions/` | Create translation session | none (mints the `ws_token`) |
| GET | `/sessions/{id}` | Get session | `Authorization: Bearer <ws_token>` |
| PATCH | `/sessions/{id}/complete` | Mark session as completed | `Authorization: Bearer <ws_token>` |
| DELETE | `/sessions/{id}` | Delete session | `Authorization: Bearer <ws_token>` |
| GET | `/sessions/{id}/transcriptions` | ASR results for session | `Authorization: Bearer <ws_token>` |
| GET | `/sessions/{id}/translations` | MT results for session | `Authorization: Bearer <ws_token>` |
| WS | `/pipeline/ws/{session_id}` | Real-time audio streaming | `Sec-WebSocket-Protocol: <ws_token>` |

`ws_token` is returned once, in the `POST /sessions/` response body — every other
`/sessions/{id}...` route 401s without it (unknown or already-deleted session ids also
401, never 404, to avoid leaking which ids exist).

Interactive docs: `http://localhost:8000/docs`

---

## WebSocket protocol

Handshake: `Sec-WebSocket-Protocol: <ws_token>` (the token comes from the
`POST /sessions/` response). The client then streams binary WAV chunks
(16 kHz mono pcm_s16le) and sends `{"type":"input_audio.commit"}` to end the
stream.

Every server frame is a JSON event with a `type`, except the synthesized audio,
which travels as a bare binary frame right after the `audio.delta` announcing
it. Events are keyed by `segment_index`, **not** by the chunk the client sent:
one segment per chunk is what the pipeline happens to do today, and the client
must not rely on it.

| Event | Payload |
|---|---|
| `session.created` | `session_id` |
| `transcription.completed` | `segment_index`, `transcript`, `language_code` |
| `translation.completed` | `segment_index`, `text`, `target_language` |
| `audio.delta` | `segment_index`, `seq`, `size_bytes` — followed by one binary frame |
| `audio.done` | `segment_index`, `watermarked`, `watermark_method` |
| `segment.metrics` | `segment_index`, `asr_ms`, `mt_ms`, `tts_ms`, `e2e_ms` |
| `session.completed` | `session_id`, `total_segments` |
| `error` | `code`, `message`, `segment_index` |

A malformed control frame gets `error` with `code: "invalid_event"` and the
connection stays open; a pipeline failure gets `code: "pipeline_failed"`, marks
the session failed and closes. A binary frame over the size limit closes the
connection with 1009.

---

## Quickstart (Docker)

```bash
cp .env.example .env
docker-compose up --build
```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run a **single process**. The ASR/MT/TTS engines are built once per process at
startup, so `uvicorn --workers N` loads N copies of the models and multiplies the
VRAM by N. Scale sessions per process, not workers.

## Migrations

```bash
make migrate MSG="description"   # generate migration
make push                        # apply pending migrations
make rollback                    # revert last migration
```

## Tests

```bash
make test         # or: pytest tests/ -v
```

Each test uses an isolated SQLite in-memory database.

### Checking it by hand

The suite runs in-process against SQLite. Three tools exercise the real server
instead:

| Tool | What it is for |
|---|---|
| `scripts/consola.html` | Speak into the microphone from a browser and watch the WebSocket protocol live: transcription, translation, synthesized audio and per-segment metrics. Serve it (`cd scripts && python3 -m http.server 3000`) rather than opening the file directly — `http://localhost:3000` is an allowed CORS origin, a `file://` page is not. |
| `scripts/smoke_test.py` | Non-interactive end-to-end run against a recorded fixture. What CI would use. |
| `scripts/demo_vad.py` | Segmentation only, on your own recording. `--barrido` compares silence thresholds, which is how the endpoint threshold gets calibrated against a real speaker. |

---

## Running the real models

- **ASR**: set `ASR_MODEL=parakeet`. First startup downloads the pinned int8
  ONNX export of Parakeet TDT 0.6B v3 (~650 MB, cached by `huggingface_hub`);
  the process refuses to start if the weights cannot be fetched. An
  unrecognized value also refuses to start instead of silently falling back
  to the stub.
- **MT**: set `MT_MODEL=opus-mt`. Downloads one pinned CTranslate2 model per
  pair in `SUPPORTED_LANGUAGE_PAIRS` (~155 MB each). Same startup semantics.
- **TTS**: still a stub. The ratified candidate is Chatterbox Multilingual; a
  real `_synthesize()` must keep the `_apply_watermark()` hook in the output
  path (ethical requirement).

`ASR_DEVICE`/`MT_DEVICE`/`TTS_DEVICE` accept `cpu` (default) or `cuda`. CPU
runs the same code path as production and is correctness-only: latency numbers
measured without the target GPU are not evidence.

---

## Environment variables

```bash
# Database
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_DB=s2st_db
DATABASE_HOST=db
DATABASE_PORT=5433
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/s2st_db

# Server
PORT=8000
HOST=0.0.0.0
PUBLIC_URL=http://localhost:8000

# ASR
ASR_MODEL=stub        # stub | parakeet
ASR_DEVICE=cpu        # cuda for GPU

# MT
MT_MODEL=stub         # stub | opus-mt
MT_DEVICE=cpu

# TTS
TTS_MODEL=stub        # stub (Chatterbox planned)
TTS_DEVICE=cpu

# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION_MS=3000
```

---

## Model licenses & attribution

| Stage | Model | License | Artifact served |
|-------|-------|---------|-----------------|
| ASR | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) © NVIDIA | **CC-BY-4.0** | ONNX export [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx), pinned by commit |
| MT | [OPUS-MT es-en / en-es](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) by Helsinki-NLP | Apache-2.0 | CTranslate2 conversions [`michaelfeil/ct2fast-opus-mt-{es-en,en-es}`](https://huggingface.co/michaelfeil/ct2fast-opus-mt-es-en), pinned by commit |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | ONNX weights vendored under `app/pipeline/weights/` |

The ASR weights are NVIDIA's Parakeet TDT 0.6B v3, used under the
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license; this notice
is the attribution that license requires. The runtime loads a third-party ONNX
conversion of those weights (by [@istupakov](https://huggingface.co/istupakov),
same license), pinned by commit SHA in `app/services/asr_service.py`.
