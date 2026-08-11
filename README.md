# Speech-to-Speech Translation - Backend

Real-time speech translation pipeline: **ASR -> MT -> TTS** (all stages stubbed).

TPP - Ingeniería en Informática, FIUBA.  
Authors: Victor Cipriano (106593) · Ricardo Contreras (107239)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI (async) + WebSocket |
| Database | PostgreSQL 15 + SQLAlchemy 2 async + Alembic |
| ASR | Whisper / faster-whisper *(stub for now)* |
| MT | MarianMT / NLLB-200 *(stub for now)* |
| TTS | XTTS v2 / OpenVoice V2 / Chatterbox *(stub for now)* |
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
│   ├── asr_service.py                     # Whisper stub -> replace with real model
│   ├── mt_service.py                      # MarianMT/NLLB stub -> replace with real model
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

```
Client  ->  Server   Sec-WebSocket-Protocol: <ws_token>   (handshake; from POST /sessions/ response)
Client  ->  Server   binary frame   (WAV chunk, 16 kHz mono pcm_s16le)
Server  ->  Client   JSON frame     PipelineResult (texts, metrics, watermark info)
Server  ->  Client   binary frame   synthesized WAV (iff synthesized_audio_size_bytes > 0)
Client  ->  Server   text "END"     signals end of stream
Server  ->  Client   JSON           { "status": "completed", ... }
```

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

---

## Integrating a real ASR model

1. Install: `pip install faster-whisper`
2. In [app/services/asr_service.py](app/services/asr_service.py) replace `_transcribe()` with faster-whisper inference.
3. Set `ASR_MODEL=whisper-base` (or larger) and `ASR_DEVICE=cuda` in `.env`.

## Integrating a real MT model

1. Install: `pip install transformers sentencepiece`
2. In [app/services/mt_service.py](app/services/mt_service.py) replace `_translate()` with MarianMT or NLLB-200 inference.
3. Set `MT_MODEL=Helsinki-NLP/opus-mt-en-es` in `.env`.

## Integrating a real TTS model

1. Model TBD (XTTS v2 / OpenVoice V2 / Chatterbox candidates).
2. In [app/services/tts_service.py](app/services/tts_service.py) replace `_synthesize()` with real inference; keep the `_apply_watermark()` hook in the output path (ethical requirement).
3. Set `TTS_MODEL=...` and `TTS_DEVICE=cuda` in `.env`.

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
ASR_MODEL=stub        # whisper-base | whisper-small | whisper-medium | whisper-large-v3
ASR_DEVICE=cpu        # cuda for GPU

# MT
MT_MODEL=stub         # Helsinki-NLP/opus-mt-en-es | facebook/nllb-200-distilled-600M
MT_DEVICE=cpu

# TTS
TTS_MODEL=stub        # XTTS v2 | OpenVoice V2 | Chatterbox (model TBD)
TTS_DEVICE=cpu

# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION_MS=3000
```
