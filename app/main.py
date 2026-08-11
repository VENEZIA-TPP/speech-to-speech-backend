import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.api import api_router
from app.core.config import settings
from app.core.middleware import setup_middlewares
from app.db.session import engine
from app.dependencies import init_engines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Speech-to-Speech Translation API")
    logger.info(
        f"ASR model: {settings.ASR_MODEL} | MT model: {settings.MT_MODEL} "
        f"| TTS model: {settings.TTS_MODEL}"
    )
    # One process, many sessions: the three engines are built here, once,
    # before the app accepts traffic. `uvicorn --workers N` means N processes
    # and N times the VRAM.
    init_engines()
    yield
    logger.info("Shutting down Speech-to-Speech Translation API")
    # ponytail: no engine teardown while all three are stubs - there is
    # nothing to release. CUDA/ONNX release lands with the real backend
    # (fase-1-concurrencia.md C8).
    await engine.dispose()


app = FastAPI(
    title="Speech-to-Speech Translation API",
    description="Real-time speech translation pipeline: ASR -> MT -> TTS",
    version="0.1.0",
    lifespan=lifespan,
)

setup_middlewares(app)
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "service": "Speech-to-Speech Translation API",
        "version": "0.1.0",
        "docs": "/docs",
    }
