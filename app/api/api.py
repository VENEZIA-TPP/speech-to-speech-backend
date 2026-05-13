from fastapi import APIRouter

from app.api.controller import health, sessions, pipeline

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
