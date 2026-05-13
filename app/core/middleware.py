import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        # log request
        logger.info(f"REQUEST: {request.method} {request.url.path}")

        # procesar request
        response = await call_next(request)

        # log response
        logger.info(
            f"RESPONSE: {request.method} {request.url.path} "
            f"[{response.status_code}]"
        )

        return response


def setup_middlewares(app: FastAPI):
    """configura todos los middlewares de la app"""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: luego cambiar por la direccion
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    # log
    app.add_middleware(LoggingMiddleware)
