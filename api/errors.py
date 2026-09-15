"""Перевод доменных ошибок в HTTP-коды — единственное место, где они встречаются."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from services.errors import (
    DialogueError,
    LongTermUnavailable,
    NothingToArchive,
    ProviderUnavailable,
    SessionNotFound,
    UnknownPersona,
)

log = logging.getLogger(__name__)

_STATUS: tuple[tuple[type[DialogueError], int], ...] = (
    (SessionNotFound, 404),
    (UnknownPersona, 422),
    (NothingToArchive, 400),
    (LongTermUnavailable, 503),
    (ProviderUnavailable, 502),
)


def status_for(exc: DialogueError) -> int:
    for error_type, status in _STATUS:
        if isinstance(exc, error_type):
            return status
    return 500


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DialogueError)
    async def domain_error(request: Request, exc: DialogueError) -> JSONResponse:
        status = status_for(exc)
        if status >= 500:
            log.error("%s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(RedisError)
    async def redis_unavailable(request: Request, exc: RedisError) -> JSONResponse:
        """Session-память недоступна — говорим прямо, а не пятисоткой с трейсбеком."""
        log.error("Redis недоступен: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"Redis недоступен (docker compose up -d): {exc}"},
        )
