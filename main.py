"""Точка сборки приложения: конфиг → сервис → роуты. Логики здесь нет."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.errors import register_error_handlers
from api.routes import dialogue_router, system_router
from services.dialogue import DialogueService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = DialogueService.from_settings()
    await service.start()
    app.state.service = service
    try:
        yield
    finally:
        await service.close()


def create_app() -> FastAPI:
    app = FastAPI(title="echoes", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(dialogue_router)
    app.include_router(system_router)
    return app


app = create_app()
