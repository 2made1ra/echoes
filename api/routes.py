"""Роуты: только валидация входа, вызов сервиса и упаковка ответа.

Ни сборки промпта, ни обращений к Redis/Qdrant, ни try/except вокруг LLM —
доменные ошибки превращаются в HTTP-коды обработчиками из api.errors.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from api.deps import ServiceDep
from api.schemas import (
    ChatRequest,
    ChatResponse,
    CloseSessionRequest,
    HealthResponse,
    NewSessionRequest,
    NewSessionResponse,
    PersonasResponse,
    RecordResponse,
    SessionsResponse,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

dialogue_router = APIRouter(prefix="/api", tags=["dialogue"])
system_router = APIRouter(tags=["system"])


@dialogue_router.get("/personas", response_model=PersonasResponse)
async def list_personas(service: ServiceDep) -> PersonasResponse:
    return PersonasResponse.of(service.personas(), default=service.default_persona)


@dialogue_router.post("/sessions", response_model=NewSessionResponse)
async def create_session(payload: NewSessionRequest, service: ServiceDep) -> NewSessionResponse:
    return NewSessionResponse.of(await service.open_session(payload.user_id, payload.persona))


@dialogue_router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, service: ServiceDep) -> ChatResponse:
    return ChatResponse.of(
        await service.reply(payload.user_id, payload.session_id, payload.message)
    )


@dialogue_router.post("/sessions/{session_id}/close", response_model=RecordResponse)
async def close_session(
    session_id: str, payload: CloseSessionRequest, service: ServiceDep
) -> RecordResponse:
    return RecordResponse.of(await service.close_session(payload.user_id, session_id))


@dialogue_router.get("/sessions", response_model=SessionsResponse)
async def list_sessions(user_id: str, service: ServiceDep) -> SessionsResponse:
    return SessionsResponse.of(await service.sessions(user_id))


@system_router.get("/health", response_model=HealthResponse)
async def health(service: ServiceDep) -> HealthResponse:
    return HealthResponse(**await service.health())


@system_router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
