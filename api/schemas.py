"""Схемы валидации API.

Контракт наружу описан здесь и только здесь: доменные модели (Record, Reply,
PersonaCard) в ответы не отдаются напрямую, чтобы изменение хранилища не
ломало клиента.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import get_settings
from memory.record import Record
from memory.session import SessionInfo
from prompt_builder import PersonaCard
from services.dialogue import OpenedSession, Reply

# Ограничения входа читаются один раз: схемы строятся при импорте.
_limits = get_settings()


class UserRequest(BaseModel):
    """Общая часть: пользователь известен по идентификатору, без аутентификации."""

    user_id: str = Field(min_length=1, max_length=_limits.api_user_id_max_chars)


class NewSessionRequest(UserRequest):
    # Не указана — берётся DEFAULT_PERSONA из конфига.
    persona: str | None = Field(
        default=None, min_length=1, max_length=_limits.api_persona_max_chars
    )


class CloseSessionRequest(UserRequest):
    pass


class ChatRequest(UserRequest):
    session_id: str = Field(min_length=1, max_length=_limits.api_session_id_max_chars)
    message: str = Field(min_length=1, max_length=_limits.api_message_max_chars)


class NewSessionResponse(BaseModel):
    session_id: str
    persona: str
    # Стартовая зарисовка рассказчика: показывается отдельно от диалога.
    scene: str
    message: str

    @classmethod
    def of(cls, opened: OpenedSession) -> "NewSessionResponse":
        return cls(
            session_id=opened.session_id,
            persona=opened.persona,
            scene=opened.scene,
            message=opened.message,
        )


class ChatResponse(BaseModel):
    reply: str
    user_turn: int
    reinjected: bool
    memories_used: int

    @classmethod
    def of(cls, reply: Reply) -> "ChatResponse":
        return cls(
            reply=reply.text,
            user_turn=reply.user_turn,
            reinjected=reply.reinjected,
            memories_used=reply.memories_used,
        )


class RecordResponse(BaseModel):
    record_id: str
    persona: str
    counterpart: str
    story: str
    response: str
    outcome: str
    paid: bool | None = None
    fee: str | None = None
    debt: str | None = None
    nickname: str | None = None

    @classmethod
    def of(cls, record: Record) -> "RecordResponse":
        return cls(**record.model_dump(include=set(cls.model_fields)))


class SessionItem(BaseModel):
    session_id: str
    persona: str | None

    @classmethod
    def of(cls, info: SessionInfo) -> "SessionItem":
        return cls(session_id=info.session_id, persona=info.persona)


class SessionsResponse(BaseModel):
    sessions: list[SessionItem]

    @classmethod
    def of(cls, infos: list[SessionInfo]) -> "SessionsResponse":
        return cls(sessions=[SessionItem.of(info) for info in infos])


class PersonaUIResponse(BaseModel):
    title: str
    placeholder: str
    new_session: str
    close_session: str
    archived: str
    accent: str


class PersonaResponse(BaseModel):
    name: str
    display_name: str
    ui: PersonaUIResponse


class PersonasResponse(BaseModel):
    default: str
    personas: list[PersonaResponse]

    @classmethod
    def of(cls, personas: list[PersonaCard], *, default: str) -> "PersonasResponse":
        return cls(
            default=default,
            personas=[
                PersonaResponse(
                    name=card.name,
                    display_name=card.display_name,
                    ui=PersonaUIResponse(**card.settings.ui.model_dump()),
                )
                for card in personas
            ],
        )


class HealthResponse(BaseModel):
    redis: bool
    longterm: bool
    chat_model: str
    default_persona: str
    # N переинжекта по персонам.
    reinject_every: dict[str, int]
