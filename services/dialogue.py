"""Бизнес-логика диалога с персонажем.

Здесь живёт единственный путь запроса: персона сессии → история → память →
счётчик реплик → сборка блоков → вызов LLM → запись в память. Ни FastAPI, ни
HTTP-кодов тут нет — сервис одинаково вызывается из API и из скриптов.

Персона выбирается при создании сессии и дальше берётся из неё: сменить
персону можно только новой сессией. Стартовая зарисовка отдаётся клиенту и
нигде не сохраняется — в контекст модели она не попадает.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Mapping

from config import Settings, get_settings
from llm_client import LLMClient
from memory.record import Record
from memory.session import SessionInfo
from memory.store import MemoryStore
from memory.summarizer import summarize_session
from prompt_builder import (
    PersonaCard,
    build_messages,
    default_persona_name,
    load_personas,
    should_reinject,
)
from services.errors import (
    LongTermUnavailable,
    NothingToArchive,
    ProviderUnavailable,
    SessionNotFound,
    UnknownPersona,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenedSession:
    session_id: str
    persona: str
    # Зарисовка рассказчика для интерфейса; в историю сессии не пишется.
    scene: str
    message: str


@dataclass(frozen=True)
class Reply:
    text: str
    user_turn: int
    reinjected: bool
    memories_used: int


class DialogueService:
    def __init__(
        self,
        settings: Settings,
        personas: Mapping[str, PersonaCard],
        llm: LLMClient,
        store: MemoryStore,
    ) -> None:
        self._default_persona = default_persona_name(dict(personas), settings.default_persona)
        for persona in personas.values():
            Record.check_template(persona.memory.record)
        self._settings = settings
        self._personas = dict(personas)
        self._llm = llm
        self._store = store

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "DialogueService":
        settings = settings or get_settings()
        personas = load_personas(settings.persona_dir)
        llm = LLMClient(settings)
        return cls(settings, personas, llm, MemoryStore.from_settings(settings, llm))

    async def start(self) -> None:
        await self._store.connect()

    async def close(self) -> None:
        await self._llm.close()
        await self._store.close()

    @property
    def default_persona(self) -> str:
        return self._default_persona

    def personas(self) -> list[PersonaCard]:
        return list(self._personas.values())

    async def open_session(self, user_id: str, persona: str | None = None) -> OpenedSession:
        card = self._persona(persona or self._default_persona)
        session_id = await self._store.start_session(user_id, card.name, card.first_message)
        return OpenedSession(
            session_id=session_id,
            persona=card.name,
            scene=random.choice(card.scenes),
            message=card.first_message,
        )

    async def reply(self, user_id: str, session_id: str, message: str) -> Reply:
        persona = await self._session_persona(user_id, session_id)
        history = await self._store.history(
            user_id, session_id, limit=self._settings.session_window
        )
        if not history:
            raise SessionNotFound(session_id)

        records = await self._store.relevant_memories(user_id, persona.name, message)
        turn = await self._store.next_user_turn(user_id, session_id)
        reinject = should_reinject(turn, persona.reinject_every(self._settings.reinject_every))

        messages = build_messages(
            persona,
            history=history,
            user_message=message,
            memories=[record.render(persona.memory.record) for record in records],
            reinject=reinject,
            window=self._settings.session_window,
        )
        text = await self._ask(messages, stage="ответ")
        await self._store.record_exchange(user_id, session_id, message, text)

        return Reply(text=text, user_turn=turn, reinjected=reinject, memories_used=len(records))

    async def close_session(self, user_id: str, session_id: str) -> Record:
        """Сжать сессию в запись и положить в long-term память персоны."""
        persona = await self._session_persona(user_id, session_id)
        history = await self._store.history(user_id, session_id)
        if len(history) < 2:
            raise NothingToArchive()
        if not self._store.longterm_ready:
            raise LongTermUnavailable()

        try:
            record = await summarize_session(
                self._llm,
                prompt=persona.memory.summarizer,
                user_id=user_id,
                session_id=session_id,
                persona=persona.name,
                history=history,
            )
            await self._store.save_record(record)
        except Exception as exc:
            log.exception("не удалось занести запись в long-term память")
            raise ProviderUnavailable("суммаризация/индексация", exc) from exc

        log.info("запись занесена (%s): %s", persona.name, record.render(persona.memory.record))
        return record

    async def sessions(self, user_id: str) -> list[SessionInfo]:
        return await self._store.sessions(user_id)

    async def health(self) -> dict[str, object]:
        return {
            "redis": await self._store.session_alive(),
            "longterm": self._store.longterm_ready,
            "chat_model": self._settings.chat_model,
            "default_persona": self._default_persona,
            "reinject_every": {
                name: card.reinject_every(self._settings.reinject_every)
                for name, card in self._personas.items()
            },
        }

    def _persona(self, name: str) -> PersonaCard:
        try:
            return self._personas[name]
        except KeyError:
            raise UnknownPersona(name, sorted(self._personas)) from None

    async def _session_persona(self, user_id: str, session_id: str) -> PersonaCard:
        name = await self._store.session_persona(user_id, session_id)
        if name is None:
            raise SessionNotFound(session_id)
        return self._persona(name)

    async def _ask(self, messages: list[dict[str, str]], *, stage: str) -> str:
        """Единственное место, где вызывается LLM: ошибку не маскируем."""
        try:
            return await self._llm.chat(messages)
        except Exception as exc:
            log.exception("вызов LLM не удался (%s)", stage)
            raise ProviderUnavailable(stage, exc) from exc
