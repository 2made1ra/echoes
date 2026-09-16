"""Слой хранилищ: одна точка входа к session- и long-term памяти.

Выше по стеку (сервис, роуты) не должно быть знания о том, что session-память
живёт в Redis, а записи прошлых сессий — в Qdrant, и что запись перед
укладкой надо превратить в вектор. Здесь же держится и деградация: если
Qdrant не поднят, сервис продолжает работать без long-term памяти, а не падает.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from config import Settings
from memory.longterm import LongTermMemory
from memory.record import Record
from memory.session import Message, SessionInfo, SessionMemory

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """Всё, что нужно хранилищу от LLM-клиента."""

    async def embed_one(self, text: str) -> list[float]: ...


class MemoryStore:
    def __init__(
        self,
        session: SessionMemory,
        longterm: LongTermMemory,
        embedder: Embedder,
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> None:
        self._session = session
        self._longterm = longterm
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold
        self.longterm_ready = False

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> "MemoryStore":
        return cls(
            SessionMemory.from_url(
                settings.redis_url, ttl_seconds=settings.session_ttl_days * 24 * 60 * 60
            ),
            LongTermMemory.from_url(
                settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim
            ),
            embedder,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )

    async def connect(self) -> None:
        try:
            await self._longterm.ensure_collection()
            self.longterm_ready = True
        except Exception:
            # Qdrant может быть не поднят на демо-машине. Не маскируем: пишем в
            # лог и честно показываем деградацию в /health.
            log.exception("Qdrant недоступен — long-term память отключена")
            self.longterm_ready = False

    async def close(self) -> None:
        await self._session.close()
        await self._longterm.close()

    async def session_alive(self) -> bool:
        try:
            return bool(await self._session.ping())
        except Exception:
            return False

    # --- session-память -------------------------------------------------

    async def start_session(self, user_id: str, persona: str, first_message: str) -> str:
        session_id = str(uuid.uuid4())
        await self._session.register(user_id, session_id, persona)
        # Первая реплика кладётся в историю как реплика ассистента: она задаёт
        # длину и тон, которые модель копирует весь диалог.
        await self._session.append(user_id, session_id, "assistant", first_message)
        return session_id

    async def draw_opening(self, user_id: str, persona: str, deck: str, size: int) -> int:
        return await self._session.draw_opening(user_id, persona, deck, size)

    async def session_persona(self, user_id: str, session_id: str) -> str | None:
        return await self._session.persona(user_id, session_id)

    async def history(self, user_id: str, session_id: str, limit: int | None = None) -> list[Message]:
        return await self._session.history(user_id, session_id, limit=limit)

    async def next_user_turn(self, user_id: str, session_id: str) -> int:
        return await self._session.next_user_turn(user_id, session_id)

    async def record_exchange(self, user_id: str, session_id: str, question: str, reply: str) -> None:
        await self._session.append(user_id, session_id, "user", question)
        await self._session.append(user_id, session_id, "assistant", reply)

    async def sessions(self, user_id: str) -> list[SessionInfo]:
        return await self._session.sessions(user_id)

    # --- long-term память -----------------------------------------------

    async def relevant_memories(self, user_id: str, persona: str, query: str) -> list[Record]:
        """Ретрив не должен ронять диалог: long-term память — приятное дополнение."""
        if not self.longterm_ready:
            return []
        try:
            vector = await self._embedder.embed_one(query)
            return await self._longterm.search(
                user_id,
                persona,
                vector,
                top_k=self._top_k,
                score_threshold=self._score_threshold,
            )
        except Exception:
            log.exception("ретрив из long-term памяти не удался — продолжаем без неё")
            return []

    async def save_record(self, record: Record) -> None:
        vector = await self._embedder.embed_one(record.embedding_text())
        await self._longterm.store(record, vector)
