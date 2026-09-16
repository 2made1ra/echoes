"""Session-память: сырая история текущего диалога в Redis.

Это не "запоминание фактов", а подпитка контекстом, удерживающая модель в
персонажном распределении: последние N реплик работают как примеры диалога,
написанные самой моделью. Поэтому персона фиксируется при создании сессии и
не меняется: чужие реплики в истории модель начала бы копировать.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

Message = dict[str, str]

# Ключи держим с user_id в префиксе: аутентификации в MVP нет, но данные
# сразу ключуются по пользователю, чтобы развернуть сервис на несколько
# человек можно было без переделки схемы.
_MESSAGES = "chat:session:{user_id}:{session_id}:messages"
_TURNS = "chat:session:{user_id}:{session_id}:user_turns"
_META = "chat:session:{user_id}:{session_id}:meta"
_SESSIONS = "chat:user:{user_id}:sessions"
# Колода первых реплик пользователя: ещё не показанные номера и последний.
_DECK = "chat:user:{user_id}:openings:{persona}:{deck}"
_DECK_LAST = "chat:user:{user_id}:openings:{persona}:last"


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    # None — сессия истекла, от неё остался только идентификатор.
    persona: str | None


class SessionMemory:
    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "SessionMemory":
        return cls(Redis.from_url(url, decode_responses=True), **kwargs)

    async def register(self, user_id: str, session_id: str, persona: str) -> None:
        meta = _META.format(user_id=user_id, session_id=session_id)
        await self._redis.hset(meta, mapping={"persona": persona})
        await self._redis.expire(meta, self._ttl)
        await self._redis.sadd(_SESSIONS.format(user_id=user_id), session_id)

    async def persona(self, user_id: str, session_id: str) -> str | None:
        meta = _META.format(user_id=user_id, session_id=session_id)
        return await self._redis.hget(meta, "persona")

    async def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        key = _MESSAGES.format(user_id=user_id, session_id=session_id)
        await self._redis.rpush(key, json.dumps({"role": role, "content": content}))
        await self._redis.expire(key, self._ttl)
        # Метаданные живут столько же, сколько история.
        await self._redis.expire(_META.format(user_id=user_id, session_id=session_id), self._ttl)

    async def history(self, user_id: str, session_id: str, limit: int | None = None) -> list[Message]:
        key = _MESSAGES.format(user_id=user_id, session_id=session_id)
        start = -limit if limit else 0
        raw = await self._redis.lrange(key, start, -1)
        return [json.loads(item) for item in raw]

    async def next_user_turn(self, user_id: str, session_id: str) -> int:
        """Порядковый номер новой реплики пользователя — по нему считается переинжект."""
        key = _TURNS.format(user_id=user_id, session_id=session_id)
        turn = await self._redis.incr(key)
        await self._redis.expire(key, self._ttl)
        return int(turn)

    async def sessions(self, user_id: str) -> list[SessionInfo]:
        ids = sorted(await self._redis.smembers(_SESSIONS.format(user_id=user_id)))
        return [SessionInfo(session_id=sid, persona=await self.persona(user_id, sid)) for sid in ids]

    async def drop(self, user_id: str, session_id: str) -> None:
        await self._redis.delete(
            _MESSAGES.format(user_id=user_id, session_id=session_id),
            _TURNS.format(user_id=user_id, session_id=session_id),
            _META.format(user_id=user_id, session_id=session_id),
        )
        await self._redis.srem(_SESSIONS.format(user_id=user_id), session_id)

    async def draw_opening(self, user_id: str, persona: str, deck: str, size: int) -> int:
        """Номер следующей первой реплики: без повторов, пока колода не кончится.

        Колода тасуется заново, когда пуста; на стыке колод последняя
        показанная реплика не выпадает первой. deck — отпечаток набора
        реплик: после его правки старая колода просто перестаёт читаться.
        """
        key = _DECK.format(user_id=user_id, persona=persona, deck=deck)
        last_key = _DECK_LAST.format(user_id=user_id, persona=persona)
        raw = await self._redis.lpop(key)
        if raw is None or int(raw) >= size:
            last = await self._redis.get(last_key)
            order = random.sample(range(size), size)
            if size > 1 and last is not None and order[0] == int(last):
                order[0], order[-1] = order[-1], order[0]
            raw, rest = order[0], order[1:]
            if rest:
                await self._redis.rpush(key, *rest)
        await self._redis.expire(key, self._ttl)
        await self._redis.set(last_key, int(raw), ex=self._ttl)
        return int(raw)

    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    async def close(self) -> None:
        await self._redis.aclose()
