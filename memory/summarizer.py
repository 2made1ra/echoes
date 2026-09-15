"""Суммаризация сессии в запись long-term памяти.

Служебный вызов: модель работает не в образе персонажа, а как аккуратный
летописец. Карточка персоны сюда не подмешивается намеренно — иначе в
long-term память попадут подколы вместо фактов. От персоны берётся только
промпт суммаризатора из её memory.md.
"""

from __future__ import annotations

import json
import logging
import re

from llm_client import LLMClient, Message
from memory.record import Record

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> dict:
    cleaned = _FENCE_RE.sub("", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"суммаризатор вернул не JSON: {raw[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def _render_log(history: list[Message]) -> str:
    # Роли нейтральные: кто есть кто, объясняет промпт персоны.
    names = {"user": "Пользователь", "assistant": "Персонаж"}
    lines = [f"{names.get(m['role'], m['role'])}: {m['content']}" for m in history]
    return "\n".join(lines)


async def summarize_session(
    llm: LLMClient,
    *,
    prompt: str,
    user_id: str,
    session_id: str,
    persona: str,
    history: list[Message],
) -> Record:
    if not history:
        raise ValueError("нечего суммаризировать: пустая сессия")

    raw = await llm.chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": _render_log(history)},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    data = _extract_json(raw)
    return Record(
        user_id=user_id,
        session_id=session_id,
        persona=persona,
        counterpart=str(data.get("counterpart") or "неизвестный"),
        story=str(data.get("story") or ""),
        response=str(data.get("response") or ""),
        outcome=str(data.get("outcome") or ""),
        # Поле оплаты есть только у персон, чей промпт его просит.
        paid=bool(data["paid"]) if "paid" in data else None,
        fee=data.get("fee") or None,
        debt=data.get("debt") or None,
        nickname=data.get("nickname") or None,
    )
