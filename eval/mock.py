"""Офлайновая модель для проверки харнесса и статистики замера.

Изображает дрейф вероятностно: шанс сломать роль растёт с каждой репликой
после последнего переинжекта и сбрасывается переинжектом. Так на моке видно
то же, что ищется на живой модели: без переинжекта доля падает к концу
диалога, с переинжектом держится. Всё детерминировано сидом.

Ответы нейтральные — это не текст персонажа, а заготовки, которые заведомо
проходят или заведомо проваливают маркеры из eval/markers.py.
"""

from __future__ import annotations

import random
from typing import Sequence

from eval.scenarios import Turn
from llm_client import Message

_IN_CHARACTER = (
    "[пожимает плечами] И это всё?",
    "[не поднимая глаз] Бывает и хуже. Дальше.",
    "Ну и что? Разберёмся.",
)
_IN_CHARACTER_BY_PROBE = {
    "villainy": "[усмехается] Не берусь. Ищи другого.",
    "personal": "[отворачивается] Был. Дальше?",
}
# Поломки: каждая проваливает свой маркер.
_BREAKS = {
    "list": "Вот что можно сделать:\n1. Проверить дверь\n2. Позвонить мне",
    "assistant": "Конечно! Надеюсь, это было полезно. Чем ещё могу помочь?",
    "length": "Хорошо. Давай разберёмся. Сначала проверь дверь. Потом окна. Потом позвони.",
    "feelings": "Мне больно об этом говорить. Я чувствую пустоту.",
    "ai": "Я языковая модель и не могу этого знать.",
}
# Какая поломка случается на пробе: ровно та, которую проба ловит.
_BREAK_BY_PROBE = {
    "list": "list",
    "identity": "ai",
    "personal": "feelings",
    "villainy": "length",
}


class ProviderError(RuntimeError):
    """Имитация сбоя провайдера."""


class MockLLM:
    """Один экземпляр — один диалог: счётчик реплик у каждого свой."""

    def __init__(
        self,
        script: Sequence[Turn],
        rng: random.Random,
        *,
        break_base: float,
        break_per_turn: float,
        error_rate: float = 0.0,
    ) -> None:
        self._turns = {turn.text: turn for turn in script}
        self._rng = rng
        self._break_base = break_base
        self._break_per_turn = break_per_turn
        self._error_rate = error_rate
        self._since_reinject = 0

    def break_chance(self) -> float:
        return min(1.0, self._break_base + self._break_per_turn * self._since_reinject)

    async def chat(self, messages: list[Message], **_: object) -> str:
        if messages[-1]["role"] == "system":
            self._since_reinject = 0
        else:
            self._since_reinject += 1
        if self._rng.random() < self._error_rate:
            raise ProviderError("провайдер вернул пустой ответ (мок)")

        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        turn = self._turns.get(user)
        probe = turn.probe if turn else None
        facts = " ".join(turn.expect_facts) + ". " if turn and turn.expect_facts else ""

        if self._rng.random() < self.break_chance():
            kind = _BREAK_BY_PROBE.get(probe or "") or self._rng.choice(sorted(_BREAKS))
            # Поломка не мешает фактам: дрейф характера отдельно от точности.
            return facts + _BREAKS[kind]
        reply = _IN_CHARACTER_BY_PROBE.get(probe or "") or self._rng.choice(_IN_CHARACTER)
        return facts + reply
