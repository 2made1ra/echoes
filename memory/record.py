"""Запись long-term памяти: сжатая сессия одной персоны.

В векторную БД кладутся только записи, никогда не сырые логи: сжатая запись
релевантнее лога и даёт естественный способ сослаться на прошлое своими
словами вместо ассистентского "как вы упоминали ранее".

Модель одна на всех персон. Как запись называется ("дело", "встреча"...) и как
выглядит в промпте, решает шаблон персоны (memory.md), а не этот код.
"""

from __future__ import annotations

import string
import uuid
from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, Field


class Record(BaseModel):
    """собеседник → что рассказал → что ответил → чем кончилось."""

    PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset(
        {"date", "counterpart", "story", "response", "outcome", "money"}
    )

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    persona: str

    counterpart: str
    story: str
    response: str
    outcome: str = ""

    # Оплата и долг — явные поля, а не вывод из текста: это источник реплик
    # в характере и дешёвый признак преемственности между сессиями.
    # paid=None — к персоне неприменимо (в её сценарии никто не платит).
    paid: bool | None = None
    fee: str | None = None
    debt: str | None = None

    nickname: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def check_template(cls, template: str) -> None:
        """Шаблон с неизвестной подстановкой ломается при старте, а не на ретриве."""
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
        unknown = fields - cls.PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"в шаблоне записи неизвестные подстановки {sorted(unknown)}, "
                f"доступны: {sorted(cls.PLACEHOLDERS)}"
            )

    def embedding_text(self) -> str:
        """Текст, по которому запись ищется. Без служебных полей."""
        parts = [self.counterpart, self.story, self.response, self.outcome]
        if self.nickname:
            parts.append(self.nickname)
        return ". ".join(p.strip() for p in parts if p and p.strip())

    def render(self, template: str) -> str:
        """Как запись выглядит в промпте — по шаблону персоны."""
        counterpart = self.counterpart
        if self.nickname:
            counterpart += f" ({self.nickname})"
        return template.format(
            date=self.created_at.strftime("%d.%m.%Y"),
            counterpart=counterpart,
            story=self.story,
            response=self.response,
            outcome=self.outcome,
            money=self._money(),
        ).strip()

    def _money(self) -> str:
        if self.paid is None:
            return ""
        money = "оплачено" if self.paid else "не оплачено"
        if self.fee:
            money += f" (сумма: {self.fee})"
        if self.debt:
            money += f", долг: {self.debt}"
        return money
