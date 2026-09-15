"""Доменные ошибки сервиса.

Слой бизнес-логики не знает про HTTP: он бросает эти ошибки, а слой API
переводит их в коды. Так роуты остаются тонкими, а логику можно дёргать из
скриптов и замеров без FastAPI.
"""

from __future__ import annotations


class DialogueError(Exception):
    """Базовая ошибка диалога."""


class SessionNotFound(DialogueError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"сессия {session_id} не найдена или истекла — начните новую")


class UnknownPersona(DialogueError):
    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(f"персона {name!r} не найдена, доступны: {', '.join(available)}")
        self.name = name


class NothingToArchive(DialogueError):
    def __init__(self) -> None:
        super().__init__("в сессии нет разговора — нечего заносить в память")


class LongTermUnavailable(DialogueError):
    def __init__(self) -> None:
        super().__init__("long-term память недоступна: Qdrant не поднят")


class ProviderUnavailable(DialogueError):
    """Ошибка LLM-провайдера. Наружу отдаётся как есть, без маскировки."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"LLM-провайдер ({stage}): {cause}")
        self.cause = cause
