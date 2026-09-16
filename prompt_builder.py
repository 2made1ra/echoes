"""Разбор персон и сборка запроса к LLM из блоков.

Порядок блоков — главное правило проекта (см. public_docs/architecture.md):

    system:   общий блок безопасности (если персона его подключает)
    system:   карточка персонажа целиком
    system:   записи из long-term памяти (блока нет, если ретрив пуст)
    messages: последние N сообщений сессии
    messages: новое сообщение пользователя
    system:   блок переинжекта — после истории, каждые N реплик пользователя

Факты из памяти и черты характера живут в разных блоках намеренно: черты
переинжектятся по расписанию, факты подтягиваются по релевантности. Если их
смешать, при пустом ретриве вымывается и часть характера.

Персона — это директория persona/<name>/ с card.md, reinject.md, memory.md,
scenes.md, persona.toml и необязательным greetings.md. Текст персонажа живёт
только там; здесь его лишь разбирают.

Стартовые зарисовки (scenes.md) — голос рассказчика для интерфейса. В сборку
запроса они не попадают намеренно: повествование в контексте модель начала бы
копировать вместо реплик персонажа. Готовые первые реплики под зарисовки
(greetings.md) написаны заранее: в историю сессии идёт только реплика.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

Message = dict[str, str]

SHARED_DIR = "_shared"

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Строка, на которой фраза закончилась: перенос после неё — смысловой.
_PHRASE_END = re.compile(r"[.!?…:\]»\"]\s*$")
# Разделитель реплик внутри одной секции greetings.md.
_GREETING_SPLIT_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)


class PersonaUI(BaseModel):
    """Надписи клиента под персону."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    placeholder: str
    new_session: str
    close_session: str
    archived: str
    accent: str


class PersonaSettings(BaseModel):
    """persona.toml: параметры персоны, без текста характера."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str
    reinject_every: int | None = None
    safety: bool = True
    ui: PersonaUI


@dataclass(frozen=True)
class MemoryTemplates:
    """memory.md: как сессия сжимается в запись и как запись выглядит в промпте."""

    summarizer: str
    header: str
    record: str


@dataclass(frozen=True)
class Opening:
    """Начало сессии: зарисовка для интерфейса и первая реплика персонажа."""

    scene: str
    message: str


@dataclass(frozen=True)
class PersonaCard:
    """Карточка персонажа как есть плюс всё, что к ней прилагается."""

    name: str
    raw: str
    reinject: str
    sections: dict[str, str]
    settings: PersonaSettings
    memory: MemoryTemplates
    # Пары «зарисовка + первая реплика»; зарисовка в контекст модели не идёт.
    openings: tuple[Opening, ...]
    # Текст общего блока безопасности; None — персона его не подключает.
    safety: str | None = None

    @property
    def first_message(self) -> str:
        return _unwrap_soft_breaks(self.sections.get("first message", "").strip())

    @property
    def display_name(self) -> str:
        return self.settings.display_name

    def reinject_every(self, default: int) -> int:
        """N персоны, а если в persona.toml его нет — общий из окружения."""
        value = self.settings.reinject_every
        return default if value is None else value


def _unwrap_soft_breaks(text: str) -> str:
    """Склеить переносы, поставленные ради ширины markdown, а не ради смысла.

    Первая реплика уходит в историю и в интерфейс как есть: перенос посреди
    фразы рвёт строку в чате, а модель копирует такую вёрстку в ответах.
    Перенос сохраняется, только если строка закончила фразу или ремарку.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        previous = lines[-1] if lines else ""
        if previous.strip() and line.strip() and not _PHRASE_END.search(previous):
            lines[-1] = previous.rstrip() + " " + line.strip()
        else:
            lines.append(line)
    return "\n".join(lines)


def _split_sections(raw: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(raw))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[match.group(1).strip().lower()] = raw[match.end() : end].strip()
    return sections


def _read(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"у персоны нет файла {path}")
    return path.read_text(encoding="utf-8").strip()


def _load_memory(path: Path) -> MemoryTemplates:
    sections = _split_sections(_read(path))
    missing = [s for s in ("summarizer", "memory header", "record") if not sections.get(s)]
    if missing:
        raise ValueError(f"в {path} нет секций: {', '.join('## ' + s.title() for s in missing)}")
    return MemoryTemplates(
        summarizer=sections["summarizer"],
        header=sections["memory header"],
        record=sections["record"],
    )


def _load_scenes(path: Path) -> dict[str, str]:
    # Переносы внутри абзаца — только для удобства правки markdown: клиент
    # показывает текст с pre-wrap, и жёсткие переносы рвали бы строки.
    scenes = {
        title: re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        for title, text in _split_sections(_read(path)).items()
        if text
    }
    if not scenes:
        raise ValueError(f"в {path} нет ни одной зарисовки (секции '## ...')")
    return scenes


def _load_openings(directory: Path, first_message: str) -> tuple[Opening, ...]:
    """Зарисовки с их первыми репликами.

    greetings.md повторяет заголовки секций scenes.md, в секции — реплики под
    эту зарисовку через строку '---'. Файла нет — к каждой зарисовке идёт
    First Message из карточки.
    """
    scenes = _load_scenes(directory / "scenes.md")
    path = directory / "greetings.md"
    if not path.is_file():
        return tuple(Opening(scene, first_message) for scene in scenes.values())

    greetings = _split_sections(_read(path))
    unknown = sorted(set(greetings) - set(scenes))
    if unknown:
        raise ValueError(f"в {path} секции без зарисовки в scenes.md: {', '.join(unknown)}")
    openings: list[Opening] = []
    for title, scene in scenes.items():
        messages = [
            _unwrap_soft_breaks(part.strip())
            for part in _GREETING_SPLIT_RE.split(greetings.get(title, ""))
            if part.strip()
        ]
        if not messages:
            raise ValueError(f"в {path} нет реплик для зарисовки '## {title}'")
        openings.extend(Opening(scene, message) for message in messages)
    return tuple(openings)


def load_persona(directory: Path) -> PersonaCard:
    directory = Path(directory)
    raw = _read(directory / "card.md")
    sections = _split_sections(raw)
    if not sections.get("first message"):
        raise ValueError(f"в карточке {directory / 'card.md'} нет секции '## First Message'")

    settings = PersonaSettings.model_validate(
        tomllib.loads(_read(directory / "persona.toml"))
    )
    safety = _read(directory.parent / SHARED_DIR / "safety.md") if settings.safety else None

    first_message = _unwrap_soft_breaks(sections["first message"])
    return PersonaCard(
        name=directory.name,
        raw=raw,
        reinject=_read(directory / "reinject.md"),
        sections=sections,
        settings=settings,
        memory=_load_memory(directory / "memory.md"),
        openings=_load_openings(directory, first_message),
        safety=safety,
    )


def load_personas(persona_dir: Path) -> dict[str, PersonaCard]:
    """Все персоны из persona/: каждая поддиректория с card.md, кроме _shared."""
    personas = {
        path.name: load_persona(path)
        for path in sorted(Path(persona_dir).iterdir())
        if path.is_dir() and path.name != SHARED_DIR and (path / "card.md").is_file()
    }
    if not personas:
        raise ValueError(f"в {persona_dir} нет ни одной персоны")
    return personas


def default_persona_name(personas: dict[str, PersonaCard], configured: str | None) -> str:
    """DEFAULT_PERSONA, а если не задан — первая персона по алфавиту."""
    if not configured:
        return next(iter(personas))
    if configured not in personas:
        raise ValueError(f"DEFAULT_PERSONA={configured!r} не найдена среди {sorted(personas)}")
    return configured


def should_reinject(user_turn: int, every: int) -> bool:
    """user_turn — порядковый номер реплики пользователя в сессии, начиная с 1."""
    if every <= 0:
        return False
    return user_turn % every == 0


def render_memories(persona: PersonaCard, records: Sequence[str]) -> str:
    """Блок long-term памяти: сжатые записи, а не сырые логи."""
    body = "\n\n".join(record.strip() for record in records if record.strip())
    return persona.memory.header + "\n\n" + body


def build_messages(
    persona: PersonaCard,
    *,
    history: Sequence[Message] = (),
    user_message: str,
    memories: Sequence[str] = (),
    reinject: bool = False,
    window: int | None = None,
) -> list[Message]:
    messages: list[Message] = []

    if persona.safety:
        messages.append({"role": "system", "content": persona.safety})

    messages.append({"role": "system", "content": persona.raw})

    if memories:
        messages.append({"role": "system", "content": render_memories(persona, memories)})

    recent = list(history)
    if window is not None and window > 0:
        recent = recent[-window:]
    messages.extend({"role": m["role"], "content": m["content"]} for m in recent)

    messages.append({"role": "user", "content": user_message})

    if reinject:
        messages.append({"role": "system", "content": persona.reinject})

    return messages
