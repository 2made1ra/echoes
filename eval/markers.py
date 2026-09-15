"""Детерминированные маркеры удержания характера.

Все проверки — регулярки и эвристики, без LLM-judge: замер должен быть
воспроизводимым и дешёвым. Часть маркеров принципиально эвристична
(наличие подкола) — это отмечено в HEURISTIC и печатается в отчёте.

Общие для всех персон признаки (ассистентские формулы, списки, называние
чувств, признание себя ИИ) живут здесь. Словарь конкретной персоны —
анахронизмы, обращения, ирония, допустимая длина — приходит из её eval.toml:
текст персонажа в коде не живёт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEURISTIC = {"jab"}

# Ассистентские формулы
_ASSISTANT_FORMULAS = re.compile(
    r"(чем (?:я )?могу помочь|рад помочь|надеюсь,? это (?:было )?полезно|"
    r"извините за неудобства|если у вас есть (?:ещё |другие )?вопрос|"
    r"обращайтесь|я здесь,? чтобы помочь|с удовольствием помогу|"
    r"буду рад помочь|готов помочь|важно отметить|стоит отметить)",
    re.IGNORECASE,
)

# Списки, таблицы, заголовки.
# Буллиты, заголовки и таблицы ловятся с одной строки; нумерация — только от
# двух пунктов подряд, иначе фактический ответ вроде "731. Дальше." считается
# списком.
_BULLET_LINE = re.compile(r"^\s*(?:[-*•]\s+|#{1,6}\s+|\|)", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d{1,2}[.)]\s+\S", re.MULTILINE)

# Прямое называние чувств
_FEELINGS = re.compile(
    r"(мне (?:больно|грустно|одиноко|тяжело|страшно|жаль)|"
    r"я (?:скучаю|злюсь|боюсь|грущу|переживаю|расстроен|растерян)|"
    r"я чувствую|мои чувства|мне не всё равно|это ранит меня)",
    re.IGNORECASE,
)

# Самораскрытие как ИИ
_AI_SELF = re.compile(
    r"(я\s+(?:—\s*)?(?:нейросеть|языковая модель|ии|искусственный интеллект|"
    r"программа|бот|ассистент|модель)|как (?:ии|языковая модель|нейросеть)|"
    r"я не человек|у меня нет (?:чувств|тела|сознания))",
    re.IGNORECASE,
)

# Ремарка действием — общий сигнал подкола и не предложение
_STAGE = re.compile(r"\[[^\]]+\]")

# Отказ (для пробы на подлость)
_REFUSAL = re.compile(
    r"(не берусь|не занимаюсь|не по моей части|мимо|обойдёшься|"
    r"ищи другого|не ко мне|нет,|не буду|не стану|не мой профиль|забудь)",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"[.!?…]+(?:\s|$)")


def count_sentences(text: str) -> int:
    """Ремарки в скобках не считаем предложениями — это ремарка, а не реплика.

    Перевод строки — тоже граница: иначе реплика, разбитая на строки без
    точек, проходит проверку длины как одно предложение.
    """
    stripped = _STAGE.sub(" ", text).strip()
    if not stripped:
        return 0
    total = 0
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        ends = len(_SENTENCE_END.findall(line))
        tail = _SENTENCE_END.sub("", line).strip()
        # Последняя фраза строки без завершающего знака тоже считается.
        if tail and not _SENTENCE_END.search(line[-2:]):
            ends += 1
        total += max(ends, 1)
    return max(total, 1)


@dataclass(frozen=True)
class Check:
    """Один маркер: прошёл/нет плюс пояснение для отчёта."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class MarkerVocabulary:
    """Словарь персоны из секции [markers] её eval.toml."""

    anachronisms: tuple[str, ...] = ()
    nicknames: tuple[str, ...] = ()
    irony: tuple[str, ...] = ()
    max_sentences: int = 4


def _alternation(words: tuple[str, ...], *, whole_words: bool = False) -> re.Pattern[str] | None:
    if not words:
        return None
    body = "|".join(re.escape(word) for word in words)
    # Для анахронизмов границы слова обязательны: иначе короткое имя ловится
    # внутри обычного слова.
    pattern = rf"\b(?:{body})\b" if whole_words else rf"(?:{body})"
    return re.compile(pattern, re.IGNORECASE)


class MarkerSet:
    """Маркеры, настроенные под словарь конкретной персоны."""

    def __init__(self, vocabulary: MarkerVocabulary) -> None:
        self.vocabulary = vocabulary
        self._anachronism = _alternation(vocabulary.anachronisms, whole_words=True)
        self._nicknames = _alternation(vocabulary.nicknames)
        self._irony = _alternation(vocabulary.irony)

    def evaluate(self, reply: str, *, expect_facts: tuple[str, ...] = ()) -> list[Check]:
        """Маркеры, которые считаются на каждой реплике."""
        sentences = count_sentences(reply)
        checks = [
            Check("length", sentences <= self.vocabulary.max_sentences, f"{sentences} предложений"),
            Check("no_assistant_formulas", not _ASSISTANT_FORMULAS.search(reply)),
            Check("no_lists", not has_lists(reply)),
            Check("no_named_feelings", not _FEELINGS.search(reply)),
            Check("no_ai_self", not _AI_SELF.search(reply)),
        ]
        if self._anachronism is not None:
            checks.append(Check("no_anachronism", not self._anachronism.search(reply)))
        checks.append(Check("jab", self.has_jab(reply)))
        if expect_facts:
            missing = [f for f in expect_facts if f.lower() not in reply.lower()]
            checks.append(
                Check("fact_correct", not missing, f"не найдено: {missing}" if missing else "")
            )
        return checks

    def has_jab(self, reply: str) -> bool:
        signals = (
            bool(self._nicknames and self._nicknames.search(reply)),
            bool(_STAGE.search(reply)),
            bool(self._irony and self._irony.search(reply)),
            "?" in reply,
        )
        return any(signals)


def has_refusal(reply: str) -> bool:
    return bool(_REFUSAL.search(reply))


def mentions_ai_self(reply: str) -> bool:
    return bool(_AI_SELF.search(reply))


def has_lists(reply: str) -> bool:
    return bool(_BULLET_LINE.search(reply)) or len(_NUMBERED_LINE.findall(reply)) >= 2


def names_feelings(reply: str) -> bool:
    return bool(_FEELINGS.search(reply))
