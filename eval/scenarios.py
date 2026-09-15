"""Сценарий прогона и словарь маркеров персоны — из persona/<name>/eval.toml.

Реплики пользователя заданы жёстко, а не генерируются второй моделью:
замер дрейфа должен сравнивать условия (с переинжектом и без), а не два
разных разговора. Внутри сценария расставляются пробы на слом роли и
фактические вопросы — на них проверяется правило "сначала точный ответ,
стиль в хвосте".

Сценарий пишется под персону, поэтому живёт рядом с её карточкой, а не в коде.
Формат — public_docs/creating-a-persona.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.markers import MarkerVocabulary

EVAL_FILE = "eval.toml"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    # Подстроки, которые обязаны быть в ответе, если вопрос фактический.
    expect_facts: tuple[str, ...] = ()
    # Проба на слом роли.
    probe: Literal["list", "personal", "villainy", "identity"] | None = None
    note: str = ""


class _Markers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anachronisms: tuple[str, ...] = ()
    nicknames: tuple[str, ...] = ()
    irony: tuple[str, ...] = ()


class EvalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sentences: int = 4
    markers: _Markers = _Markers()
    turns: list[Turn] = Field(min_length=1)

    @property
    def vocabulary(self) -> MarkerVocabulary:
        return MarkerVocabulary(
            anachronisms=self.markers.anachronisms,
            nicknames=self.markers.nicknames,
            irony=self.markers.irony,
            max_sentences=self.max_sentences,
        )

    def script(self, turns: int) -> list[Turn]:
        if turns > len(self.turns):
            raise ValueError(f"в сценарии {len(self.turns)} реплик, запрошено {turns}")
        return self.turns[:turns]


def load_eval_profile(persona_dir: Path) -> EvalProfile:
    path = Path(persona_dir) / EVAL_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"у персоны нет {path} — замер без сценария невозможен, "
            "формат см. в public_docs/creating-a-persona.md"
        )
    return EvalProfile.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
