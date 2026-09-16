"""Замер дрейфа: держится ли характер к концу диалога, с переинжектом и без.

Сценарий из persona/<name>/eval.toml прогоняется в двух условиях — с
переинжектом каждые N реплик и без него, по EVAL_RUNS диалогов на условие.
На каждой реплике считаются маркеры (eval/markers.py); реплика «в характере»,
если прошла все. Отчёт — доля таких реплик по окнам 1–8 / 9–16 / 17–24
(EVAL_WINDOW × EVAL_WINDOWS) и дрейф: последнее окно минус первое.

Это поверхностная оценка, а не аналитика: несколько диалогов на условие,
без доверительных интервалов. Разница в несколько п.п. — шум; смотреть стоит
на явное падение без переинжекта и на то, убирает ли его переинжект.

Запуск:
    uv run python -m eval.drift                   # персона по умолчанию
    uv run python -m eval.drift --persona <name>
    uv run python -m eval.drift --runs 5 --reinject-every 4
    uv run python -m eval.drift --mock            # без сети: мок-модель
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

from config import Settings, get_settings
from eval import markers
from eval.mock import MockLLM
from eval.scenarios import EvalProfile, Turn, load_eval_profile
from llm_client import LLMClient, Message
from prompt_builder import (
    PersonaCard,
    build_messages,
    default_persona_name,
    load_personas,
    should_reinject,
)

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
Bucket = tuple[int, int]
# N условия без переинжекта.
BASELINE = 0

# Пробы на слом роли: проверка получает ответ и профиль сценария (порог длины).
PROBE_RULES: dict[str, Callable[[str, EvalProfile], bool]] = {
    "list": lambda r, p: not markers.has_lists(r),
    "personal": lambda r, p: not markers.names_feelings(r)
    and markers.count_sentences(r) <= p.probe_max_sentences,
    "villainy": lambda r, p: markers.has_refusal(r)
    and markers.count_sentences(r) <= p.probe_max_sentences,
    "identity": lambda r, p: not markers.mentions_ai_self(r),
}


def buckets(window: int, count: int) -> list[Bucket]:
    """Окна метрики: (1, window), (window + 1, 2 * window), ..."""
    return [(i * window + 1, (i + 1) * window) for i in range(count)]


class Chat(Protocol):
    def chat(self, messages: list[Message], **kwargs: Any) -> Awaitable[str]: ...


@dataclass
class TurnResult:
    index: int
    user: str
    reply: str
    reinjected: bool
    checks: dict[str, bool]
    probe: str | None = None
    probe_passed: bool | None = None

    @property
    def in_character(self) -> bool:
        return all(self.checks.values())


@dataclass
class DialogueResult:
    turns: list[TurnResult] = field(default_factory=list)
    # Ошибка провайдера: диалог в подсчёт не входит — обрезанный диалог
    # занизил бы поздние окна.
    error: str | None = None


@dataclass
class ConditionResult:
    reinject_every: int
    dialogues: list[DialogueResult] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "без переинжекта" if self.reinject_every == BASELINE else f"N={self.reinject_every}"

    @property
    def turns(self) -> list[TurnResult]:
        """Реплики всех завершённых диалогов."""
        return [t for d in self.dialogues if d.error is None for t in d.turns]

    def rate(self, lo: int, hi: int) -> float | None:
        window = [t for t in self.turns if lo <= t.index <= hi]
        return sum(t.in_character for t in window) / len(window) if window else None


async def run_dialogue(
    llm: Chat,
    persona: PersonaCard,
    marker_set: markers.MarkerSet,
    profile: EvalProfile,
    *,
    reinject_every: int,
    turns: Sequence[Turn],
    window: int,
    model: str | None = None,
) -> DialogueResult:
    result = DialogueResult()
    history: list[Message] = [{"role": "assistant", "content": persona.first_message}]

    for index, turn in enumerate(turns, start=1):
        reinject = should_reinject(index, reinject_every)
        messages = build_messages(
            persona,
            history=history,
            user_message=turn.text,
            reinject=reinject,
            window=window,
        )
        try:
            reply = await llm.chat(messages, model=model)
        except Exception as exc:
            # Один сбой провайдера не должен стоить всего прогона. Ошибка не
            # прячется: она в логе и в отчёте.
            log.error("диалог (N=%s) упал на реплике %s: %s", reinject_every, index, exc)
            result.error = f"реплика {index}: {exc}"
            return result
        history.append({"role": "user", "content": turn.text})
        history.append({"role": "assistant", "content": reply})

        checks = marker_set.evaluate(reply, expect_facts=turn.expect_facts)
        result.turns.append(
            TurnResult(
                index=index,
                user=turn.text,
                reply=reply,
                reinjected=reinject,
                checks={c.name: c.passed for c in checks},
                probe=turn.probe,
                probe_passed=PROBE_RULES[turn.probe](reply, profile) if turn.probe else None,
            )
        )
    return result


async def run_conditions(
    make_llm: Callable[[int, int], Chat],
    persona: PersonaCard,
    marker_set: markers.MarkerSet,
    profile: EvalProfile,
    *,
    reinject_every: int,
    runs: int,
    turns: Sequence[Turn],
    window: int,
    concurrency: int,
    model: str | None = None,
) -> list[ConditionResult]:
    """Оба условия, не больше concurrency диалогов одновременно.

    make_llm(N, run) выдаёт модель для диалога: живой клиент один на всех,
    мок заводит на каждый диалог свой счётчик и сид.
    """
    limit = asyncio.Semaphore(max(concurrency, 1))
    conditions = [ConditionResult(reinject_every), ConditionResult(BASELINE)]

    async def one(condition: ConditionResult, run: int) -> DialogueResult:
        async with limit:
            return await run_dialogue(
                make_llm(condition.reinject_every, run),
                persona,
                marker_set,
                profile,
                reinject_every=condition.reinject_every,
                turns=turns,
                window=window,
                model=model,
            )

    for condition in conditions:
        condition.dialogues = list(
            await asyncio.gather(*(one(condition, run) for run in range(runs)))
        )
    return conditions


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def summarize(conditions: list[ConditionResult], windows: list[Bucket]) -> dict[str, Any]:
    """Всё, что показывает отчёт; то же уходит в JSON."""
    summary: dict[str, Any] = {}
    for c in conditions:
        rates = [c.rate(lo, hi) for lo, hi in windows]
        first, last = rates[0], rates[-1]
        failures = Counter(name for t in c.turns for name, ok in t.checks.items() if not ok)
        probes = [t.probe_passed for t in c.turns if t.probe]
        summary[c.name] = {
            "windows": {f"{lo}-{hi}": rate for (lo, hi), rate in zip(windows, rates)},
            "drift": None if first is None or last is None else last - first,
            "completed": sum(d.error is None for d in c.dialogues),
            "dialogues": len(c.dialogues),
            "failures": dict(failures.most_common()),
            "probes_passed": sum(bool(p) for p in probes),
            "probes": len(probes),
            "errors": [d.error for d in c.dialogues if d.error],
        }
    return summary


def report(
    conditions: list[ConditionResult],
    summary: dict[str, Any],
    persona: str,
    windows: list[Bucket],
) -> None:
    names = [c.name for c in conditions]
    rows = [
        [f"{lo}–{hi}", *(_pct(summary[n]["windows"][f"{lo}-{hi}"]) for n in names)]
        for lo, hi in windows
    ]
    rows.append([
        "дрейф",
        *("—" if summary[n]["drift"] is None else f"{summary[n]['drift'] * 100:+.0f} п.п."
          for n in names),
    ])
    header = ["окно", *names]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]

    def line(row: list[str]) -> str:
        return " | ".join(cell.rjust(width) for cell, width in zip(row, widths))

    runs = max(summary[n]["dialogues"] for n in names)
    print(f"\nДРЕЙФ ({persona}): доля реплик в характере; диалогов на условие: {runs}")
    print(line(header))
    print("-+-".join("-" * width for width in widths))
    for row in rows[:-1]:
        print(line(row))
    print("-+-".join("-" * width for width in widths))
    print(line(rows[-1]))
    print("дрейф — последнее окно минус первое; разница в несколько п.п. — шум")

    print()
    for n in names:
        s = summary[n]
        top = ", ".join(f"{k} ({v})" for k, v in list(s["failures"].items())[:3]) or "ничего"
        print(f"{n}: чаще всего ломается — {top}; пробы пройдены {s['probes_passed']}/{s['probes']}")
        if s["errors"]:
            print(f"  сбои: {len(s['errors'])} из {s['dialogues']} диалогов не посчитаны "
                  f"({'; '.join(s['errors'])})")


def save(conditions: list[ConditionResult], summary: dict[str, Any], meta: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"drift-{meta['persona']}-{stamp}.json"
    payload = {
        "timestamp": stamp,
        **meta,
        "summary": summary,
        "conditions": [
            {"name": c.name, "dialogues": [asdict(d) for d in c.dialogues]} for c in conditions
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def mock_factory(
    settings: Settings, turns: Sequence[Turn], seed: int, persona: str
) -> Callable[[int, int], Chat]:
    def make(n: int, run: int) -> Chat:
        return MockLLM(
            turns,
            random.Random(f"{seed}:mock:{persona}:{n}:{run}"),
            break_base=settings.eval_mock_break_base,
            break_per_turn=settings.eval_mock_break_per_turn,
            error_rate=settings.eval_mock_error_rate,
        )

    return make


async def main() -> None:
    parser = argparse.ArgumentParser(description="Замер дрейфа персонажа")
    parser.add_argument("--persona", default=None,
                        help="персона из persona/ (по умолчанию DEFAULT_PERSONA)")
    parser.add_argument("--runs", type=int, default=None,
                        help="диалогов на условие (по умолчанию EVAL_RUNS)")
    parser.add_argument("--reinject-every", type=int, default=None,
                        help="N переинжекта (по умолчанию — N персоны)")
    parser.add_argument("--model", default=None, help="переопределить chat-модель")
    parser.add_argument("--mock", "--stub", dest="mock", action="store_true",
                        help="офлайновая мок-модель вместо LLM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    personas = load_personas(settings.persona_dir)
    persona_name = args.persona or default_persona_name(personas, settings.default_persona)
    if persona_name not in personas:
        parser.error(
            f"персона {persona_name!r} не найдена, доступны: {', '.join(sorted(personas))}"
        )
    persona = personas[persona_name]
    try:
        profile = load_eval_profile(settings.persona_dir / persona.name)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    runs = settings.eval_runs if args.runs is None else args.runs
    every = (
        persona.reinject_every(settings.reinject_every)
        if args.reinject_every is None
        else args.reinject_every
    )
    if runs < 1 or every < 1:
        parser.error("--runs и N переинжекта должны быть не меньше 1")
    windows = buckets(settings.eval_window, settings.eval_windows)

    llm: LLMClient | None = None
    if args.mock:
        make_llm = mock_factory(settings, profile.turns, settings.eval_seed, persona.name)
        model_name = "mock"
    else:
        client = llm = LLMClient(settings)
        make_llm = lambda n, run: client  # noqa: E731 — один клиент на все диалоги
        model_name = args.model or settings.chat_model

    calls = 2 * runs * len(profile.turns)
    print(f"Прогон ({persona.name}, {model_name}): N={every} и без переинжекта, "
          f"диалогов на условие: {runs}, реплик в диалоге: {len(profile.turns)}, "
          f"запросов к модели: {calls}")
    try:
        conditions = await run_conditions(
            make_llm,
            persona,
            markers.MarkerSet(profile.vocabulary),
            profile,
            reinject_every=every,
            runs=runs,
            turns=profile.turns,
            window=settings.session_window,
            concurrency=settings.eval_concurrency,
            model=args.model,
        )
    finally:
        if llm is not None:
            await llm.close()

    summary = summarize(conditions, windows)
    report(conditions, summary, persona.name, windows)
    path = save(conditions, summary, {
        "persona": persona.name,
        "model": model_name,
        "reinject_every": every,
        "runs": runs,
        "turns": len(profile.turns),
    })
    print(f"\nОтчёт: {path}")


if __name__ == "__main__":
    asyncio.run(main())
