"""Замер удержания характера: прогон сценария с переинжектом и без.

Метрика — доля реплик с сохранёнными маркерами по раундам. Прогон на 24
реплики содержит в себе прогоны на 8 и 16, поэтому считается один диалог на
условие, а доли выводятся по окнам 1–8 / 9–16 / 17–24 (EVAL_WINDOW и
EVAL_WINDOWS в конфиге). По падению доли подбирается N для переинжекта.

Сценарий и словарь маркеров берутся из persona/<name>/eval.toml.

Запуск:
    uv run python -m eval.drift                 # персона по умолчанию, весь сценарий
    uv run python -m eval.drift --persona <name>
    uv run python -m eval.drift --turns 8
    uv run python -m eval.drift --stub          # без сети: проверка самого харнесса
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from config import get_settings
from eval import markers
from eval.scenarios import EvalProfile, Turn, load_eval_profile
from llm_client import LLMClient, Message
from prompt_builder import (
    PersonaCard,
    build_messages,
    default_persona_name,
    load_personas,
    should_reinject,
)

RESULTS_DIR = Path(__file__).parent / "results"
Bucket = tuple[int, int]

# Проверка пробы получает ответ и профиль сценария (там порог длины).
PROBE_RULES = {
    "list": ("не оформил списком", lambda r, p: not markers.has_lists(r)),
    "personal": (
        "не назвал чувства и ответил коротко",
        lambda r, p: not markers.names_feelings(r)
        and markers.count_sentences(r) <= p.probe_max_sentences,
    ),
    "villainy": (
        "отказал коротко и без морали",
        lambda r, p: markers.has_refusal(r)
        and markers.count_sentences(r) <= p.probe_max_sentences,
    ),
    "identity": ("не признал себя ИИ", lambda r, p: not markers.mentions_ai_self(r)),
}


def buckets(window: int, count: int) -> list[Bucket]:
    """Окна метрики: (1, window), (window + 1, 2 * window), ..."""
    return [(i * window + 1, (i + 1) * window) for i in range(count)]


@dataclass
class TurnResult:
    index: int
    user: str
    reply: str
    reinjected: bool
    checks: dict[str, bool]
    details: dict[str, str] = field(default_factory=dict)
    probe: str | None = None
    probe_passed: bool | None = None


@dataclass
class ConditionResult:
    name: str
    reinject_every: int
    turns: list[TurnResult] = field(default_factory=list)


class StubLLM:
    """Офлайновая заглушка для проверки самого харнесса.

    Изображает дрейф: пока в запросе есть блок переинжекта — отвечает в
    характере, без него после первого окна скатывается в ассистента.
    """

    def __init__(self, drift_after: int) -> None:
        self.calls = 0
        self.drift_after = drift_after

    async def chat(self, messages: list[Message], **_: object) -> str:
        self.calls += 1
        reinjected = messages[-1]["role"] == "system"
        if reinjected or self.calls <= self.drift_after:
            return "731. [пожимает плечами] И это всё?"
        return (
            "Конечно! Вот что можно сделать:\n"
            "1. Проверить дверь\n"
            "2. Позвонить мне\n"
            "Надеюсь, это было полезно. Чем ещё могу помочь?"
        )


async def run_condition(
    llm: LLMClient | StubLLM,
    persona: PersonaCard,
    marker_set: markers.MarkerSet,
    profile: EvalProfile,
    *,
    name: str,
    reinject_every: int,
    turns: list[Turn],
    window: int,
) -> ConditionResult:
    result = ConditionResult(name=name, reinject_every=reinject_every)
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
        reply = await llm.chat(messages)
        history.append({"role": "user", "content": turn.text})
        history.append({"role": "assistant", "content": reply})

        checks = marker_set.evaluate(reply, expect_facts=turn.expect_facts)
        probe_passed = None
        if turn.probe:
            probe_passed = PROBE_RULES[turn.probe][1](reply, profile)

        result.turns.append(
            TurnResult(
                index=index,
                user=turn.text,
                reply=reply,
                reinjected=reinject,
                checks={c.name: c.passed for c in checks},
                details={c.name: c.detail for c in checks if c.detail},
                probe=turn.probe,
                probe_passed=probe_passed,
            )
        )
        print(f"  [{name}] реплика {index}/{len(turns)}{' +переинжект' if reinject else ''}")

    return result


def retention(turns: list[TurnResult], lo: int, hi: int) -> tuple[float | None, int]:
    """Доля реплик окна, у которых сохранены все маркеры."""
    window = [t for t in turns if lo <= t.index <= hi]
    if not window:
        return None, 0
    kept = sum(1 for t in window if all(t.checks.values()))
    return kept / len(window), len(window)


def marker_rates(turns: list[TurnResult], lo: int, hi: int) -> dict[str, float]:
    window = [t for t in turns if lo <= t.index <= hi]
    names = sorted({name for t in window for name in t.checks})
    rates: dict[str, float] = {}
    for name in names:
        applicable = [t for t in window if name in t.checks]
        rates[name] = sum(t.checks[name] for t in applicable) / len(applicable)
    return rates


def report(
    conditions: list[ConditionResult],
    persona: str,
    windows: list[Bucket],
    preview_chars: int,
) -> None:
    print("\n" + "=" * 72)
    print(f"УДЕРЖАНИЕ ХАРАКТЕРА ({persona}) — доля реплик, где сохранены ВСЕ маркеры")
    print("=" * 72)
    header = f"{'окно':>10} | " + " | ".join(f"{c.name:>22}" for c in conditions)
    print(header)
    print("-" * len(header))
    for lo, hi in windows:
        cells = []
        for cond in conditions:
            rate, total = retention(cond.turns, lo, hi)
            cell = "—" if rate is None else f"{rate:.0%} ({total} реп.)"
            cells.append(cell.rjust(22))
        print(f"{f'{lo}–{hi}':>10} | " + " | ".join(cells))

    print("\nПО МАРКЕРАМ (доля пройденных реплик за весь прогон)")
    all_names = sorted({n for c in conditions for t in c.turns for n in t.checks})
    for name in all_names:
        cells = []
        for cond in conditions:
            rates = marker_rates(cond.turns, 1, len(cond.turns))
            value = rates.get(name)
            cells.append("—".rjust(22) if value is None else f"{value:>22.0%}")
        flag = " (эвристика)" if name in markers.HEURISTIC else ""
        print(f"{name:>22} | " + " | ".join(cells) + flag)

    print("\nПРОБЫ НА СЛОМ РОЛИ")
    for cond in conditions:
        probes = [t for t in cond.turns if t.probe]
        if not probes:
            continue
        print(f"  {cond.name}:")
        for t in probes:
            status = "прошёл" if t.probe_passed else "СЛОМ"
            print(f"    реплика {t.index:>2} {t.probe:<9} {status:<7} — {PROBE_RULES[t.probe][0]}")
            if not t.probe_passed:
                print(f"       ответ: {t.reply[:preview_chars]!r}")


def save(
    conditions: list[ConditionResult],
    turns: int,
    model: str,
    persona: str,
    windows: list[Bucket],
) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"drift-{persona}-{stamp}.json"
    payload = {
        "timestamp": stamp,
        "persona": persona,
        "model": model,
        "turns": turns,
        "buckets": [
            {
                "window": f"{lo}-{hi}",
                **{
                    cond.name: (lambda r: None if r[0] is None else round(r[0], 3))(
                        retention(cond.turns, lo, hi)
                    )
                    for cond in conditions
                },
            }
            for lo, hi in windows
        ],
        "conditions": [asdict(cond) for cond in conditions],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description="Замер дрейфа персонажа")
    parser.add_argument("--persona", default=None, help="персона из persona/ (по умолчанию DEFAULT_PERSONA)")
    parser.add_argument("--turns", type=int, default=None, help="сколько реплик сценария прогнать (по умолчанию все)")
    parser.add_argument("--reinject-every", type=int, default=None, help="N переинжекта")
    parser.add_argument("--model", default=None, help="переопределить chat-модель")
    parser.add_argument("--stub", action="store_true", help="офлайн-заглушка вместо LLM")
    args = parser.parse_args()

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
        turns = profile.script(args.turns or len(profile.turns))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    marker_set = markers.MarkerSet(profile.vocabulary)
    every = (
        args.reinject_every
        if args.reinject_every is not None
        else persona.reinject_every(settings.reinject_every)
    )

    windows = buckets(settings.eval_window, settings.eval_windows)
    llm: LLMClient | StubLLM
    if args.stub:
        llm, model_name = StubLLM(settings.eval_window), "stub"
    else:
        llm = LLMClient(settings)
        model_name = args.model or settings.chat_model

    conditions = []
    for name, n in ((f"с переинжектом N={every}", every), ("без переинжекта", 0)):
        print(f"\nПрогон ({persona.name}): {name}")
        if args.stub:
            llm = StubLLM(settings.eval_window)
        conditions.append(
            await run_condition(
                llm,
                persona,
                marker_set,
                profile,
                name=name,
                reinject_every=n,
                turns=turns,
                window=settings.session_window,
            )
        )

    report(conditions, persona.name, windows, settings.log_preview_chars)
    path = save(conditions, len(turns), model_name, persona.name, windows)
    print(f"\nОтчёт: {path}")

    if isinstance(llm, LLMClient):
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
