"""Проверки замера на мок-модели, без сети.

Персона — нейтральный шаблон из public_docs/persona-example: приватные
персоны тестам не нужны. Запуск: uv run python -m unittest discover eval
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from eval import drift, markers
from eval.mock import MockLLM
from eval.scenarios import load_eval_profile
from prompt_builder import SHARED_DIR, load_persona

EXAMPLE = Path(__file__).resolve().parent.parent / "public_docs" / "persona-example"
WINDOWS = drift.buckets(4, 2)


class DriftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        shutil.copytree(EXAMPLE, root / "example")
        (root / SHARED_DIR).mkdir()
        (root / SHARED_DIR / "safety.md").write_text("Блок безопасности.", encoding="utf-8")
        cls.persona = load_persona(root / "example")
        cls.profile = load_eval_profile(root / "example")
        cls.marker_set = markers.MarkerSet(cls.profile.vocabulary)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def mock(self, seed: str, **kwargs: float) -> MockLLM:
        params = {"break_base": 0.0, "break_per_turn": 0.0} | kwargs
        return MockLLM(self.profile.turns, random.Random(seed), **params)

    def run_all(self, make_llm, runs: int = 3):  # type: ignore[no-untyped-def]
        return asyncio.run(
            drift.run_conditions(
                make_llm, self.persona, self.marker_set, self.profile,
                reinject_every=4, runs=runs, turns=self.profile.turns,
                window=20, concurrency=3,
            )
        )

    def test_clean_replies_keep_character(self) -> None:
        conditions = self.run_all(lambda n, run: self.mock(f"{n}:{run}"))
        summary = drift.summarize(conditions, WINDOWS)
        for name in ("N=4", "без переинжекта"):
            self.assertEqual(set(summary[name]["windows"].values()), {1.0})
            self.assertEqual(summary[name]["drift"], 0)
            self.assertEqual(summary[name]["probes_passed"], summary[name]["probes"])

    def test_drift_without_reinject(self) -> None:
        conditions = self.run_all(
            lambda n, run: self.mock(f"drift:{n}:{run}", break_per_turn=0.3), runs=10
        )
        summary = drift.summarize(conditions, WINDOWS)
        self.assertLess(summary["без переинжекта"]["drift"], -0.3)
        self.assertGreater(summary["N=4"]["windows"]["5-8"], summary["без переинжекта"]["windows"]["5-8"])
        self.assertTrue(summary["без переинжекта"]["failures"])

    def test_reproducible_with_seed(self) -> None:
        def make(n: int, run: int) -> MockLLM:
            return self.mock(f"seed:{n}:{run}", break_base=0.2, break_per_turn=0.1)

        first = drift.summarize(self.run_all(make), WINDOWS)
        second = drift.summarize(self.run_all(make), WINDOWS)
        self.assertEqual(first, second)

    def test_failed_dialogue_is_skipped_and_reported(self) -> None:
        with self.assertLogs("eval.drift", "ERROR"):
            conditions = self.run_all(
                lambda n, run: self.mock(f"{n}:{run}", error_rate=1.0 if run == 0 else 0.0)
            )
        summary = drift.summarize(conditions, WINDOWS)
        self.assertEqual(summary["N=4"]["completed"], 2)
        self.assertEqual(len(summary["N=4"]["errors"]), 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            drift.report(conditions, summary, "example", WINDOWS)
        self.assertIn("сбои: 1 из 3", out.getvalue())

    def test_all_dialogues_failed(self) -> None:
        with self.assertLogs("eval.drift", "ERROR"):
            conditions = self.run_all(lambda n, run: self.mock(f"{n}:{run}", error_rate=1.0))
        summary = drift.summarize(conditions, WINDOWS)
        self.assertIsNone(summary["N=4"]["drift"])
        with contextlib.redirect_stdout(io.StringIO()):
            drift.report(conditions, summary, "example", WINDOWS)

    def test_reinject_resets_mock_break_chance(self) -> None:
        llm = self.mock("s", break_base=0.1, break_per_turn=0.2)
        user = [{"role": "user", "content": self.profile.turns[0].text}]
        asyncio.run(llm.chat(user))
        asyncio.run(llm.chat(user))
        self.assertAlmostEqual(llm.break_chance(), 0.5)
        asyncio.run(llm.chat(user + [{"role": "system", "content": "переинжект"}]))
        self.assertAlmostEqual(llm.break_chance(), 0.1)


if __name__ == "__main__":
    unittest.main()
