"""MATH benchmark runner.

Thin wrapper around the existing logic in ``verification/src/MATH/eval_math.py``.
The unified ``BenchmarkEvaluator`` handles model loading, generation, and
output writing; this module only provides the dataset loader, prompt formatter,
and scorer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["MathEvaluator", "run"]


class MathEvaluator(BenchmarkEvaluator):
    name = "math"

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        # Defaults to the bundled question.jsonl from EAGLE/eagle/data/competition_math/.
        path = getattr(args, "question_file", None) or "EAGLE/eagle/data/competition_math/question.jsonl"
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question.get("problem") or question.get("question") or question["text"]

    def score(self, question: Dict[str, Any], response: str) -> Optional[bool]:
        # Defer to the existing answer-extraction logic to keep parity with
        # the paper. If the helper isn't present, return None — the user can
        # rescore offline via scripts/results_analysis.py.
        try:
            from verification.src.MATH.eval_math import extract_answer, is_equiv  # type: ignore
        except Exception:
            return None
        gold = question.get("answer") or question.get("gold")
        if gold is None:
            return None
        pred = extract_answer(response)
        return bool(is_equiv(pred, gold))


def run(args, method_cfg):
    return MathEvaluator().run(args, method_cfg)
