"""MBPP+ benchmark runner. See base class for the driving loop."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["MbppPlusEvaluator", "run"]


class MbppPlusEvaluator(BenchmarkEvaluator):
    name = "mbppplus"

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or "EAGLE/eagle/data/mbppplus/question.jsonl"
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question.get("prompt") or question["question"]

    def score(self, question: Dict[str, Any], response: str) -> Optional[bool]:
        # MBPP+ scoring requires actually executing the generated code against
        # the test suite, which is handled out of process by
        # ``scripts/eval_mbppplus.py``. We return None here so the per-row
        # output is still well-formed; the user runs the offline scorer for
        # Pass@1.
        return None


def run(args, method_cfg):
    return MbppPlusEvaluator().run(args, method_cfg)
