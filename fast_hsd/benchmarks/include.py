"""INCLUDE multilingual-understanding benchmark runner."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional

from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["IncludeEvaluator", "run"]

_LETTER_RE = re.compile(r"\b([A-D])\b")


class IncludeEvaluator(BenchmarkEvaluator):
    name = "include"

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or "EAGLE/eagle/data/include/question.jsonl"
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question.get("prompt") or question["question"]

    def score(self, question: Dict[str, Any], response: str) -> Optional[bool]:
        # INCLUDE is multiple-choice; we match on the first uppercase A-D
        # letter that appears in the response.
        gold = question.get("answer")
        if gold is None:
            return None
        m = _LETTER_RE.search(response)
        if m is None:
            return False
        return m.group(1) == str(gold).strip().upper()


def run(args, method_cfg):
    return IncludeEvaluator().run(args, method_cfg)
