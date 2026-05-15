"""BFCL (Berkeley Function-Calling Leaderboard) benchmark runner."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["BfclEvaluator", "run"]


class BfclEvaluator(BenchmarkEvaluator):
    name = "bfcl"

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or "EAGLE/eagle/data/bfcl/question.jsonl"
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question.get("prompt") or question["question"]

    def score(self, question: Dict[str, Any], response: str) -> Optional[bool]:
        # BFCL scoring uses AST-based function-call matching, which is
        # implemented in ``scripts/eval_bfcl_eagle.py``. We return None so the
        # per-row JSONL is still well-formed; final scoring is offline.
        return None


def run(args, method_cfg):
    return BfclEvaluator().run(args, method_cfg)
