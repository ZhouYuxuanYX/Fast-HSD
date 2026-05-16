"""INCLUDE multilingual-understanding benchmark runner.

Ports the prompting and scoring of ``verification/src/INCLUDE/
run_include_qwen2.5_simple.py``. The bundled
``EAGLE/eagle/data/include/question.jsonl`` already contains the
zero-shot CoT prompt (with "Answer: Let's think step by step." at the
tail) and the gold letter under ``reference[0]``, so it's used directly.
Legacy adds K few-shot CoT examples before each question; that few-shot
ablation is left out for now since the dev examples aren't bundled.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, Optional, Tuple

from fast_hsd.benchmarks._include_scoring import score_include
from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["IncludeEvaluator", "run"]

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

DEFAULT_QUESTION_FILE = "EAGLE/eagle/data/include/question.jsonl"


class IncludeEvaluator(BenchmarkEvaluator):
    name = "include"
    system_prompt = SYSTEM_PROMPT

    def __init__(self):
        # Deterministic RNG for the extract-failure random-guess fallback.
        # Re-seeded inside ``run`` so it's tied to ``--seed``.
        self._rng = random.Random(0)

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or DEFAULT_QUESTION_FILE
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield _normalize_row(row)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question["prompt_user"]

    def score(
        self, question: Dict[str, Any], response: str
    ) -> Optional[Tuple[Optional[bool], str]]:
        gold = question.get("gold")
        if gold is None:
            return None
        correct, pred = score_include(response, gold, rng=self._rng)
        return (correct, pred)

    def run(self, args, method_cfg) -> int:
        self._rng = random.Random(args.seed)
        return super().run(args, method_cfg)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "turns" in row:
        turns = row["turns"]
        prompt_user = turns[0] if isinstance(turns, list) and turns else str(turns)
    else:
        prompt_user = row.get("prompt") or row.get("question") or ""

    gold = row.get("answer") or row.get("gold")
    if gold is None and "reference" in row:
        ref = row["reference"]
        if isinstance(ref, list) and ref:
            gold = str(ref[0]).strip().upper()
        else:
            gold = str(ref).strip().upper()

    if "question_id" in row:
        qid = row["question_id"]
    elif "id" in row:
        qid = row["id"]
    else:
        qid = None

    return {
        "question_id": qid,
        "prompt_user": prompt_user,
        "gold": gold,
        "level": row.get("level"),
        "prob_type": row.get("category") or row.get("subject"),
        "language": row.get("language"),
    }


def run(args, method_cfg):
    return IncludeEvaluator().run(args, method_cfg)
