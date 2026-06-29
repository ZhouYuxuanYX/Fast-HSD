"""MATH benchmark runner.

Reproduces the scoring + prompting of ``verification/src/MATH/eval_math.py``:
- Math-assistant system prompt + chat template (``apply_chat_template`` on the
  target tokenizer with ``add_generation_prompt=True``).
- Gold answers come from the bundled EAGLE ``question.jsonl`` (``reference[0]``)
  via ``extract_boxed_answer``, or from a HF ``competition_math``-style row
  (``solution`` / ``answer``).
- Predictions extracted with the legacy multi-pattern ``extract_answer``;
  equivalence checked with the legacy normalization rules.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from fast_hsd.benchmarks._math_scoring import (
    extract_answer,
    extract_boxed_answer,
    is_equiv,
)
from fast_hsd.benchmarks.base import BenchmarkEvaluator

logger = logging.getLogger("fast_hsd.benchmarks.math")

__all__ = ["MathEvaluator", "run"]

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful math assistant. "
    "Solve the problem step by step and provide the final answer in \\boxed{} format."
)

DEFAULT_QUESTION_FILE = "EAGLE/eagle/data/competition_math/question.jsonl"


class MathEvaluator(BenchmarkEvaluator):
    name = "math"
    system_prompt = SYSTEM_PROMPT

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        """Read the bundled ``question.jsonl`` and normalize each row.

        The EAGLE format stores the prompt under ``turns`` (list of user
        strings) and the worked solution under ``reference`` (list of strings).
        The gold answer is the contents of the last ``\\boxed{}`` in
        ``reference[0]``. HF ``competition_math`` rows (``problem`` /
        ``solution`` / ``level`` / ``type``) also work.
        """
        path = getattr(args, "question_file", None) or DEFAULT_QUESTION_FILE
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield _normalize_row(row)

    def format_prompt(self, question: Dict[str, Any]) -> str:
        """Return the *user* content. ``BenchmarkEvaluator.generate`` wraps it
        with the system prompt + chat template."""
        return question["prompt_user"]

    def score(
        self, question: Dict[str, Any], response: str
    ) -> Optional[Tuple[Optional[bool], str]]:
        gold = question.get("gold")
        extracted = extract_answer(response)
        if gold is None:
            return (None, extracted)
        return (is_equiv(extracted, gold), extracted)

    def run(self, args, method_cfg) -> int:
        rc = super().run(args, method_cfg)
        # Inline the end-of-run analyzer report so users get the same view as
        # ``scripts/results_analysis_math.py`` without running a second tool.
        # Uses the shared ``_math_report`` module so the numbers here match
        # the per-row ``correct`` field written during the loop.
        try:
            from fast_hsd.benchmarks import _math_report  # local import: avoids cycles
            rows_path = os.path.join(args.output_dir, self.name, args.name, "rows.jsonl")
            question_file = getattr(args, "question_file", None) or DEFAULT_QUESTION_FILE
            refs = _math_report.load_references(question_file)
            stats = _math_report.analyze(rows_path, refs=refs)
            _math_report.print_stats(rows_path, stats, expected_n=len(refs))
        except Exception as e:
            logger.warning("end-of-run math report failed (%s: %s)", type(e).__name__, e)
        return rc


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce an EAGLE-format or HF-format row into the canonical shape."""
    # Prompt (user message).
    if "turns" in row:
        turns = row["turns"]
        prompt_user = turns[0] if isinstance(turns, list) and turns else str(turns)
    elif "problem" in row:
        prompt_user = row["problem"]
    elif "question" in row:
        prompt_user = row["question"]
    else:
        prompt_user = row.get("text", "")

    # Gold answer.
    gold = row.get("gold") or row.get("answer")
    if gold is None:
        if "reference" in row and isinstance(row["reference"], list) and row["reference"]:
            gold = extract_boxed_answer(row["reference"][0]) or None
        elif "solution" in row:
            gold = extract_boxed_answer(row["solution"]) or None

    # Use `in` rather than truthiness because question_id=0 is a real value.
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
        "prob_type": row.get("type") or row.get("category"),
    }


def run(args, method_cfg):
    return MathEvaluator().run(args, method_cfg)
