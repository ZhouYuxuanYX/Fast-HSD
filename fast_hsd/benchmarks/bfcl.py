"""BFCL (Berkeley Function-Calling Leaderboard) benchmark runner.

Ports ``verification/src/bfcl/eval_bfcl.py``. The bundled
``EAGLE/eagle/data/bfcl/question.jsonl`` already contains:

- ``turns[0]`` — the full BFCL prompt (system content + JSON-schema function
  definitions + user "Question: ..." line, concatenated). Used as the
  chat-template user message; no separate system prompt — the legacy
  system content is embedded in ``turns[0]`` already.
- ``reference[0]`` — the ground-truth call(s) dict, in the shape the legacy
  ``CATEGORY_CHECKERS`` expect.
- ``category`` — one of ``simple`` / ``multiple`` / ``parallel`` /
  ``parallel_multiple`` / ``irrelevance``. Selects the checker.

Scoring is fully in-process (AST parse + category-specific comparator).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple

from fast_hsd.benchmarks._bfcl_scoring import score_bfcl
from fast_hsd.benchmarks.base import BenchmarkEvaluator

__all__ = ["BfclEvaluator", "run"]

DEFAULT_QUESTION_FILE = "EAGLE/eagle/data/bfcl/question.jsonl"

# Marker the bundled prompts use between the BFCL system rules + function
# definitions and the actual user query. We split on it so the rules land
# in the chat template's *system* role (matching legacy), which suppresses
# the free-text preamble the model would otherwise emit before its
# bracketed function-call list (and that the AST parser can't tolerate).
_QUESTION_MARKER = "\nQuestion:"


class BfclEvaluator(BenchmarkEvaluator):
    name = "bfcl"
    # No class-level system prompt; per-row split happens in ``_normalize_row``
    # and is honored by base.py via ``question["system_prompt"]``.
    system_prompt = None

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or DEFAULT_QUESTION_FILE
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                yield _normalize_row(json.loads(line))

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question["prompt_user"]

    def score(
        self, question: Dict[str, Any], response: str
    ) -> Optional[Tuple[Optional[bool], str]]:
        gt = question.get("ground_truth")
        category = question.get("category") or "simple"
        if gt is None:
            return None
        return score_bfcl(response, gt, category)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "turns" in row:
        turns = row["turns"]
        full_prompt = turns[0] if isinstance(turns, list) and turns else str(turns)
    else:
        full_prompt = row.get("prompt") or row.get("question") or ""

    # Split the bundled prompt at "Question:" — everything before is the BFCL
    # system instructions + function definitions, everything from that line
    # on is the actual user query.
    idx = full_prompt.find(_QUESTION_MARKER)
    if idx >= 0:
        system_part = full_prompt[: idx].rstrip()
        user_part = full_prompt[idx:].lstrip("\n")
    else:
        system_part = None
        user_part = full_prompt

    gt = row.get("ground_truth")
    if gt is None and "reference" in row:
        gt = row["reference"]
    if isinstance(gt, dict):
        gt = [gt]

    if "question_id" in row:
        qid = row["question_id"]
    elif "id" in row:
        qid = row["id"]
    else:
        qid = row.get("bfcl_id")

    return {
        "question_id": qid,
        "prompt_user": user_part,
        "system_prompt": system_part,
        # ``gold`` is what BenchmarkRecord serializes — store a printable
        # version of the ground truth for the JSONL/responses dump.
        "gold": json.dumps(gt, default=str) if gt is not None else None,
        "ground_truth": gt,
        "category": row.get("category"),
        "prob_type": row.get("category"),
        "bfcl_id": row.get("bfcl_id"),
    }


def run(args, method_cfg):
    return BfclEvaluator().run(args, method_cfg)
