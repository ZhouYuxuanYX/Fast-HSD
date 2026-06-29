"""MBPP+ benchmark runner.

Ports ``verification/src/mbppplus/eval_mbppplus.py``:

- **Prompting** mirrors the legacy chat-template setup with the
  ``"You are Qwen ... code generation model. Output only the Python function
  definition ..."`` system prompt. The bundled
  ``EAGLE/eagle/data/mbppplus/question.jsonl`` already appends
  ``"The function should be named ``X``."`` to each prompt, so we don't
  recompute the function name from test assertions.
- **Scoring** is the legacy in-process Pass@1: extract the Python code from
  the response, exec it inside a sandbox with common imports, then run each
  assertion. Wall-clock-limited via SIGALRM. The test list is fetched from
  HuggingFace ``evalplus/mbppplus`` (matched by ``task_id``); the bundled
  JSONL only ships prompts and reference solutions, not the test suite.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fast_hsd.benchmarks._mbppplus_scoring import score_mbppplus
from fast_hsd.benchmarks.base import BenchmarkEvaluator

logger = logging.getLogger("fast_hsd.benchmarks.mbppplus")

__all__ = ["MbppPlusEvaluator", "run"]

SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a code generation model. "
    "Output only the Python function definition without any explanation or "
    "markdown formatting."
)

DEFAULT_QUESTION_FILE = "EAGLE/eagle/data/mbppplus/question.jsonl"
HF_DATASET = "evalplus/mbppplus"
MBPP_TIMEOUT_SECONDS = 10


class MbppPlusEvaluator(BenchmarkEvaluator):
    name = "mbppplus"
    system_prompt = SYSTEM_PROMPT

    def __init__(self):
        # Lazy-loaded ``{task_id: (test_list, test_imports)}`` keyed by task_id.
        self._tests_by_task_id: Optional[Dict[int, Tuple[List[str], List[str]]]] = None

    # ----- loading & prompting ------------------------------------------

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:
        path = getattr(args, "question_file", None) or DEFAULT_QUESTION_FILE
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                yield _normalize_row(json.loads(line))

    def format_prompt(self, question: Dict[str, Any]) -> str:
        return question["prompt_user"]

    # ----- scoring -------------------------------------------------------

    def _load_hf_tests(self) -> Dict[int, Tuple[List[str], List[str]]]:
        """Build a ``task_id → (test_list, test_imports)`` index from HF.

        Cached on the instance. Returns ``{}`` if HF isn't reachable so the
        run continues; per-row ``correct`` will be ``None`` in that case.
        """
        if self._tests_by_task_id is not None:
            return self._tests_by_task_id
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_DATASET, split="test")
            self._tests_by_task_id = {
                int(r["task_id"]): (
                    list(r.get("test_list") or []),
                    list(r.get("test_imports") or []),
                )
                for r in ds
            }
            logger.info("loaded %d MBPP+ test rows from %s", len(self._tests_by_task_id), HF_DATASET)
        except Exception as e:
            logger.warning(
                "could not load %s for Pass@1 scoring (%s); per-row 'correct' will be None",
                HF_DATASET,
                e,
            )
            self._tests_by_task_id = {}
        return self._tests_by_task_id

    def score(
        self, question: Dict[str, Any], response: str
    ) -> Optional[Tuple[Optional[bool], str]]:
        tests = self._load_hf_tests()
        if not tests:
            return None
        # The bundled question_id is the EAGLE row index (0..377), and on the
        # HF side rows are ordered by task_id which is the MBPP+ canonical id.
        # The two orderings match, so we look up by question_id-as-index when
        # task_id isn't explicit.
        task_id = question.get("task_id")
        qid = question.get("question_id")
        if task_id is None and isinstance(qid, int) and tests:
            # Sort task_ids by their natural order and use qid as positional index.
            task_id = _ith_task_id(tests, qid)
        entry = tests.get(int(task_id)) if task_id is not None else None
        if entry is None:
            return None
        test_list, test_imports = entry
        return score_mbppplus(response, test_list, test_imports, timeout=MBPP_TIMEOUT_SECONDS)


# --- helpers ---------------------------------------------------------------


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "turns" in row:
        turns = row["turns"]
        prompt_user = turns[0] if isinstance(turns, list) and turns else str(turns)
    else:
        prompt_user = row.get("prompt") or row.get("question") or ""

    if "question_id" in row:
        qid = row["question_id"]
    elif "id" in row:
        qid = row["id"]
    else:
        qid = None

    # Reference solution shipped in the bundled file; we surface it as the
    # JSONL ``gold`` field so it shows up in the human-readable dump.
    gold = None
    if "reference" in row:
        ref = row["reference"]
        gold = (ref[0] if isinstance(ref, list) and ref else ref)

    return {
        "question_id": qid,
        "prompt_user": prompt_user,
        "gold": gold,
        "task_id": row.get("task_id"),
        "prob_type": row.get("category"),
    }


def _ith_task_id(tests: Dict[int, Any], i: int) -> Optional[int]:
    """Return the i-th task_id in sorted order, or None if out of range."""
    try:
        sorted_ids = sorted(tests.keys())
        return sorted_ids[i] if 0 <= i < len(sorted_ids) else None
    except Exception:
        return None


def run(args, method_cfg):
    return MbppPlusEvaluator().run(args, method_cfg)
