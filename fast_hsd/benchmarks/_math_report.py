"""End-of-run math evaluation report.

Shared between the live inference (``fast-hsd-eval --benchmark math``) and
the standalone ``scripts/results_analysis_math.py``. Reads either schema:

- legacy EAGLE ``gen_ea_answer_*.py``: ``choices[0].{turns,new_tokens,
  wall_time,step_stats}``.
- refactor ``fast-hsd-eval`` ``rows.jsonl``: flat ``response``/
  ``output_tokens``/``decoding_seconds``/``n_matched_per_block`` +
  ``draft_eval_per_block``.

Scoring uses the unified extractors and ``is_equiv`` from
:mod:`fast_hsd.benchmarks._math_scoring`, so a number printed here will
match the per-row ``correct`` field the inference loop produced.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fast_hsd.benchmarks._math_scoring import (
    extract_answer,
    extract_ref_answer,
    is_equiv,
)

__all__ = [
    "load_references",
    "analyze",
    "print_stats",
    "_normalize_record",
    "_qid",
]


def _qid(x):
    """Coerce a question id to int when it's an integer string, so the
    refactor's stringified ids match the integer ids from competition_math."""
    s = str(x)
    return int(s) if s.lstrip("-").isdigit() else x


def _normalize_record(record: Dict[str, Any]):
    """Return ``(response, new_tokens, wall_time, step_stats)`` for either
    the legacy ``choices`` schema or the refactor flat ``rows.jsonl`` schema.
    """
    if "choices" in record:  # legacy
        turns = record["choices"][0]
        return (
            turns["turns"][-1],
            turns.get("new_tokens", []),
            turns.get("wall_time", []),
            turns.get("step_stats", []),
        )
    # refactor rows.jsonl
    nm = record.get("n_matched_per_block") or []
    de = record.get("draft_eval_per_block") or []
    return (
        record.get("response", ""),
        [record.get("output_tokens", 0)],
        [record.get("decoding_seconds", 0.0)],
        [[int(a), int(b)] for a, b in zip(nm, de)],
    )


def load_references(question_file: str) -> Dict[Any, Optional[str]]:
    """Return ``{question_id: gold_answer_str}`` from a ``question.jsonl``."""
    refs: Dict[Any, Optional[str]] = {}
    with open(question_file) as f:
        for line in f:
            record = json.loads(line)
            qid = _qid(record["question_id"])
            ref_field = record.get("reference")
            if isinstance(ref_field, list):
                ref_text = ref_field[-1] if ref_field else ""
            else:
                ref_text = ref_field or ""
            refs[qid] = extract_ref_answer(ref_text)
    return refs


def analyze(path: str, refs: Optional[Dict[Any, Optional[str]]] = None) -> Dict[str, Any]:
    total_tokens = 0
    total_time = 0.0
    question_times = []
    correct = 0
    graded = 0
    total_accepted = 0
    total_block_slots = 0
    total_max_slots = 0
    total_steps = 0

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            response, new_tokens, wall_time, step_stats = _normalize_record(record)
            total_tokens += sum(new_tokens)
            q_time = sum(wall_time)
            total_time += q_time
            question_times.append(q_time)

            for accepted, block_size in step_stats:
                total_accepted += accepted
                total_block_slots += block_size
                total_max_slots += (block_size - 1)
                total_steps += 1

            if refs is not None:
                qid = _qid(record["question_id"])
                pred = extract_answer(response)
                ref = refs.get(qid)
                if ref is not None:
                    graded += 1
                    if is_equiv(pred, ref):
                        correct += 1

    n = len(question_times)
    result: Dict[str, Any] = {
        "questions": n,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "speed": (total_tokens / total_time) if total_time > 0 else float("nan"),
        "time_per_q": (total_time / n) if n else float("nan"),
    }
    if total_block_slots > 0:
        result["block_efficiency"] = total_accepted / total_block_slots
        result["be_ratio"] = total_accepted / total_max_slots if total_max_slots else 0.0
        result["avg_accepted_per_step"] = total_accepted / total_steps
    if refs is not None:
        result["pass1"] = (correct / graded) if graded > 0 else float("nan")
        result["correct"] = correct
        result["graded"] = graded
    return result


def print_stats(path: str, s: Dict[str, Any], expected_n: Optional[int] = None) -> None:
    n = s["questions"]
    warn = (
        f"  *** WARNING: only {n} questions (expected {expected_n}) — results incomplete ***"
        if expected_n and n < expected_n
        else ""
    )
    print(f"\n{path}{warn}")
    print(f"  Questions           : {n}{' INCOMPLETE' if expected_n and n < expected_n else ''}")
    print(f"  Total tokens        : {s['total_tokens']}")
    print(f"  Total time (s)      : {s['total_time']:.3f}")
    print(f"  Decoding speed      : {s['speed']:.2f} tokens/s")
    print(f"  Time / question     : {s['time_per_q']:.3f} s")
    if "block_efficiency" in s:
        print(
            f"  Block efficiency    : {s['block_efficiency']:.3f}  (ratio: {s.get('be_ratio', 0):.3f})"
        )
        print(f"  Avg accepted/step   : {s['avg_accepted_per_step']:.3f}")
    if "pass1" in s:
        print(f"  Pass@1              : {s['pass1']:.1%}  ({s['correct']}/{s['graded']})")
