"""MBPP+ code extraction + in-process Pass@1 scoring.

Ported verbatim from ``verification/src/mbppplus/eval_mbppplus.py``
(``extract_code_from_response`` lines 407-463; ``test_answer`` lines 465-561).

Timeout uses ``signal.SIGALRM`` (Unix only); on platforms without it the
test simply runs without a wall-clock limit.
"""

from __future__ import annotations

import builtins
import re
import signal
from typing import List, Optional, Sequence, Tuple

__all__ = ["extract_code", "run_pass_at_1", "score_mbppplus"]


_PY_BLOCK_RE = re.compile(r"```[pP]ython\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_PY_LIKE_RE = re.compile(r"^\s*(def |import |from |class )", re.MULTILINE)
_FUNC_DEF_RE = re.compile(
    r"((?:import\s+\w+.*?\n)*(?:from\s+\w+.*?\n)*\s*def\s+\w+\s*\([^)]*\)\s*:.*?)"
    r"(?=\n(?:def\s|\Z|```|[A-Z][a-z]+:|\n\n[A-Z]))",
    re.DOTALL,
)


def extract_code(response: str) -> str:
    """Extract the Python function definition from a model response."""
    matches = _PY_BLOCK_RE.findall(response)
    if matches:
        return max(matches, key=len).strip()

    matches = _GENERIC_BLOCK_RE.findall(response)
    if matches:
        python_matches = [m for m in matches if _PY_LIKE_RE.search(m)]
        if python_matches:
            return max(python_matches, key=len).strip()
        return max(matches, key=len).strip()

    matches = _FUNC_DEF_RE.findall(response)
    if matches:
        return "\n\n".join(m.strip() for m in matches)

    # Fallback: line-by-line scan for python-looking code.
    code_lines: List[str] = []
    in_code = False
    for line in response.split("\n"):
        if re.match(r"^(import |from |def |class |@)", line):
            in_code = True
        if in_code:
            if re.match(r"^[A-Z][a-z]+.*:$", line) and not line.strip().endswith('"""'):
                break
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return response


_COMMON_IMPORTS = """
import math
import heapq
import itertools
import collections
import functools
import operator
import re
import sys
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import permutations, combinations, product
from functools import reduce, lru_cache
from typing import List, Dict, Tuple, Optional, Set
"""


def run_pass_at_1(
    code: str,
    test_list: Sequence[str],
    test_imports: Optional[Sequence[str]] = None,
    timeout: int = 10,
) -> Tuple[bool, int, int]:
    """Execute ``code`` and run each assertion in ``test_list``.

    Returns ``(all_passed, num_passed, total_tests)``. Timeouts and
    runtime errors are caught; the function does not raise.
    """

    def _timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    total_tests = len(test_list) if test_list else 0
    try:
        extracted = extract_code(code)
        exec_globals = {"__builtins__": builtins.__dict__}
        try:
            exec(_COMMON_IMPORTS, exec_globals)
        except Exception:
            pass

        old_handler = None
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            if test_imports:
                for imp in test_imports:
                    if imp and imp.strip():
                        try:
                            exec(imp, exec_globals)
                        except Exception:
                            pass

            exec(extracted, exec_globals)

            num_passed = 0
            for test in test_list:
                try:
                    exec(test, exec_globals)
                    num_passed += 1
                except AssertionError:
                    pass
                except Exception:
                    pass

            return (num_passed == total_tests and total_tests > 0, num_passed, total_tests)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

    except TimeoutError:
        return (False, 0, total_tests)
    except Exception:
        return (False, 0, total_tests)


def score_mbppplus(
    response: str,
    test_list: Sequence[str],
    test_imports: Optional[Sequence[str]] = None,
    timeout: int = 10,
) -> tuple:
    """Return ``(passed, extracted_code_snippet, (n_passed, n_total))``.

    - ``passed`` — strict all-assertions-pass (problem-level correctness).
    - ``extracted_code_snippet`` — ``"[n/m] <first 200 chars of code>"``, kept
      short so it fits the JSONL ``extracted_answer`` field.
    - ``(n_passed, n_total)`` — assertion-level counts, aggregated by the
      runner into the summary's ``subtest_pass_rate`` (MBPP+ "test pass rate").
    """
    passed, n_passed, n_total = run_pass_at_1(response, test_list, test_imports, timeout=timeout)
    extracted = extract_code(response)
    summary = f"[{n_passed}/{n_total}] " + (extracted[:200].replace("\n", "\\n"))
    return (passed, summary, (n_passed, n_total))
