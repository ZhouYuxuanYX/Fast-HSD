"""Evaluate EAGLE output JSONL files on MBPP+ pass@1.

Loads test cases from evalplus/mbppplus, executes generated code against
them, and reports pass@1 accuracy alongside decoding speed / block efficiency.

Usage:
    python scripts/eval_mbppplus.py mbppplus/*.jsonl
    python scripts/eval_mbppplus.py mbppplus/run1.jsonl mbppplus/run2.jsonl
"""
import argparse
import json
import os
import re
import signal
import sys

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_test_cases():
    """Return list of dicts with keys: task_id, prompt, test_list, test_imports.
    Indexed by question_id (0-based, matching question.jsonl order)."""
    ds = load_dataset("evalplus/mbppplus", split="test")
    return list(ds)


# ---------------------------------------------------------------------------
# Code extraction (from reference eval_mbppplus.py)
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    # 1) ```python ... ```
    matches = re.findall(r'```[pP]ython\s*\n(.*?)```', response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 2) ``` ... ``` generic block
    matches = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if matches:
        py = [m for m in matches if re.search(r'^\s*(def |import |from |class )', m, re.MULTILINE)]
        return max(py or matches, key=len).strip()

    # 3) Inline function definition(s)
    matches = re.findall(
        r'((?:import\s+\w+.*?\n)*(?:from\s+\w+.*?\n)*\s*def\s+\w+\s*\([^)]*\)\s*:.*?)'
        r'(?=\n(?:def\s|\Z|```|[A-Z][a-z]+:|\n\n[A-Z]))',
        response, re.DOTALL
    )
    if matches:
        return '\n\n'.join(m.strip() for m in matches)

    # 4) Fallback: everything from first def/import onward
    lines, in_code, code_lines = response.split('\n'), False, []
    for line in lines:
        if re.match(r'^(import |from |def |class |@)', line):
            in_code = True
        if in_code:
            if re.match(r'^[A-Z][a-z]+.*:$', line) and not line.strip().endswith('"""'):
                break
            code_lines.append(line)
    return '\n'.join(code_lines).strip() if code_lines else response


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

_COMMON_IMPORTS = """
import math, heapq, itertools, collections, functools, operator, re, sys
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import permutations, combinations, product
from functools import reduce, lru_cache
from typing import List, Dict, Tuple, Optional, Set
"""

_BUILTIN_NAMES = {
    "set", "list", "tuple", "sorted", "len", "str", "int", "float", "bool",
    "dict", "frozenset", "sum", "max", "min", "round", "abs", "type",
    "range", "enumerate", "zip", "map", "filter", "any", "all", "print",
    "isinstance", "repr", "hash",
}


def _expected_func_name(test_list: list) -> "str | None":
    """Extract the function name the tests actually call (skipping builtins)."""
    if not test_list:
        return None
    test = test_list[0]
    m = re.search(r'\bassert\b', test)
    if not m:
        return None
    for hit in re.finditer(r'\b(\w+)\s*\(', test[m.end():]):
        if hit.group(1) not in _BUILTIN_NAMES:
            return hit.group(1)
    return None


def _inject_alias(code: str, test_list: list) -> str:
    """If the code doesn't define the expected function name, append an alias."""
    expected = _expected_func_name(test_list)
    if expected is None:
        return code
    defined = re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if not defined or expected in defined:
        return code
    # Pick the most likely candidate: last top-level def
    alias_target = defined[-1]
    return code + f"\n{expected} = {alias_target}\n"

def _timeout_handler(signum, frame):
    raise TimeoutError()


def run_tests(code: str, test_list: list, test_imports: list, timeout: int = 10):
    """Execute code and test assertions. Returns (all_passed, num_passed, total)."""
    extracted = extract_code(code)
    extracted = _inject_alias(extracted, test_list)
    ns = {"__builtins__": __builtins__}

    try:
        exec(_COMMON_IMPORTS, ns)
    except Exception:
        pass

    old_handler = None
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        for imp in (test_imports or []):
            if imp and imp.strip():
                try:
                    exec(imp, ns)
                except Exception:
                    pass

        exec(extracted, ns)

        num_passed = 0
        for test in test_list:
            try:
                exec(test, ns)
                num_passed += 1
            except Exception:
                pass

        total = len(test_list)
        return num_passed == total and total > 0, num_passed, total

    except TimeoutError:
        return False, 0, len(test_list)
    except Exception:
        return False, 0, len(test_list)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(path: str, test_cases: list):
    total_tokens = 0
    total_time = 0.0
    question_times = []
    total_accepted = 0
    total_block_slots = 0
    total_max_slots = 0
    total_steps = 0
    correct = 0
    num_passed_total = 0
    num_tests_total = 0

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    for record in tqdm(records, desc=os.path.basename(path), leave=False):
        turns = record["choices"][0]
        total_tokens += sum(turns["new_tokens"])
        q_time = sum(turns["wall_time"])
        total_time += q_time
        question_times.append(q_time)

        if "step_stats" in turns:
            for accepted, block_size in turns["step_stats"]:
                total_accepted += accepted
                total_block_slots += block_size
                total_max_slots += (block_size - 1)
                total_steps += 1

        qid = record["question_id"]
        if qid < len(test_cases):
            tc = test_cases[qid]
            answer_text = turns["turns"][-1]
            passed, np_, nt_ = run_tests(answer_text, tc["test_list"], tc.get("test_imports", []))
            correct += int(passed)
            num_passed_total += np_
            num_tests_total += nt_

    n = len(records)
    result = {
        "questions": n,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "speed": total_tokens / total_time if total_time > 0 else 0.0,
        "time_per_q": total_time / n if n > 0 else 0.0,
        "pass1": correct / n if n > 0 else 0.0,
        "correct": correct,
        "test_pass_rate": num_passed_total / num_tests_total if num_tests_total > 0 else 0.0,
    }
    if total_block_slots > 0:
        result["block_efficiency"] = total_accepted / total_block_slots
        result["be_ratio"] = total_accepted / total_max_slots if total_max_slots > 0 else 0.0
        result["avg_accepted_per_step"] = total_accepted / total_steps
    return result


def print_stats(path: str, s: dict, expected_n: int = None):
    n = s["questions"]
    warn = f"  *** WARNING: only {n} questions (expected {expected_n}) ***" if expected_n and n < expected_n else ""
    print(f"\n{path}{warn}")
    print(f"  Questions           : {n}")
    print(f"  Total tokens        : {s['total_tokens']}")
    print(f"  Total time (s)      : {s['total_time']:.3f}")
    print(f"  Decoding speed      : {s['speed']:.2f} tokens/s")
    print(f"  Time / question     : {s['time_per_q']:.3f} s")
    if "block_efficiency" in s:
        print(f"  Block efficiency    : {s['block_efficiency']:.3f}  (ratio: {s.get('be_ratio', 0):.3f})")
        print(f"  Avg accepted/step   : {s['avg_accepted_per_step']:.3f}")
    print(f"  Pass@1              : {s['pass1']:.1%}  ({s['correct']}/{n})")
    print(f"  Test pass rate      : {s['test_pass_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="EAGLE output JSONL file(s)")
    args = parser.parse_args()

    print("Loading MBPP+ test cases...", flush=True)
    test_cases = load_test_cases()
    expected_n = len(test_cases)
    print(f"Loaded {expected_n} test cases.\n")

    results = []
    for path in args.files:
        s = analyze(path, test_cases)
        results.append((path, s))
        print_stats(path, s, expected_n=expected_n)

    if len(results) > 1:
        has_block = "block_efficiency" in results[0][1]
        print("\n--- Summary ---")
        header = f"{'File':<60} {'Speed (tok/s)':>14} {'Time/Q (s)':>12}"
        if has_block:
            header += f" {'BlkEff':>8} {'Acc/step':>9}"
        header += f" {'Pass@1':>8} {'TestPass':>9}"
        print(header)
        print("-" * len(header))
        for path, s in results:
            row = f"{path:<60} {s['speed']:>14.2f} {s['time_per_q']:>12.3f}"
            if has_block:
                row += f" {s['block_efficiency']:>8.3f} {s.get('be_ratio',0):>8.3f} {s['avg_accepted_per_step']:>9.3f}"
            row += f" {s['pass1']:>8.1%} {s['test_pass_rate']:>9.1%}"
            if expected_n and s["questions"] < expected_n:
                row += f"  ⚠ {s['questions']}/{expected_n}"
            print(row)


if __name__ == "__main__":
    main()
