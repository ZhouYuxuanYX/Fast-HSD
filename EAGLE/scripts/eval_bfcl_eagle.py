"""Evaluate EAGLE output JSONL files on BFCL (Berkeley Function-Calling Leaderboard).

Reads ground truth from eagle/data/bfcl/question.jsonl (created by prepare_bfcl_questions.py),
parses predicted function calls from model outputs, and reports per-category and
overall accuracy alongside decoding speed / block efficiency.

Usage:
    python scripts/eval_bfcl_eagle.py bfcl/*.jsonl
    python scripts/eval_bfcl_eagle.py bfcl/run1.jsonl bfcl/run2.jsonl
"""

import argparse
import ast
import json
import os
import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_references(question_file: str):
    """Return {question_id: {"category": str, "bfcl_id": str, "reference": list}}."""
    refs = {}
    with open(question_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            refs[rec["question_id"]] = {
                "category":  rec.get("category", "simple"),
                "bfcl_id":   rec.get("bfcl_id", ""),
                "reference": rec.get("reference", []),
            }
    return refs


# ---------------------------------------------------------------------------
# Function-call parsing
# ---------------------------------------------------------------------------

def parse_function_calls(model_output: str) -> List[Dict]:
    """
    Parse model output into a list of {func_name: {param: value}} dicts.
    Expected format: [func_name1(p1=v1, ...), func_name2(p2=v2, ...)]
    """
    text = model_output.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Ensure bracketed
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return []

    if isinstance(tree.body, (ast.List, ast.Tuple)):
        nodes = tree.body.elts
    elif isinstance(tree.body, ast.Call):
        nodes = [tree.body]
    else:
        return []

    results = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        call = _resolve_call(node)
        if call:
            results.append(call)
    return results


def _resolve_call(node: ast.Call) -> Optional[Dict]:
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        n = node.func
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        func_name = ".".join(reversed(parts))
    else:
        return None

    params = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        params[kw.arg] = _ast_literal(kw.value)
    for i, arg in enumerate(node.args):
        params[f"_pos_{i}"] = _ast_literal(arg)

    return {func_name: params}


def _ast_literal(node):
    try:
        return ast.literal_eval(ast.Expression(body=node))
    except Exception:
        pass
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_ast_literal(k): _ast_literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_literal(node.operand)
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    return repr(node)


# ---------------------------------------------------------------------------
# Accuracy checking
# ---------------------------------------------------------------------------

def check_correct(model_calls: List[Dict], reference: List[Dict], category: str) -> bool:
    if category in ("parallel", "parallel_multiple"):
        return _check_parallel(model_calls, reference)
    else:
        return _check_simple(model_calls, reference)


def _check_simple(model_calls, ground_truth):
    """One correct call must appear in model_calls."""
    if not model_calls or not ground_truth:
        return False
    gt = ground_truth[0]
    gt_func = list(gt.keys())[0]
    gt_params = gt[gt_func]
    for call in model_calls:
        fn = list(call.keys())[0]
        if fn == gt_func and _params_match(call[fn], gt_params):
            return True
    return False


def _check_parallel(model_calls, ground_truth):
    """All ground-truth calls must appear in model_calls."""
    if len(model_calls) < len(ground_truth):
        return False
    for gt_call in ground_truth:
        gt_func = list(gt_call.keys())[0]
        gt_params = gt_call[gt_func]
        matched = False
        for mc in model_calls:
            if list(mc.keys())[0] == gt_func and _params_match(mc[gt_func], gt_params):
                matched = True
                break
        if not matched:
            return False
    return True


def _params_match(model_params: Dict, gt_params: Dict) -> bool:
    for param, acceptable in gt_params.items():
        if not isinstance(acceptable, list):
            acceptable = [acceptable]
        is_optional = "" in acceptable
        acceptable = [v for v in acceptable if v != ""]

        if param not in model_params:
            if not is_optional:
                return False
            continue

        if acceptable and not _value_matches(model_params[param], acceptable):
            return False
    return True


def _value_matches(val, acceptable: list) -> bool:
    return any(_values_equal(val, a) for a in acceptable)


def _values_equal(a, b) -> bool:
    if a == b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False
    if isinstance(a, str) and isinstance(b, (int, float)):
        try:
            return _values_equal(float(a) if "." in a else int(a), b)
        except (ValueError, TypeError):
            return False
    if isinstance(b, str) and isinstance(a, (int, float)):
        return _values_equal(b, a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    return False


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(path: str, refs: dict):
    total_tokens = 0
    total_time   = 0.0
    total_accepted = total_block_slots = total_max_slots = total_steps = 0
    correct = 0
    graded  = 0

    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            turn   = record["choices"][0]
            total_tokens += sum(turn["new_tokens"])
            total_time   += sum(turn["wall_time"])

            if "step_stats" in turn:
                for acc, bsz in turn["step_stats"]:
                    total_accepted    += acc
                    total_block_slots += bsz
                    total_max_slots   += (bsz - 1)
                    total_steps       += 1

            qid = record["question_id"]
            if qid not in refs:
                continue

            ref      = refs[qid]
            category = ref["category"]
            cat_total[category] = cat_total.get(category, 0) + 1
            graded += 1

            model_calls = parse_function_calls(turn["turns"][-1])
            if check_correct(model_calls, ref["reference"], category):
                correct += 1
                cat_correct[category] = cat_correct.get(category, 0) + 1

    n = total_tokens  # number of questions = len of records
    # recalculate n properly
    n_records = 0
    with open(path) as f:
        for _ in f:
            n_records += 1

    result = {
        "questions": n_records,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "speed": total_tokens / total_time if total_time > 0 else 0.0,
        "acc": correct / graded * 100 if graded > 0 else float("nan"),
        "correct": correct,
        "graded": graded,
        "cat_correct": cat_correct,
        "cat_total": cat_total,
    }
    if total_block_slots > 0:
        result["be"] = total_accepted / total_steps if total_steps > 0 else 0.0
        result["block_efficiency"] = total_accepted / total_block_slots
        result["avg_accepted_per_step"] = total_accepted / total_steps
    return result


def print_stats(path: str, s: dict, expected_n: int = None, show_cat: bool = False):
    n = s["questions"]
    warn = (f"  *** WARNING: only {n} questions (expected {expected_n}) ***"
            if expected_n and n < expected_n else "")
    print(f"\n{path}{warn}")
    print(f"  Questions           : {n}")
    print(f"  Total tokens        : {s['total_tokens']}")
    print(f"  Total time (s)      : {s['total_time']:.3f}")
    print(f"  Decoding speed      : {s['speed']:.2f} tokens/s")
    if "avg_accepted_per_step" in s:
        print(f"  Avg accepted/step   : {s['avg_accepted_per_step']:.3f}"
              f"  (block_eff: {s['block_efficiency']:.3f})")
    print(f"  Accuracy            : {s['acc']:.2f}%  ({s['correct']}/{s['graded']})")

    if show_cat and s["cat_total"]:
        print("  Per-category accuracy:")
        for cat in sorted(s["cat_total"]):
            c = s["cat_correct"].get(cat, 0)
            t = s["cat_total"][cat]
            print(f"    {cat:<20} {c}/{t}  ({c/t:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="EAGLE output JSONL file(s)")
    parser.add_argument(
        "--question-file",
        default=None,
        help="Path to bfcl question.jsonl (default: eagle/data/bfcl/question.jsonl)",
    )
    parser.add_argument(
        "--show-cat", action="store_true",
        help="Print per-category accuracy breakdown",
    )
    args = parser.parse_args()

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    eagle_dir     = os.path.dirname(script_dir)
    question_file = args.question_file or os.path.join(
        eagle_dir, "eagle", "data", "bfcl", "question.jsonl"
    )

    if not os.path.exists(question_file):
        print(f"ERROR: question file not found: {question_file}")
        print("Run:  python scripts/prepare_bfcl_questions.py")
        raise SystemExit(1)

    refs       = load_references(question_file)
    expected_n = len(refs)
    print(f"Loaded {expected_n} BFCL references from {question_file}\n")

    results = []
    for path in args.files:
        s = analyze(path, refs)
        results.append((path, s))
        print_stats(path, s, expected_n=expected_n, show_cat=args.show_cat)

    if len(results) > 1:
        print("\n--- Summary ---")
        header = f"{'File':<70} {'Speed':>8} {'BE':>6} {'Acc (%)':>9}"
        print(header)
        print("-" * len(header))
        for path, s in results:
            be_str = f"{s['avg_accepted_per_step']:.2f}" if "avg_accepted_per_step" in s else "  N/A"
            row = f"{path:<70} {s['speed']:>8.2f} {be_str:>6} {s['acc']:>9.2f}"
            if expected_n and s["questions"] < expected_n:
                row += f"  ⚠ {s['questions']}/{expected_n}"
            print(row)


if __name__ == "__main__":
    main()
