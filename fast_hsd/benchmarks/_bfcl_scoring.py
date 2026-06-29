"""BFCL function-call extraction + AST-based checkers.

Ported from ``verification/src/bfcl/eval_bfcl.py`` (lines 167-429). Self-
contained so the refactor doesn't import the legacy module.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

__all__ = [
    "parse_function_calls",
    "check_simple",
    "check_parallel",
    "check_irrelevance",
    "CATEGORY_CHECKERS",
    "score_bfcl",
]


# ---------- parsing ----------------------------------------------------


def parse_function_calls(model_output: str) -> List[Dict[str, Dict]]:
    """Parse ``[func_name(p=v, ...), ...]`` into ``[{func_name: {p: v}}, ...]``.

    Strips markdown code fences first; wraps a bare single call in ``[]``.
    Returns ``[]`` on syntax errors.
    """
    model_output = (model_output or "").strip()

    if model_output.startswith("```"):
        lines = model_output.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        model_output = "\n".join(lines).strip()

    if not model_output.startswith("["):
        model_output = "[" + model_output
    if not model_output.endswith("]"):
        model_output = model_output + "]"

    try:
        tree = ast.parse(model_output, mode="eval")
    except SyntaxError:
        return []

    if isinstance(tree.body, (ast.List, ast.Tuple)):
        calls = tree.body.elts
    elif isinstance(tree.body, ast.Call):
        calls = [tree.body]
    else:
        return []

    results = []
    for node in calls:
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_ast_call(node)
        if resolved:
            results.append(resolved)
    return results


def _resolve_ast_call(node: ast.Call) -> Optional[Dict]:
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

    params: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        params[kw.arg] = _ast_literal(kw.value)
    for i, arg in enumerate(node.args):
        params[f"_pos_{i}"] = _ast_literal(arg)
    return {func_name: params}


def _ast_unparse(node) -> str:
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    try:
        return repr(ast.literal_eval(compile(ast.Expression(body=node), "<>", "eval")))
    except Exception:
        pass
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return "[" + ", ".join(_ast_unparse(e) for e in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_ast_unparse(e) for e in node.elts) + ")"
    if isinstance(node, ast.Dict):
        return "{" + ", ".join(
            f"{_ast_unparse(k)}: {_ast_unparse(v)}" for k, v in zip(node.keys, node.values)
        ) + "}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_ast_unparse(node.operand)}"
    return repr(node)


def _ast_literal(node):
    try:
        return eval(compile(ast.Expression(body=node), "<>", "eval"))
    except Exception:
        return _ast_unparse(node)


# ---------- category checkers -----------------------------------------


def check_simple(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    """``simple``/``multiple``: one expected call must match one of the model calls."""
    if not model_calls or not ground_truth:
        return False
    gt = ground_truth[0]
    gt_func_name = next(iter(gt.keys()))
    gt_params = gt[gt_func_name]
    for call in model_calls:
        call_func = next(iter(call.keys()))
        if call_func != gt_func_name:
            continue
        if _params_match(call[call_func], gt_params):
            return True
    return False


def check_parallel(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    """``parallel``/``parallel_multiple``: all expected calls must be matched (order-agnostic)."""
    if len(model_calls) < len(ground_truth):
        return False
    gt_matched = [False] * len(ground_truth)
    for gt_idx, gt_call in enumerate(ground_truth):
        gt_func = next(iter(gt_call.keys()))
        gt_params = gt_call[gt_func]
        for model_call in model_calls:
            model_func = next(iter(model_call.keys()))
            if model_func != gt_func:
                continue
            if _params_match(model_call[model_func], gt_params):
                gt_matched[gt_idx] = True
                break
    return all(gt_matched)


def check_irrelevance(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    return len(model_calls) == 0


CATEGORY_CHECKERS = {
    "simple": check_simple,
    "multiple": check_simple,
    "parallel": check_parallel,
    "parallel_multiple": check_parallel,
    "irrelevance": check_irrelevance,
}


# ---------- value comparison ------------------------------------------


def _params_match(model_params: Dict, gt_params: Dict) -> bool:
    for param_name, acceptable_values in gt_params.items():
        if not isinstance(acceptable_values, list):
            acceptable_values = [acceptable_values]
        is_optional = "" in acceptable_values
        acceptable_clean = [v for v in acceptable_values if v != ""]

        if param_name not in model_params:
            if is_optional:
                continue
            return False

        if not _value_matches(model_params[param_name], acceptable_clean):
            return False
    return True


def _value_matches(model_val, acceptable_vals: list) -> bool:
    return any(_values_equal(model_val, v) for v in acceptable_vals)


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
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    return False


# ---------- top-level scorer ------------------------------------------


def score_bfcl(response: str, ground_truth: List[Dict], category: str) -> tuple:
    """Return ``(correct, model_calls_str)`` for the given BFCL category."""
    checker = CATEGORY_CHECKERS.get(category, check_simple)
    model_calls = parse_function_calls(response)
    correct = checker(model_calls, ground_truth or [])
    # Stash a compact stringified version of the parsed calls so the JSONL
    # row keeps something diff-friendly even when the raw response is long.
    import json as _json
    try:
        rendered = _json.dumps(model_calls, default=str)
    except Exception:
        rendered = repr(model_calls)
    return (correct, rendered)
