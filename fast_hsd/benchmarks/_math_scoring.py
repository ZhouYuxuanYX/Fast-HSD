"""MATH answer extraction and equivalence — ported from the legacy script.

Source: ``verification/src/MATH/eval_math.py`` (methods
``extract_boxed_answer``, ``extract_answer_from_response``, ``normalize_answer``,
``extract_elements_from_answer``, ``normalize_element``, ``compare_answers``).

The legacy logic lives on a class. We lift it to free functions so the
refactored benchmark runner can import it without dragging in the class's
heavy model-loading dependencies.
"""

from __future__ import annotations

import re
from typing import List, Optional


def extract_boxed_answer(solution: str) -> str:
    """Extract the inside of the last ``\\boxed{...}`` block in ``solution``.

    Handles nested braces. Falls back to a ``\\boxed <something>`` shape
    (no braces) before giving up.
    """
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, solution))
    if not matches:
        alt_match = re.search(r"\\boxed\s*(\S+)", solution)
        if alt_match:
            return alt_match.group(1)
        return ""

    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    pos = start
    while pos < len(solution) and brace_count > 0:
        if solution[pos] == "{":
            brace_count += 1
        elif solution[pos] == "}":
            brace_count -= 1
        pos += 1
    if brace_count == 0:
        return solution[start : pos - 1].strip()
    return ""


def extract_answer(response: str) -> str:
    """Extract the model's answer from a response string.

    Tries ``\\boxed{}`` first, then "the answer is X", then ``= X`` at end of
    line, then ``$...$`` at the end, then ``**X**``. Returns ``""`` if no
    pattern matches.
    """
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed

    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\.\n]+)", response)
    if m:
        return m.group(1).strip()

    matches = re.findall(r"=\s*([^\n=]+?)\s*$", response, re.MULTILINE)
    if matches:
        return matches[-1].strip()

    m = re.search(r"\$([^$]+)\$\s*\.?\s*$", response)
    if m:
        return m.group(1).strip()

    m = re.search(r"\*\*([^*]+)\*\*\s*\.?\s*$", response)
    if m:
        return m.group(1).strip()

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for string comparison.

    Strips common LaTeX commands, rewrites ``\\frac{a}{b}`` as ``(a)/(b)``,
    flattens ``\\sqrt``, collapses whitespace, removes trailing periods.
    """
    if not answer:
        return ""
    answer = answer.strip()
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\$", "", answer)
    answer = re.sub(r"\\,", "", answer)
    answer = re.sub(r"\\;", "", answer)
    answer = re.sub(r"\\!", "", answer)
    answer = re.sub(r"\\quad", " ", answer)
    answer = re.sub(r"\\qquad", " ", answer)

    # \frac{-a}{b} -> -\frac{a}{b}
    answer = re.sub(r"\\frac\{-([^{}]*)\}\{([^{}]*)\}", r"-\\frac{\1}{\2}", answer)
    answer = re.sub(r"\\dfrac\{-([^{}]*)\}\{([^{}]*)\}", r"-\\dfrac{\1}{\2}", answer)

    while r"\frac" in answer:
        frac_match = re.search(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", answer)
        if frac_match:
            num, den = frac_match.groups()
            answer = answer[: frac_match.start()] + f"({num})/({den})" + answer[frac_match.end() :]
        else:
            break
    while r"\dfrac" in answer:
        frac_match = re.search(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", answer)
        if frac_match:
            num, den = frac_match.groups()
            answer = answer[: frac_match.start()] + f"({num})/({den})" + answer[frac_match.end() :]
        else:
            break

    answer = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", answer)
    answer = re.sub(r"\\([a-zA-Z]+)", r"\1", answer)

    answer = re.sub(r"\s*,\s*", ",", answer)
    answer = re.sub(r"\(\s+", "(", answer)
    answer = re.sub(r"\s+\)", ")", answer)
    answer = " ".join(answer.split())
    answer = answer.rstrip(".")
    return answer


def _extract_elements(answer: str) -> List[str]:
    answer = answer.strip()
    if (answer.startswith("(") and answer.endswith(")")) or (
        answer.startswith("{") and answer.endswith("}")
    ):
        answer = answer[1:-1]
    return [e.strip() for e in answer.split(",")]


def _normalize_element(elem: str) -> Optional[float]:
    elem = elem.strip()
    frac_match = re.match(r"^(-?)\(?(-?\d+)\)?/\(?(\d+)\)?$", elem)
    if frac_match:
        sign = -1 if frac_match.group(1) == "-" else 1
        num = int(frac_match.group(2))
        den = int(frac_match.group(3))
        return sign * num / den
    pct_match = re.match(r"^(-?[\d.]+)\s*%$", elem)
    if pct_match:
        return float(pct_match.group(1)) / 100
    try:
        return float(elem)
    except Exception:
        return None


def is_equiv(pred: str, gt: str) -> bool:
    """Compare a predicted answer to ground truth, returning True if equivalent.

    Handles: multiple gt forms separated by ``=``, direct (case-insensitive)
    string equality after normalization, comma-separated tuples (order-
    independent, both string and numeric), and numerical equality with 1e-6
    tolerance.
    """
    if not pred or not gt:
        return False

    if "=" in gt:
        for gt_alt in (g.strip() for g in gt.split("=")):
            if is_equiv(pred, gt_alt):
                return True

    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    if pred_norm.lower() == gt_norm.lower():
        return True

    if "," in pred_norm or "," in gt_norm:
        pred_elements = _extract_elements(pred_norm)
        gt_elements = _extract_elements(gt_norm)
        if len(pred_elements) == len(gt_elements) and len(pred_elements) > 1:
            if sorted(e.lower() for e in pred_elements) == sorted(
                e.lower() for e in gt_elements
            ):
                return True
            pred_nums = [_normalize_element(e) for e in pred_elements]
            gt_nums = [_normalize_element(e) for e in gt_elements]
            if None not in pred_nums and None not in gt_nums:
                pred_sorted = sorted(pred_nums)
                gt_sorted = sorted(gt_nums)
                if all(abs(p - g) <= 1e-6 for p, g in zip(pred_sorted, gt_sorted)):
                    return True

    # Numerical equality on the whole answer.
    pred_num = _normalize_element(pred_norm)
    gt_num = _normalize_element(gt_norm)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) <= 1e-6

    return False
