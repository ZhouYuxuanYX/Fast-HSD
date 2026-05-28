import argparse
import json
import re
from fractions import Fraction
from typing import Optional

# ── answer extraction ────────────────────────────────────────────────────────

def _extract_boxed(text):
    """Extract content of the last \\boxed{} with proper brace matching."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start:pos - 1].strip() if depth == 0 else None


def extract_ref_answer(text):
    """Extract the ground-truth answer from a reference solution.

    Prefers \\boxed{}, then #### marker, then last number.
    """
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_model_answer(text):
    """Extract the predicted answer from a model response.

    Tries in order:
      1. \\boxed{...}
      2. #### marker  (GSM8K style)
      3. "The answer is X" / "the final answer is X"
      4. Last "= X" on a line
      5. Last $...$ expression at end
      6. Last **X** bold marker
    """
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\.\n]+)", text)
    if m:
        return m.group(1).strip()

    matches = re.findall(r"=\s*([^\n=]+?)\s*$", text, re.MULTILINE)
    if matches:
        return matches[-1].strip()

    m = re.search(r"\$([^$]+)\$\s*\.?\s*$", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"\*\*([^*]+)\*\*\s*\.?\s*$", text)
    if m:
        return m.group(1).strip()

    return None

# ── answer normalisation & comparison ────────────────────────────────────────

def _normalize(answer):
    """Strip LaTeX formatting and normalise whitespace."""
    s = str(answer).strip()
    # remove common LaTeX text commands
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\[,;!]", "", s)
    s = re.sub(r"\\quad|\\qquad", " ", s)
    # \frac{a}{b} -> (a)/(b)
    for cmd in (r"\\frac", r"\\dfrac"):
        while cmd in s:
            fm = re.search(cmd.replace("\\", "\\\\") + r"\{([^{}]*)\}\{([^{}]*)\}", s)
            if fm:
                s = s[:fm.start()] + f"({fm.group(1)})/({fm.group(2)})" + s[fm.end():]
            else:
                break
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)  # remove remaining backslash commands
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = " ".join(s.split()).rstrip(".")
    return s


def _to_float(s):
    """Try to parse s as a number or fraction; return float or None."""
    s = s.strip()
    m = re.match(r"^\(?(-?\d+)\)?/\(?(\d+)\)?$", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.match(r"^(-?[\d.]+)\s*%$", s)
    if m:
        try:
            return float(m.group(1)) / 100
        except ValueError:
            return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _answers_equal(pred, ref):
    """Return True if pred and ref represent the same mathematical answer."""
    if pred is None or ref is None:
        return False

    # Handle "a = b" alternative forms in ground truth
    if "=" in ref:
        for alt in ref.split("="):
            if _answers_equal(pred, alt.strip()):
                return True

    pred_n = _normalize(pred)
    ref_n  = _normalize(ref)

    if pred_n.lower() == ref_n.lower():
        return True

    # Comma-separated list / tuple comparison (order-independent)
    if "," in pred_n or "," in ref_n:
        def _elements(s):
            s = s.strip()
            if (s.startswith("(") and s.endswith(")")) or \
               (s.startswith("{") and s.endswith("}")):
                s = s[1:-1]
            return [e.strip() for e in s.split(",")]
        pe = _elements(pred_n)
        re_ = _elements(ref_n)
        if len(pe) == len(re_) and len(pe) > 1:
            pv = [_to_float(e) for e in pe]
            rv = [_to_float(e) for e in re_]
            if all(v is not None for v in pv + rv):
                return sorted(pv) == sorted(rv)  # exact after float conversion

    # Numeric comparison with small tolerance
    pv = _to_float(pred_n)
    rv = _to_float(ref_n)
    if pv is not None and rv is not None:
        return abs(pv - rv) <= 1e-6 * max(1.0, abs(rv))

    # Symbolic comparison via sympy (optional, skipped if sympy unavailable)
    try:
        from sympy import simplify, nsimplify
        from sympy.parsing.latex import parse_latex

        def _sym(s):
            try:
                return parse_latex(s)
            except Exception:
                pass
            try:
                s2 = s.replace("^", "**").replace("sqrt", "__import__('sympy').sqrt")
                return eval(s2, {"__builtins__": {}})
            except Exception:
                return None

        ps = _sym(pred_n)
        rs = _sym(ref_n)
        if ps is not None and rs is not None:
            diff = simplify(ps - rs)
            if diff == 0:
                return True
    except Exception:
        pass

    return False


def _qid(x):
    """Coerce a question id to int when it's an integer string, so the
    refactor's stringified ids match the competition_math integer ids."""
    s = str(x)
    return int(s) if s.lstrip("-").isdigit() else x


def _normalize_record(record):
    """Return (response_text, new_tokens, wall_time, step_stats) for either
    schema:

    - legacy EAGLE ``gen_ea_answer_*.py``: ``choices[0].{turns,new_tokens,
      wall_time,step_stats}``;
    - refactor ``fast-hsd-eval`` rows.jsonl: flat ``response`` /
      ``output_tokens`` / ``decoding_seconds`` / ``n_matched_per_block`` +
      ``draft_eval_per_block``.
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


def load_references(question_file):
    """Return dict of question_id -> reference answer string."""
    refs = {}
    with open(question_file) as f:
        for line in f:
            record = json.loads(line)
            qid = _qid(record["question_id"])
            ref_text = record["reference"][-1] if isinstance(record["reference"], list) else record["reference"]
            refs[qid] = extract_ref_answer(ref_text)
    return refs

def analyze(path, refs=None):
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

            # Block efficiency from per-step stats: list of [accepted_draft_tokens, block_size]
            for accepted, block_size in step_stats:
                total_accepted += accepted
                total_block_slots += block_size
                total_max_slots += (block_size - 1)
                total_steps += 1

            if refs is not None:
                qid = _qid(record["question_id"])
                pred = extract_model_answer(response)
                ref = refs.get(qid)
                if ref is not None:
                    graded += 1
                    if _answers_equal(pred, ref):
                        correct += 1

    n = len(question_times)
    result = {
        "questions": n,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "speed": total_tokens / total_time,
        "time_per_q": total_time / n,
    }
    if total_block_slots > 0:
        result["block_efficiency"] = total_accepted / total_block_slots
        result["be_ratio"] = total_accepted / total_max_slots if total_max_slots > 0 else 0.0
        result["avg_accepted_per_step"] = total_accepted / total_steps
    if refs is not None:
        result["pass1"] = correct / graded if graded > 0 else float("nan")
        result["correct"] = correct
        result["graded"] = graded
    return result

def print_stats(path, s, expected_n=None):
    n = s['questions']
    warn = f"  *** WARNING: only {n} questions (expected {expected_n}) — results incomplete ***" if expected_n and n < expected_n else ""
    print(f"\n{path}{warn}")
    print(f"  Questions           : {n}{' ⚠ INCOMPLETE' if expected_n and n < expected_n else ''}")
    print(f"  Total tokens        : {s['total_tokens']}")
    print(f"  Total time (s)      : {s['total_time']:.3f}")
    print(f"  Decoding speed      : {s['speed']:.2f} tokens/s")
    print(f"  Time / question     : {s['time_per_q']:.3f} s")
    if "block_efficiency" in s:
        print(f"  Block efficiency    : {s['block_efficiency']:.3f}  (ratio: {s.get('be_ratio',0):.3f})")
        print(f"  Avg accepted/step   : {s['avg_accepted_per_step']:.3f}")
    if "pass1" in s:
        print(f"  Pass@1              : {s['pass1']:.1%}  ({s['correct']}/{s['graded']})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Path(s) to JSONL results file(s)")
    parser.add_argument("--question-file", default='eagle/data/gsm8k/question.jsonl', help="Path to question.jsonl for pass@1 scoring")
    args = parser.parse_args()

    refs = load_references(args.question_file) if args.question_file else None
    expected_n = len(refs) if refs is not None else None

    results = [(path, analyze(path, refs)) for path in args.files]
    for path, s in results:
        print_stats(path, s, expected_n=expected_n)

    if len(results) > 1:
        has_pass1 = "pass1" in results[0][1]
        has_block = "block_efficiency" in results[0][1]
        print("\n--- Summary ---")
        header = f"{'File':<60} {'Speed (tok/s)':>14} {'Time/Q (s)':>12}"
        if has_block:
            header += f" {'BlkEff':>8} {'Acc/step':>9}"
        if has_pass1:
            header += f" {'Pass@1':>8}"
        print(header)
        print("-" * len(header))
        for path, s in results:
            row = f"{path:<60} {s['speed']:>14.2f} {s['time_per_q']:>12.3f}"
            if has_block:
                row += f" {s['block_efficiency']:>8.3f} {s.get('be_ratio',0):>8.3f} {s['avg_accepted_per_step']:>9.3f}"
            if has_pass1:
                row += f" {s['pass1']:>8.1%}"
            if expected_n and s['questions'] < expected_n:
                row += f"  ⚠ {s['questions']}/{expected_n} questions"
            print(row)

if __name__ == "__main__":
    main()
