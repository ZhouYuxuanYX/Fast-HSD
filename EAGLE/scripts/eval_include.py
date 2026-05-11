"""Evaluate EAGLE output JSONL files on INCLUDE pass@1 accuracy.

Reads references from eagle/data/include/question.jsonl (correct letter A-D),
extracts predicted answer from model outputs, and reports per-language and
overall accuracy alongside decoding speed / block efficiency.

Usage:
    python scripts/eval_include.py include/*.jsonl
    python scripts/eval_include.py include/run1.jsonl include/run2.jsonl
"""
import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# Answer extraction (mirrors run_include_qwen2.5_simple.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str):
    # 1) "answer is (X)" or "answer is X"
    m = re.search(r"answer is \(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) "Answer: X"
    m = re.search(r"[aA]nswer:\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1).upper()
    # 3) last standalone A-D in the text
    m = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text.upper(), re.DOTALL)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_references(question_file: str):
    """Returns {question_id: {"answer": letter, "language": lang}}."""
    refs = {}
    with open(question_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            refs[rec["question_id"]] = {
                "answer": rec["reference"][0] if isinstance(rec["reference"], list) else rec["reference"],
                "language": rec.get("language", "Unknown"),
            }
    return refs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(path: str, refs: dict):
    total_tokens = 0
    total_time = 0.0
    question_times = []
    total_accepted = 0
    total_block_slots = 0
    total_max_slots = 0
    total_steps = 0
    correct = 0
    graded = 0
    lang_correct: dict = {}
    lang_total: dict = {}

    with open(path) as f:
        for line in f:
            record = json.loads(line)
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
            if qid in refs:
                ref = refs[qid]
                lang = ref["language"]
                pred = extract_answer(turns["turns"][-1])
                lang_total[lang] = lang_total.get(lang, 0) + 1
                graded += 1
                if pred == ref["answer"]:
                    correct += 1
                    lang_correct[lang] = lang_correct.get(lang, 0) + 1

    n = len(question_times)
    result = {
        "questions": n,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "speed": total_tokens / total_time if total_time > 0 else 0.0,
        "time_per_q": total_time / n if n > 0 else 0.0,
        "pass1": correct / graded if graded > 0 else float("nan"),
        "correct": correct,
        "graded": graded,
        "lang_correct": lang_correct,
        "lang_total": lang_total,
    }
    if total_block_slots > 0:
        result["block_efficiency"] = total_accepted / total_block_slots
        result["be_ratio"] = total_accepted / total_max_slots if total_max_slots > 0 else 0.0
        result["avg_accepted_per_step"] = total_accepted / total_steps
    return result


def print_stats(path: str, s: dict, expected_n: int = None, show_lang: bool = False):
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
    print(f"  Pass@1 (accuracy)   : {s['pass1']:.1%}  ({s['correct']}/{s['graded']})")

    if show_lang and s["lang_total"]:
        print("  Per-language accuracy:")
        for lang in sorted(s["lang_total"]):
            c = s["lang_correct"].get(lang, 0)
            t = s["lang_total"][lang]
            print(f"    {lang:<25} {c}/{t}  ({c/t:.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="EAGLE output JSONL file(s)")
    parser.add_argument(
        "--question-file",
        default="eagle/data/include/question.jsonl",
        help="Path to include question.jsonl",
    )
    parser.add_argument(
        "--show-lang", action="store_true",
        help="Print per-language accuracy breakdown",
    )
    args = parser.parse_args()

    refs = load_references(args.question_file)
    expected_n = len(refs)
    print(f"Loaded {expected_n} references from {args.question_file}\n")

    results = []
    for path in args.files:
        s = analyze(path, refs)
        results.append((path, s))
        print_stats(path, s, expected_n=expected_n, show_lang=args.show_lang)

    if len(results) > 1:
        has_block = "block_efficiency" in results[0][1]
        print("\n--- Summary ---")
        header = f"{'File':<60} {'Speed (tok/s)':>14} {'Time/Q (s)':>12}"
        if has_block:
            header += f" {'BlkEff':>8} {'Acc/step':>9}"
        header += f" {'Pass@1':>8}"
        print(header)
        print("-" * len(header))
        for path, s in results:
            row = f"{path:<60} {s['speed']:>14.2f} {s['time_per_q']:>12.3f}"
            if has_block:
                row += f" {s['block_efficiency']:>8.3f} {s.get('be_ratio',0):>8.3f} {s['avg_accepted_per_step']:>9.3f}"
            row += f" {s['pass1']:>8.1%}"
            if expected_n and s["questions"] < expected_n:
                row += f"  ⚠ {s['questions']}/{expected_n}"
            print(row)


if __name__ == "__main__":
    main()
