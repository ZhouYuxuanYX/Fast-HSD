import argparse
import json
import re

def extract_answer(text):
    """Extract final answer: prefer #### marker, fall back to last number."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None

def load_references(question_file):
    """Return dict of question_id -> reference answer string."""
    refs = {}
    with open(question_file) as f:
        for line in f:
            record = json.loads(line)
            qid = record["question_id"]
            # reference is a list; last element has the #### answer
            ref_text = record["reference"][-1] if isinstance(record["reference"], list) else record["reference"]
            refs[qid] = extract_answer(ref_text)
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
            turns = record["choices"][0]
            total_tokens += sum(turns["new_tokens"])
            q_time = sum(turns["wall_time"])
            total_time += q_time
            question_times.append(q_time)

            # Block efficiency from per-step stats: list of [accepted_draft_tokens, block_size]
            if "step_stats" in turns:
                for accepted, block_size in turns["step_stats"]:
                    total_accepted += accepted
                    total_block_slots += block_size
                    total_max_slots += (block_size - 1)
                    total_steps += 1

            if refs is not None:
                qid = record["question_id"]
                pred = extract_answer(turns["turns"][-1])
                ref = refs.get(qid)
                if pred is not None and ref is not None:
                    graded += 1
                    if pred == ref:
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
