#!/usr/bin/env python
"""End-of-run math evaluator — standalone CLI wrapper.

The analysis logic (load_references, analyze, print_stats, schema
auto-detection) and the scoring (extract_answer, extract_ref_answer,
is_equiv) live in :mod:`fast_hsd.benchmarks._math_report` and
:mod:`fast_hsd.benchmarks._math_scoring`. ``fast-hsd-eval --benchmark
math`` calls the same functions at the end of an inference run, so the
report printed here is identical to the one shown inline at end of
inference — there's no need to run this script separately. It's kept
for ad-hoc re-analysis of existing JSONL files (e.g. the legacy
``gen_ea_answer_*.py`` output).

Examples
--------

    # New refactor output (rows.jsonl) — schema auto-detected.
    python scripts/results_analysis_math.py \\
        outputs/math/eagle_math_min_p_sampling/rows.jsonl \\
        --question-file EAGLE/eagle/data/competition_math/question.jsonl

    # Legacy gen_ea_answer_*.py output — same script, same numbers.
    python scripts/results_analysis_math.py \\
        legacy_min_p_sampling.jsonl \\
        --question-file EAGLE/eagle/data/competition_math/question.jsonl

    # Mix both schemas in one invocation; you get one summary table.
    python scripts/results_analysis_math.py legacy_*.jsonl outputs/math/*/rows.jsonl \\
        --question-file EAGLE/eagle/data/competition_math/question.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo's package importable even when fast-hsd isn't pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fast_hsd.benchmarks._math_report import analyze, load_references, print_stats  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Path(s) to JSONL results file(s)")
    parser.add_argument(
        "--question-file",
        default="eagle/data/gsm8k/question.jsonl",
        help="Path to question.jsonl for pass@1 scoring (default: gsm8k)",
    )
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
            header += f" {'BlkEff':>8} {'BERatio':>8} {'Acc/step':>9}"
        if has_pass1:
            header += f" {'Pass@1':>8}"
        print(header)
        print("-" * len(header))
        for path, s in results:
            row = f"{path:<60} {s['speed']:>14.2f} {s['time_per_q']:>12.3f}"
            if has_block:
                row += (
                    f" {s['block_efficiency']:>8.3f} {s.get('be_ratio', 0):>8.3f}"
                    f" {s['avg_accepted_per_step']:>9.3f}"
                )
            if has_pass1:
                row += f" {s['pass1']:>8.1%}"
            if expected_n and s["questions"] < expected_n:
                row += f"  {s['questions']}/{expected_n} questions"
            print(row)


if __name__ == "__main__":
    main()
