"""Prepare competition_math question.jsonl from qwedsacf/competition_math (first 500 train samples).

Usage:
    python scripts/prepare_competition_math.py
"""
import json
import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eagle", "data", "competition_math")
OUT_FILE = os.path.join(OUT_DIR, "question.jsonl")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("qwedsacf/competition_math", split="train")
    samples = ds.select(range(500))

    with open(OUT_FILE, "w") as f:
        for i, sample in enumerate(samples):
            entry = {
                "question_id": i,
                "category": sample["type"],
                "turns": [sample["problem"]],
                "reference": [sample["solution"]],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(samples)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
