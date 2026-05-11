"""Prepare INCLUDE question.jsonl from CohereLabs/include-base-44.

Samples `--samples-per-language` questions (default 5) from each of the 44
languages (pooling validation + test), formats them as multiple-choice prompts,
and writes to eagle/data/include/question.jsonl.

Usage:
    python scripts/prepare_include.py
    python scripts/prepare_include.py --samples-per-language 10 --seed 42
"""
import argparse
import json
import os
import random

from datasets import load_dataset

LANGUAGES = [
    "Albanian", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian",
    "Bengali", "Bulgarian", "Chinese", "Croatian", "Dutch", "Estonian",
    "Finnish", "French", "Georgian", "German", "Greek", "Hebrew", "Hindi",
    "Hungarian", "Indonesian", "Italian", "Japanese", "Kazakh", "Korean",
    "Lithuanian", "Malay", "Malayalam", "Nepali", "North Macedonian", "Persian",
    "Polish", "Portuguese", "Russian", "Serbian", "Spanish", "Tagalog",
    "Tamil", "Telugu", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese",
]

CHOICES = ["A", "B", "C", "D"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eagle", "data", "include")
OUT_FILE = os.path.join(OUT_DIR, "question.jsonl")


def format_mcq(example: dict) -> str:
    opts = [
        example.get("option_a", ""),
        example.get("option_b", ""),
        example.get("option_c", ""),
        example.get("option_d", ""),
    ]
    prompt = "Question:\n" + example["question"] + "\nOptions:\n"
    for i, opt in enumerate(opts):
        if opt and opt.strip():
            prompt += f"{CHOICES[i]}. {opt}\n"
    prompt += "Answer: Let's think step by step."
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-language", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    entries = []
    for lang in LANGUAGES:
        pool = []
        for split in ("validation", "test"):
            try:
                ds = load_dataset("CohereLabs/include-base-44", lang, split=split)
                pool.extend(list(ds))
            except Exception as e:
                print(f"  {lang}/{split}: skipped ({e})")

        if not pool:
            print(f"  {lang}: no data found, skipping")
            continue

        sampled = random.sample(pool, min(args.samples_per_language, len(pool)))
        print(f"  {lang}: {len(sampled)}/{len(pool)}")

        for ex in sampled:
            answer_idx = int(ex["answer"])
            entries.append({
                "question_id": len(entries),
                "category": ex.get("subject", "unknown"),
                "language": ex.get("language", lang),
                "turns": [format_mcq(ex)],
                "reference": [CHOICES[answer_idx]],
            })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(entries)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
