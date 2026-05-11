"""Prepare mbppplus question.jsonl from evalplus/mbppplus (all 378 test samples).

Usage:
    python scripts/prepare_mbppplus.py
"""
import json
import os
import re

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eagle", "data", "mbppplus")
OUT_FILE = os.path.join(OUT_DIR, "question.jsonl")


_BUILTIN_NAMES = {
    "set", "list", "tuple", "sorted", "len", "str", "int", "float", "bool",
    "dict", "frozenset", "sum", "max", "min", "round", "abs", "type",
    "range", "enumerate", "zip", "map", "filter", "any", "all", "print",
    "isinstance", "repr", "hash",
}


def extract_func_name(test_list: list):
    """Extract expected function name from the first test assertion,
    skipping common Python built-in wrappers like set(), list(), etc."""
    if not test_list:
        return None
    test = test_list[0]
    # Find the 'assert' keyword, then collect all word(... patterns after it
    assert_m = re.search(r'\bassert\b', test)
    if not assert_m:
        return None
    after_assert = test[assert_m.end():]
    for m in re.finditer(r'\b(\w+)\s*\(', after_assert):
        name = m.group(1)
        if name not in _BUILTIN_NAMES:
            return name
    return None


def build_prompt(prompt: str, test_list: list) -> str:
    func_name = extract_func_name(test_list)
    if func_name:
        return f"{prompt}\n\nThe function must be named `{func_name}`."
    return prompt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("evalplus/mbppplus", split="test")

    with open(OUT_FILE, "w") as f:
        for i, sample in enumerate(ds):
            entry = {
                "question_id": i,
                "category": "coding",
                "turns": [build_prompt(sample["prompt"], sample["test_list"])],
                "reference": [sample["code"]],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(ds)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
