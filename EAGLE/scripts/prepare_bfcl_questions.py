"""Prepare BFCL question.jsonl for EAGLE generation.

Downloads all four BFCL categories from HuggingFace and writes
eagle/data/bfcl/question.jsonl in the same format used by include/question.jsonl.

Each line:
  {
    "question_id": <int, global 0-based>,
    "bfcl_id":     <str, e.g. "BFCL_v3_simple_0">,
    "category":    <str, "simple"|"multiple"|"parallel"|"parallel_multiple">,
    "turns":       [<str, full prompt ready to be given as user message>],
    "reference":   [<dict, {func_name: {param: [acceptable_values]}}>, ...]
  }

Usage (from EAGLE/ directory):
    python scripts/prepare_bfcl_questions.py
    python scripts/prepare_bfcl_questions.py --data-dir /tmp/bfcl_cache
"""

import argparse
import json
import os
from huggingface_hub import hf_hub_download

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

CATEGORY_FILES = {
    "simple":            ("BFCL_v3_simple.json",           "possible_answer/BFCL_v3_simple.json"),
    "multiple":          ("BFCL_v3_multiple.json",          "possible_answer/BFCL_v3_multiple.json"),
    "parallel":          ("BFCL_v3_parallel.json",          "possible_answer/BFCL_v3_parallel.json"),
    "parallel_multiple": ("BFCL_v3_parallel_multiple.json", "possible_answer/BFCL_v3_parallel_multiple.json"),
}

CATEGORY_ORDER = ["parallel_multiple"]

# Max samples per category (None = all)
CATEGORY_LIMITS = {
    "parallel_multiple": None,
}

PROMPT_TEMPLATE = """\
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:
{functions}

Question: {user_message}"""


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=None,
        help="Local directory to cache raw BFCL files (default: ~/.cache/huggingface/bfcl)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for question.jsonl (default: eagle/data/bfcl/ relative to EAGLE root)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eagle_dir  = os.path.dirname(script_dir)

    data_dir = args.data_dir or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "bfcl"
    )
    out_dir = args.out_dir or os.path.join(eagle_dir, "eagle", "data", "bfcl")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "question.jsonl")
    print(f"Downloading BFCL data to: {data_dir}")
    print(f"Writing question.jsonl to: {out_path}\n")

    question_id = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for category in CATEGORY_ORDER:
            prompt_file, answer_file = CATEGORY_FILES[category]
            print(f"  Processing category: {category}")

            prompt_path = hf_hub_download(
                BFCL_REPO, prompt_file, repo_type="dataset", local_dir=data_dir
            )
            answer_path = hf_hub_download(
                BFCL_REPO, answer_file, repo_type="dataset", local_dir=data_dir
            )

            prompts = load_jsonl(prompt_path)
            answers = load_jsonl(answer_path)
            answer_map = {a["id"]: a for a in answers}

            limit = CATEGORY_LIMITS.get(category)
            for sample in (prompts[:limit] if limit is not None else prompts):
                bfcl_id = sample["id"]

                # Build prompt: embed function schema + user question in turns[0]
                functions_str = json.dumps(sample["function"], indent=2)
                user_messages = sample["question"][0]  # first (only) turn
                user_content  = user_messages[-1]["content"]
                prompt = PROMPT_TEMPLATE.format(
                    functions=functions_str,
                    user_message=user_content,
                )

                # Ground truth: list of {func_name: {param: [values]}}
                if bfcl_id in answer_map:
                    reference = answer_map[bfcl_id].get("ground_truth", [])
                else:
                    reference = []

                record = {
                    "question_id": question_id,
                    "bfcl_id":     bfcl_id,
                    "category":    category,
                    "turns":       [prompt],
                    "reference":   reference,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                question_id += 1

            print(f"    {len(prompts)} questions written (cumulative: {question_id})")

    print(f"\nDone. Total questions: {question_id}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
