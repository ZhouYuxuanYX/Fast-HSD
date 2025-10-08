import fnmatch
import json
import warnings

import datasets
import torch
import transformers
import argparse
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--task_name",
        default="cnndailymail",
        help="Task to evaluate on, can be a single task",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Whether use additional bertscore for xsum/cnndailymail.",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        help="Whether use additional comet for wmt.",
    )
    parser.add_argument(
        "--postprocessed_output_path",
        type=str,
        default=None,
        help="Path to save the postprocessed generations",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Whether sft or not.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Selected Tasks: {args.task_name}")

    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        print("evaluation only mode")
        evaluator = Evaluator(args)
        results["results"] = evaluator.evaluate(args.task_name)
    
    # Save all args to config
    for k,v in vars(args).items():
        results[k] = v
    # Save jsonl
    dumped = json.dumps(results)
    print(dumped)
    with open(args.metric_output_path, "a+") as f:
        f.write(dumped+"\n")

'''
python3 ev.py\
        --task_name human_eval\
        --load_generations_path /u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp3/humaneval/Qwen_72B_SPD_lenience_02_test1_truncated.jsonl\
        --metric_output_path Qwen_72B_SPD_lenience_02_test1_human_eval_results.jsonl\
        --allow_code_execution
'''

'''
python3 ev.py\
        --task_name cnndailymail\
        --load_generations_path /u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp3/cnndailymail/Qwen_72B_SPD_lenience_02_test1.jsonl\
        --metric_output_path Qwen_72B_SPD_lenience_02_test1_cnndailymail_results.jsonl\
        --allow_code_execution
'''

'''
# Create a deduplicated version
python3 -c "
import json
seen = set()
with open('/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp3/humaneval/Qwen_72B_SPD_lenience_02_test1.jsonl', 'r') as f:
    with open('/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp3/humaneval/Qwen_72B_SPD_lenience_02_test1_new.jsonl', 'w') as out:
        for line in f:
            obj = json.loads(line)
            if obj['idx'] not in seen:
                seen.add(obj['idx'])
                out.write(line)
"
'''