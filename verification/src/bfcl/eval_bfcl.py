"""
BFCL (Berkeley Function Calling Leaderboard) evaluation script for speculative decoding.

Evaluates LLM function-calling accuracy under various SPD methods.
Mirrors the interface of eval_mbppplus.py and eval_math.py.

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard (HuggingFace)
Categories: simple (400), multiple (200), parallel (200), parallel_multiple (200)
Evaluation: AST-based comparison of function calls against ground truth.
"""

import re
import os
import ast
import json
import time
import random
import argparse
import mmap
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
import orjson

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model
from huggingface_hub import hf_hub_download


# ======================== BFCL Constants ========================

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Maps category name -> (prompt file, answer file)
CATEGORY_FILES = {
    "simple":            ("BFCL_v3_simple.json",            "possible_answer/BFCL_v3_simple.json"),
    "multiple":          ("BFCL_v3_multiple.json",           "possible_answer/BFCL_v3_multiple.json"),
    "parallel":          ("BFCL_v3_parallel.json",           "possible_answer/BFCL_v3_parallel.json"),
    "parallel_multiple": ("BFCL_v3_parallel_multiple.json",  "possible_answer/BFCL_v3_parallel_multiple.json"),
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.
{functions}"""


# ======================== Arg Parsing ========================

def argparse_setup():
    parser = argparse.ArgumentParser(prog='eval_bfcl')

    # SPD method args (matching eval_mbppplus.py)
    parser.add_argument('--backward', action='store_true', default=False)
    parser.add_argument('--clever', action='store_true', default=False)
    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--parallel_spd', action='store_true', default=False)
    parser.add_argument('--gamma', default=10, type=int, help='number of assisted tokens')
    parser.add_argument('--lenience', default=1, type=float, help='lenience factor')
    parser.add_argument('--fast', action='store_true', default=False)
    parser.add_argument('--cascade', action='store_true', default=False)
    parser.add_argument('--uniform-draft', action='store_true', default=False)

    # Sampling args
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=20)

    # Model args
    parser.add_argument('--target-model', default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8')
    parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8')
    parser.add_argument('--model', help='must be target or draft', default="target")

    # BFCL-specific args
    parser.add_argument('--category', type=str, default='parallel_multiple',
                        choices=['simple', 'multiple', 'parallel', 'parallel_multiple'],
                        help='BFCL category to evaluate')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Local directory to cache BFCL data (default: $HF_HOME/bfcl or ./bfcl_data)')

    # General args
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='run name')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eta_cutoff', type=float, default=None)
    parser.add_argument('--min_p', type=float, default=None)
    parser.add_argument('--eta_spd', type=float, default=None)
    parser.add_argument('--min_p_spd', type=float, default=None)
    parser.add_argument('--cos_lambda', type=float, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=199)
    parser.add_argument('--multidraft', type=int, default=1)

    args = parser.parse_args()
    print(args)
    return args


# ======================== BFCL Data Loading ========================

def download_bfcl_data(category: str, data_dir: str) -> Tuple[str, str]:
    """Download BFCL prompt and answer files from HuggingFace."""
    prompt_file, answer_file = CATEGORY_FILES[category]
    os.makedirs(data_dir, exist_ok=True)

    prompt_path = hf_hub_download(
        BFCL_REPO, prompt_file, repo_type="dataset", local_dir=data_dir
    )
    answer_path = hf_hub_download(
        BFCL_REPO, answer_file, repo_type="dataset", local_dir=data_dir
    )
    return prompt_path, answer_path


def load_bfcl_data(prompt_path: str, answer_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load BFCL JSONL data."""
    prompts = []
    with open(prompt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    answers = []
    with open(answer_path) as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))

    # Build answer lookup by id
    answer_map = {a["id"]: a for a in answers}
    return prompts, answer_map


# ======================== BFCL Prompt Formatting ========================

def format_bfcl_prompt(sample: Dict) -> str:
    """Build the user-facing prompt for a BFCL sample."""
    functions = sample["function"]
    functions_str = json.dumps(functions, indent=4)
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(functions=functions_str)

    # question is list[list[dict]]; for single-turn, take first turn's messages
    user_messages = sample["question"][0]
    user_content = user_messages[-1]["content"]  # last message in the turn

    return system_msg, user_content


# ======================== BFCL Output Parsing ========================

def parse_function_calls(model_output: str) -> List[Dict[str, Dict]]:
    """
    Parse model output into structured function calls.
    Expected format: [func_name(param1=val1, param2=val2), ...]

    Returns list of {func_name: {param1: val1, param2: val2}}
    """
    model_output = model_output.strip()

    # Strip markdown code blocks if present
    if model_output.startswith("```"):
        lines = model_output.split("\n")
        # Remove first and last ``` lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        model_output = "\n".join(lines).strip()

    # Ensure wrapped in brackets
    if not model_output.startswith("["):
        model_output = "[" + model_output
    if not model_output.endswith("]"):
        model_output = model_output + "]"

    try:
        tree = ast.parse(model_output, mode='eval')
    except SyntaxError:
        return []

    if not isinstance(tree.body, (ast.List, ast.Tuple)):
        # Single call not wrapped
        if isinstance(tree.body, ast.Call):
            calls = [tree.body]
        else:
            return []
    else:
        calls = tree.body.elts

    results = []
    for node in calls:
        if not isinstance(node, ast.Call):
            continue
        func_call = _resolve_ast_call(node)
        if func_call:
            results.append(func_call)
    return results


def _resolve_ast_call(node: ast.Call) -> Optional[Dict]:
    """Convert an ast.Call node to {func_name: {param: value}}."""
    # Get function name
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        # e.g., module.func_name
        parts = []
        n = node.func
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        func_name = ".".join(reversed(parts))
    else:
        return None

    # Get keyword arguments
    params = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        params[kw.arg] = _ast_literal(kw.value)

    # Handle positional args (less common but possible)
    for i, arg in enumerate(node.args):
        params[f"_pos_{i}"] = _ast_literal(arg)

    return {func_name: params}


def _ast_unparse(node):
    """Backport of ast.unparse for Python < 3.9."""
    if hasattr(ast, 'unparse'):
        return ast.unparse(node)
    # Fallback: use compile + eval for literals, repr for others
    try:
        return repr(ast.literal_eval(compile(ast.Expression(body=node), '<>', 'eval')))
    except Exception:
        pass
    # Manual unparse for common node types
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Num):  # Python 3.7 compat
        return repr(node.n)
    if isinstance(node, ast.Str):
        return repr(node.s)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return "[" + ", ".join(_ast_unparse(e) for e in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_ast_unparse(e) for e in node.elts) + ")"
    if isinstance(node, ast.Dict):
        pairs = []
        for k, v in zip(node.keys, node.values):
            pairs.append(f"{_ast_unparse(k)}: {_ast_unparse(v)}")
        return "{" + ", ".join(pairs) + "}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_ast_unparse(node.operand)}"
    if isinstance(node, ast.NameConstant):  # Python 3.7 compat
        return repr(node.value)
    return repr(node)


def _ast_literal(node):
    """Safely evaluate an AST node to a Python literal."""
    try:
        code = compile(ast.Expression(body=node), '<>', 'eval')
        return eval(code)
    except Exception:
        return _ast_unparse(node)


# ======================== BFCL Evaluation ========================

def check_simple(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    """
    Check a simple/multiple category sample.
    Expects exactly 1 function call matching the ground truth.
    """
    if len(model_calls) == 0 or len(ground_truth) == 0:
        return False

    gt = ground_truth[0]  # single expected call
    gt_func_name = list(gt.keys())[0]
    gt_params = gt[gt_func_name]

    # Find matching call
    for call in model_calls:
        call_func_name = list(call.keys())[0]
        if call_func_name != gt_func_name:
            continue
        call_params = call[call_func_name]
        if _params_match(call_params, gt_params):
            return True

    return False


def check_parallel(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    """
    Check parallel/parallel_multiple category sample.
    All ground truth calls must be present (order doesn't matter).
    """
    if len(model_calls) < len(ground_truth):
        return False

    gt_matched = [False] * len(ground_truth)
    for gt_idx, gt_call in enumerate(ground_truth):
        gt_func_name = list(gt_call.keys())[0]
        gt_params = gt_call[gt_func_name]
        for model_call in model_calls:
            model_func_name = list(model_call.keys())[0]
            if model_func_name != gt_func_name:
                continue
            if _params_match(model_call[model_func_name], gt_params):
                gt_matched[gt_idx] = True
                break

    return all(gt_matched)


def check_irrelevance(model_calls: List[Dict], ground_truth: List[Dict]) -> bool:
    """
    Check irrelevance category: model should NOT make any function call.
    """
    return len(model_calls) == 0


def _params_match(model_params: Dict, gt_params: Dict) -> bool:
    """
    Check if model parameters match ground truth.
    GT values are lists of acceptable values. Empty string means optional.
    """
    # Check all required GT params are present
    for param_name, acceptable_values in gt_params.items():
        if not isinstance(acceptable_values, list):
            acceptable_values = [acceptable_values]

        # Check if param is optional (empty string in acceptable values)
        is_optional = "" in acceptable_values
        acceptable_values_clean = [v for v in acceptable_values if v != ""]

        if param_name not in model_params:
            if is_optional:
                continue  # OK, param is optional
            else:
                return False  # Required param missing

        model_value = model_params[param_name]
        if not _value_matches(model_value, acceptable_values_clean):
            return False

    return True


def _value_matches(model_val, acceptable_vals: list) -> bool:
    """Check if model value matches any of the acceptable values."""
    for acc_val in acceptable_vals:
        if _values_equal(model_val, acc_val):
            return True
    return False


def _values_equal(a, b) -> bool:
    """Compare two values with type-flexible matching."""
    # Direct equality
    if a == b:
        return True

    # String comparison (case-insensitive, strip whitespace)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()

    # Numeric comparison with tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False

    # String to number
    if isinstance(a, str) and isinstance(b, (int, float)):
        try:
            return _values_equal(float(a) if '.' in a else int(a), b)
        except (ValueError, TypeError):
            return False
    if isinstance(b, str) and isinstance(a, (int, float)):
        return _values_equal(b, a)

    # List comparison
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))

    # Dict comparison
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)

    return False


# Category -> checker function
CATEGORY_CHECKERS = {
    "simple": check_simple,
    "multiple": check_simple,  # same logic: pick 1 correct function from many
    "parallel": check_parallel,
    "parallel_multiple": check_parallel,
    "irrelevance": check_irrelevance,
}


# ======================== Utilities (from eval_mbppplus.py) ========================

def move_rotary_emb_to_device(model):
    try:
        device = model.model.embed_tokens.weight.device
        if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
            model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    except AttributeError:
        print("Could not move rotary_emb.inv_freq")


def manual_device_map(model, same_device_for_input_output=True):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        prefix = 'transformer.h'
        embedding_key = 'transformer.wte'
        norm_key = 'transformer.ln_f'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        prefix = 'model.layers'
        embedding_key = 'model.embed_tokens'
        norm_key = 'model.norm'
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)
    layers_per_gpu = (num_layers + n_gpus - 1) // n_gpus
    device_map = {}

    for i in range(num_layers):
        gpu_id = i // layers_per_gpu
        device_map[f"{prefix}.{i}"] = f"cuda:{gpu_id}"

    if same_device_for_input_output:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = 'cuda:0'
    else:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = f"cuda:{n_gpus - 1}"

    device_map[norm_key] = f"cuda:{n_gpus - 1}"
    return device_map


def effective_bandwidth_Bps():
    if not torch.cuda.is_available():
        return float("nan")
    x = torch.empty((1024, 1024, 4000), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = x.clone()
    torch.cuda.synchronize()
    t1 = time.time()
    return x.numel() * x.element_size() / (t1 - t0)


# ======================== Main Evaluation Class ========================

class HSD_BFCL:
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = self.args.draft_model
        print(f'target model: {self.target_model_name}')
        print(f'draft model: {self.draft_model_name}')
        self.model_size = self.target_model_name.split("/")[1].split("-")[1]
        print(f'model size: {self.model_size}')

        if float(self.model_size[:-1]) > 3:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(f'using device: {self.device}')

        # Set random seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        # =================== Load BFCL data ===================
        self.dataset_name = f'bfcl_{self.args.category}'
        # Resolve data directory
        if self.args.data_dir is None:
            hf_home = os.environ.get("HF_HOME", None)
            if hf_home:
                self.args.data_dir = os.path.join(hf_home, "bfcl")
            else:
                self.args.data_dir = "./bfcl_data"

        print(f"Downloading BFCL data for category: {self.args.category}")
        print(f"Data directory: {self.args.data_dir}")
        prompt_path, answer_path = download_bfcl_data(self.args.category, self.args.data_dir)
        self.prompts, self.answer_map = load_bfcl_data(prompt_path, answer_path)

        total_available = len(self.prompts)
        self.start_index = self.args.start_index
        self.end_index = self.args.end_index if self.args.end_index is not None else total_available
        self.end_index = min(self.end_index, total_available)
        self.num_samples = self.end_index - self.start_index

        print(f"Total available samples: {total_available}")
        print(f"Evaluating samples [{self.start_index}, {self.end_index}) = {self.num_samples} samples")

        # =================== Model setup ===================
        self.model_setup()
        self.sd = self.test_setup()

        self.final_result = {'Block Efficiency': None, 'Decoding Speed': None}

    def model_setup(self):
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_name,
            device_map={"": self.device} if int(self.model_size[:-1]) < 32 else None
        )
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            device_map={"": self.device} if int(self.model_size[:-1]) < 32 else None
        )

        # Generation config
        for m in [self.draft_model, self.target_model]:
            m.generation_config.num_assistant_tokens = self.args.gamma
            m.generation_config.assistant_confidence_threshold = 0
            m.generation_config.temperature = self.args.temperature
            m.generation_config.top_k = self.args.top_k
            m.generation_config.top_p = self.args.top_p

        # Vocab alignment
        vocab_size = min(self.draft_model.config.vocab_size, self.target_model.config.vocab_size)
        self.draft_model.config.vocab_size = vocab_size
        self.target_model.config.vocab_size = vocab_size
        self.same_tokenizer = (
            self.target_model.config.get_text_config().vocab_size
            == self.draft_model.config.get_text_config().vocab_size
        )

        # Resize lm_head if needed
        for model in [self.draft_model, self.target_model]:
            if hasattr(model, "lm_head"):
                old_lm_head = model.lm_head
                if old_lm_head.out_features != vocab_size:
                    dtype = old_lm_head.weight.dtype
                    new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(
                        old_lm_head.weight.device, dtype=dtype
                    )
                    with torch.no_grad():
                        n = min(old_lm_head.out_features, vocab_size)
                        new_lm_head.weight[:n] = old_lm_head.weight[:n]
                    model.lm_head = new_lm_head

        # Multi-GPU dispatch
        if torch.cuda.is_available() and int(self.model_size[:-1]) > 14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2)

        print("dispatch model finished")
        self.draft_model.eval()
        self.target_model.eval()

        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

        if self.args.uniform_draft:
            self.make_draft_model_uniform()

        self.tokenizer1 = AutoTokenizer.from_pretrained(self.draft_model_name)
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.target_model_name)

    def test_setup(self):
        sd = f"Qwen_{self.model_size}_0.5B_"
        if self.args.speculative:
            if self.args.blockwise:
                sd += "blockwise"
            else:
                sd += "backward" if self.args.backward else "tokenwise"
            if self.args.naive:
                sd += "_naive"
            elif self.args.clever:
                sd += "_clever"
            elif self.args.fast:
                sd += "_fast"
        else:
            sd += self.args.model
        sd += f"_gamma_{self.args.gamma}"

        if self.args.multidraft > 1:
            sd += f"_multidraft_{self.args.multidraft}"
        if self.args.parallel_spd:
            sd += "_parallel"
        if self.args.temperature:
            sd += f"_t{self.args.temperature}"
        if self.args.top_p:
            sd += f"_topp_{self.args.top_p}"
        if self.args.lenience < 1:
            sd += f"_lenience_{self.args.lenience}"
        if self.args.eta_cutoff:
            sd += f"_eta{self.args.eta_cutoff}"
        if self.args.min_p:
            sd += f"_minp_{self.args.min_p}"
        if self.args.cascade:
            sd += "_cascade"
        if self.args.uniform_draft:
            sd += "_uniform"
        if self.args.eta_spd:
            sd += f"_eta{self.args.eta_spd}"
        if self.args.min_p_spd:
            sd += f"_minp_{self.args.min_p_spd}"
        if self.args.cos_lambda:
            sd += f"_coslambda_{self.args.cos_lambda}"

        sd += f"_bfcl_{self.args.category}"
        sd += f'{self.args.name}'
        return sd

    def speculative_decoding(self, input_ids):
        outputs, counts = self.target_model.generate(
            input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
            assistant_model=self.draft_model,
            assistant_confidence_threshold=0,
            backward=self.args.backward,
            assistant_tokenizer=self.tokenizer1 if not self.same_tokenizer else None,
            tokenizer=self.tokenizer1,
            return_probs=self.args.backward or self.args.blockwise,
            blockwise=self.args.blockwise,
            clever=self.args.clever,
            fast=self.args.fast,
            naive=self.args.naive,
            lenience=self.args.lenience,
            cascade=self.args.cascade,
            eta_cutoff=self.args.eta_cutoff,
            min_p=self.args.min_p,
            min_p_spd=self.args.min_p_spd,
            eta_spd=self.args.eta_spd,
            cos_lambda=self.args.cos_lambda,
        )

        self.total_counts["draft_eval"].append(counts["draft_eval"])
        self.total_counts["sample_length"].append(counts["sample_length"])
        self.total_counts["target_eval"].append(counts["target_eval"])
        self.total_counts["p_i"].append(counts["p_i"])
        self.total_counts["q_i"].append(counts["q_i"])
        self.total_counts["step_back_probs"].append(counts["step_back_probs"])
        self.total_counts["total_step"].append(counts["total_step"])
        self.total_counts["ids"].append(counts["ids"])
        self.total_counts["draft_raw_logits"].append(counts.get("draft_raw_logits", []))
        self.total_counts["target_raw_logits"].append(counts.get("target_raw_logits", []))

        return outputs

    def efficiency_analysis(self, file_path):
        gamma = self.args.gamma
        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            count = orjson.loads(mm[:])
            mm.close()

        if not count.get("draft_eval") or len(count["draft_eval"]) == 0:
            total_time = sum(float(t) for t in count.get("time", []))
            print("Non-speculative mode: Block Efficiency not applicable")
            print(f"Total time: {total_time:.2f}s")
            self.final_result['Block Efficiency'] = "N/A"
            self.final_result['Decoding Speed'] = "N/A"
            self.final_result['Total Time (s)'] = f"{total_time:.2f}"
            return

        draft = 0; target = 0; step = 0; sample = 0; time_ = 0; len_ = 0

        for n in range(len(count["draft_eval"])):
            draft_list = np.array(count["draft_eval"][n])
            target_list = np.array(count["target_eval"][n])
            step_list = np.array(count["total_step"][n])
            sample_list = np.array(count["sample_length"][n])

            draft += draft_list[draft_list == gamma].sum()
            target += target_list[draft_list == gamma].sum()
            step += step_list[draft_list == gamma].sum()
            sample += sample_list[draft_list == gamma].sum()
            time_ += float(count["time"][n])
            len_ += len(sample_list[draft_list == gamma])

        if len_ == 0:
            print("Warning: No valid samples found for efficiency analysis")
            self.final_result['Block Efficiency'] = "N/A"
            self.final_result['Decoding Speed'] = "N/A"
            return

        sample_length = sample / len_  # block efficiency
        DS = len_ / time_ * gamma

        total_tokens_generated = sample
        tokens_per_second = total_tokens_generated / time_ if time_ > 0 else 0

        print(f"Block Efficiency (BE): {sample_length:.2f}")
        print(f"Decoding Speed (DS): {DS:.2f} tokens/s")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Total time: {time_:.2f}s")
        print(f"Effective tokens/s: {tokens_per_second:.2f}")

        self.final_result['Block Efficiency'] = f"{sample_length:.2f}"
        self.final_result['Decoding Speed'] = f"{DS:.2f}"
        self.final_result['Total Tokens'] = int(total_tokens_generated)
        self.final_result['Total Time (s)'] = f"{time_:.2f}"
        self.final_result['Tokens/s'] = f"{tokens_per_second:.2f}"

    def __call__(self):
        if self.args.debug:
            self.debug()
            return

        self.total_counts = {
            "draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [],
            "step_back_probs": [], "p_i": [], "q_i": [], "time": [], "ids": [],
            "draft_raw_logits": [], "target_raw_logits": [],
        }

        print(f"Starting BFCL evaluation (category: {self.args.category})")

        # Output dirs
        os.makedirs(f"results/{self.args.name}/outputs/accuracy/", exist_ok=True)
        os.makedirs(f"results/{self.args.name}/outputs/efficiency/", exist_ok=True)
        os.makedirs(f"results/{self.args.name}/outputs/final_result/", exist_ok=True)

        acc_file = f'results/{self.args.name}/outputs/accuracy/{self.sd}.txt'
        checker_fn = CATEGORY_CHECKERS[self.args.category]

        self.progress = 0
        self.correct_count = 0

        with open(acc_file, 'w') as fd:
            for idx in tqdm(range(self.start_index, self.end_index), total=self.num_samples):
                sample = self.prompts[idx]
                sample_id = sample["id"]
                gt_entry = self.answer_map.get(sample_id, {})
                ground_truth = gt_entry.get("ground_truth", [])

                self.progress += 1
                print(f"\nprogress: {self.start_index + self.progress}/{self.end_index}")
                print(f"sample_id: {sample_id}")

                # Build prompt
                system_msg, user_content = format_bfcl_prompt(sample)
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ]

                input_text = self.tokenizer2.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
                input_length = input_ids.shape[1]

                embedding_device = self.draft_model.model.embed_tokens.weight.device
                input_ids = input_ids.to(embedding_device)

                start = time.time()

                if self.args.speculative:
                    outputs = self.speculative_decoding(input_ids)
                else:
                    if self.args.model == "target":
                        outputs = self.target_model.generate(
                            input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                            tokenizer=self.tokenizer2,
                            eta_cutoff=self.args.eta_cutoff,
                            min_p=self.args.min_p,
                        )
                    else:
                        outputs = self.draft_model.generate(
                            input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                            tokenizer=self.tokenizer1,
                            eta_cutoff=self.args.eta_cutoff,
                            min_p=self.args.min_p,
                        )

                end = time.time()
                self.total_counts["time"].append(end - start)

                # Decode generated tokens
                generated_tokens = outputs[0][input_length:]
                ans_ = self.tokenizer1.decode(generated_tokens, skip_special_tokens=True)

                # Parse function calls and evaluate
                model_calls = parse_function_calls(ans_)
                passed = checker_fn(model_calls, ground_truth)

                if passed:
                    self.correct_count += 1

                print(f"Model output: {ans_[:300]}")
                print(f"Parsed calls: {model_calls}")
                print(f"Ground truth: {ground_truth}")
                print(f"Passed: {passed}")
                print(f"Running accuracy: {self.correct_count}/{self.progress} = {self.correct_count / self.progress:.4f}")

                fd.write(f'Q_id: {sample_id}\n'
                         f'Q_user: {user_content}\n'
                         f'A_model:\n{ans_}\n'
                         f'A_parsed: {json.dumps(model_calls)}\n'
                         f'A_ground_truth: {json.dumps(ground_truth)}\n'
                         f'A_passed: {passed}\n\n')
                fd.flush()

                if self.progress % 10 == 0:
                    efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                    with open(efficiency_file, "w") as f:
                        json.dump(self.total_counts, f)

            # Save final efficiency file
            efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
            print(f'saving to {efficiency_file}')
            with open(efficiency_file, "w") as f:
                json.dump(self.total_counts, f)

        # Parse results
        accuracy = self.correct_count / self.num_samples if self.num_samples > 0 else 0.0
        self.efficiency_analysis(efficiency_file)

        self.final_result.update({
            'accuracy': accuracy,
            'correct': self.correct_count,
            'num_samples': self.num_samples,
            'dataset': self.dataset_name,
            'category': self.args.category,
            'target_model': self.target_model_name,
            'draft_model': self.draft_model_name,
            'gamma': self.args.gamma,
            'speculative': self.args.speculative,
        })

        final_result_file = f"results/{self.args.name}/outputs/final_result/{self.sd}_final_result.json"
        with open(final_result_file, "w") as f:
            json.dump(self.final_result, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print(f"FINAL RESULTS SUMMARY - BFCL ({self.args.category})")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Category: {self.args.category}")
        print(f"Samples: [{self.start_index}, {self.end_index}) = {self.num_samples} samples")
        print(f"Target model: {self.target_model_name}")
        print(f"Draft model: {self.draft_model_name}")
        print(f"Gamma: {self.args.gamma}")
        print("-" * 60)
        print(f"Accuracy: {accuracy:.4f} ({self.correct_count}/{self.num_samples}) [idx {self.start_index}-{self.end_index}]")
        print(f"Block Efficiency: {self.final_result.get('Block Efficiency', 'N/A')}")
        print(f"Decoding Speed: {self.final_result.get('Decoding Speed', 'N/A')} tokens/s")
        print("=" * 60)
        print(f"Results saved to: {final_result_file}")

    def debug(self):
        """Debug mode: print prompt format for first few samples."""
        print("Running in DEBUG mode")
        debug_samples = min(5, self.num_samples)

        for idx in range(self.start_index, self.start_index + debug_samples):
            sample = self.prompts[idx]
            sample_id = sample["id"]
            gt_entry = self.answer_map.get(sample_id, {})
            ground_truth = gt_entry.get("ground_truth", [])

            system_msg, user_content = format_bfcl_prompt(sample)

            print(f"\n{'='*60}")
            print(f"DEBUG Sample {idx+1}/{debug_samples}")
            print(f"ID: {sample_id}")
            print(f"User: {user_content}")
            print(f"Functions: {len(sample['function'])} defined")
            for fn in sample['function'][:3]:
                print(f"  - {fn['name']}: {fn.get('description', '')[:80]}")
            print(f"Ground truth: {ground_truth}")
            print(f"System prompt length: {len(system_msg)} chars")

            # Show what the tokenized prompt looks like
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]
            input_text = self.tokenizer2.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
            print(f"Tokenized input length: {input_ids.shape[1]} tokens")
            print(f"{'='*60}")


def main():
    hsd = HSD_BFCL()
    hsd()


if __name__ == "__main__":
    main()
