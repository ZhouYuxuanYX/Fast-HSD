import re
import os
import glob
from tqdm import tqdm
from torch import LongTensor, FloatTensor, eq, device
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
from accelerate import dispatch_model
from torch import nn
import numpy as np
import torch
import json
import argparse
import time
import random
import orjson
import mmap


def argparse_setup():

    parser = argparse.ArgumentParser(prog='eval_mbppplus')
    parser.add_argument('--backward', action='store_true', default=False)  # hsd framework
    parser.add_argument('--clever', action='store_true', default=False)  # lossless
    parser.add_argument('--multidraft', type=int, default=1)

    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)  # lossy without cap
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--gamma', default=10, type=int, help='number of assisted tokens')
    parser.add_argument('--lenience', default=1, type=float, help='lenience factor')
    parser.add_argument("--fast", action='store_true', default=False)  # lossy with cap

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--target-model', default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='target model')
    parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', help='draft model')
    parser.add_argument('--dataset', default="mbppplus", help='dataset name')
    parser.add_argument('--prompt', type=str, default='original', help='prompt type')

    parser.add_argument('--model', help='must be target or draft', default="target")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='additional name to distinguish different runs')
    parser.add_argument('--results-dir', type=str, default=None, help='absolute path for results output (set by sbatch)')
    parser.add_argument('--cascade', action='store_true', default=False)
    parser.add_argument('--uniform-draft', action='store_true', default=False, help='Make draft model output uniform distribution')
    parser.add_argument('--max_new_tokens', type=int, default=4000, help='max new tokens to generate')
    parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
    parser.add_argument('--eta_cutoff', type=float, default=None, help='eta cutoff for logits processor')
    parser.add_argument('--min_p', type=float, default=None, help='min p for logits processor')
    parser.add_argument('--eta_spd', type=float, default=None, help='eta cutoff for spd')
    parser.add_argument('--min_p_spd', type=float, default=None, help='min p for spd')
    parser.add_argument('--cos_lambda', type=float, default=None, help='COS lambda for spd')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples for evaluation (default: all samples)')
    args = parser.parse_args()
    print(args)
    return args


def tp(name, x):
    """Print name, type, and a short summary (no big dumps)."""
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    def brief(v):
        if torch is not None and isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
        if isinstance(v, np.ndarray):
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, dict):
            return f"dict(len={len(v)}, keys={list(v)[:3]})"
        if isinstance(v, (list, tuple)):
            head = v[:3]
            return f"{type(v).__name__}(len={len(v)}, head={head})"
        return repr(v)
    print(f"{name}: <{type(x).__name__}> {brief(x)}")


def to_jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if (input_ids[0][-len(stop_ids[0]):] == stop_ids[0]).all():
                print("input tail ids:", input_ids[0][-len(stop_ids[0]):])
                print("stop ids:", stop_ids[0])
                return True
        return False


def flatten_hist_lengths(hist_lengths_list):
    # list-of-lists -> flat list; skip each inner leading 0
    return [l for hl in hist_lengths_list for l in hl[1:]]


def kv_bytes_for_chunk(cfg, L_i, a_i, batch=1, dtype_bytes=2):
    """
    Closed-form KV bytes for a verification chunk that accepts a_i tokens,
    starting from sequence length L_i.
    """
    H = getattr(cfg, "num_attention_heads")
    H_kv = getattr(cfg, "num_key_value_heads", H)  # GQA-aware
    D = cfg.hidden_size
    Dh = D // H
    layers = cfg.num_hidden_layers
    # 2 * sum_{t=0}^{a_i-1}(L_i+t) + 2*a_i  =  2*a_i*L_i + a_i*(a_i-1) + 2*a_i
    factor_1 = (2 * a_i * L_i) + (a_i * (a_i - 1))
    factor_2 = (2 * a_i)
    factor = factor_1 + factor_2

    return layers * batch * H_kv * Dh * dtype_bytes * factor, factor_1, factor_2


def kv_time_ms_target(cfg, prefix_len, per_verify_lengths, batch, dtype_bytes, bw_bytes_per_s):
    L = prefix_len
    total_bytes = 0
    f1_append = []
    f2_append = []
    for a in per_verify_lengths:
        bytes_, f1, f2 = kv_bytes_for_chunk(cfg, L_i=L, a_i=a, batch=batch, dtype_bytes=dtype_bytes)
        total_bytes += bytes_
        f1_append.append(f1)
        f2_append.append(f2)

    return (total_bytes / max(1.0, bw_bytes_per_s)) * 1000.0, f1_append, f2_append


def effective_bandwidth_Bps():
    import torch
    import time
    if not torch.cuda.is_available():
        # No CUDA: return NaN so you don't divide by zero; skip KV split on CPU/MPS
        return float("nan")
    x = torch.empty((1024, 1024, 4000), device="cuda", dtype=torch.float16)  # ~1GB
    torch.cuda.synchronize()
    t0 = time.time()
    _ = x.clone()
    torch.cuda.synchronize()
    t1 = time.time()
    return x.numel() * x.element_size() / (t1 - t0)


def move_rotary_emb_to_device(model):
    # Get the device of the layer where rotary embedding is applied
    try:
        device = model.model.embed_tokens.weight.device
        if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
            model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    except AttributeError:
        print("Could not move rotary_emb.inv_freq — structure may be different")


def manual_device_map(model, same_device_for_input_output=True):
    """
    Manually create a balanced device map for a Hugging Face transformer model.

    Args:
        model: The loaded model (e.g. AutoModelForCausalLM).
        same_device_for_input_output: If True, places embedding and output (lm_head) on same device (cuda:0).

    Returns:
        device_map dict to use with `dispatch_model`.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    # Find model layers
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

    # Distribute transformer layers
    for i in range(num_layers):
        gpu_id = i // layers_per_gpu
        key = f"{prefix}.{i}"
        device_map[key] = f"cuda:{gpu_id}"

    # Assign special components
    if same_device_for_input_output:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = 'cuda:0'
    else:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = f"cuda:{n_gpus - 1}"

    # Assign final normalization to last GPU
    device_map[norm_key] = f"cuda:{n_gpus - 1}"

    return device_map


def infer_model_size_tag(model_name: str) -> str:
    """Infer model size token like 72B or 0.5B from a model path/name."""
    candidate = os.path.basename(model_name.rstrip("/")) if model_name else ""

    # Prefer basename match (works for local paths and HF repo IDs)
    match = re.search(r'(\d+(?:\.\d+)?[BM])', candidate, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: search full string
    match = re.search(r'(\d+(?:\.\d+)?[BM])', model_name or "", re.IGNORECASE)
    if match:
        return match.group(1).upper()

    print(f"Warning: could not infer model size from '{model_name}', defaulting to 72B")
    return "72B"


def resolve_model_source(model_name: str) -> str:
    """
    Resolve model source path for local HF cache-style folders.

    Supports local dirs shaped like:
      <model_dir>/refs/main
      <model_dir>/snapshots/<commit_hash>/...
    and returns the concrete snapshot directory for from_pretrained().
    """
    if not model_name or not os.path.isdir(model_name):
        return model_name

    if os.path.exists(os.path.join(model_name, "config.json")):
        return model_name

    refs_main = os.path.join(model_name, "refs", "main")
    snapshots_dir = os.path.join(model_name, "snapshots")

    if os.path.isdir(snapshots_dir):
        commit_hash = None
        if os.path.isfile(refs_main):
            try:
                with open(refs_main, "r") as f:
                    commit_hash = f.read().strip()
            except Exception:
                commit_hash = None

        if commit_hash:
            candidate = os.path.join(snapshots_dir, commit_hash)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                return candidate

        candidates = sorted(
            [p for p in glob.glob(os.path.join(snapshots_dir, "*")) if os.path.isdir(p)],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "config.json")):
                return candidate

    return model_name


def load_mbppplus_local_dataset(base_dir: str):
    """
    Load MBPP+ from local path robustly.
    Supports:
      1) standard dataset script/repo via load_dataset(base_dir, split='test')
      2) saved dataset via datasets.load_from_disk(base_dir)
      3) datasets cache arrow files (e.g. .../mbppplus-test.arrow)
    """
    # 1) Normal local dataset loading
    try:
        return load_dataset(base_dir, split='test')
    except Exception:
        pass

    # 2) load_from_disk structure
    try:
        ds_obj = datasets.load_from_disk(base_dir)
        if isinstance(ds_obj, datasets.DatasetDict):
            if 'test' in ds_obj:
                return ds_obj['test']
            first_split = next(iter(ds_obj.keys()))
            return ds_obj[first_split]
        return ds_obj
    except Exception:
        pass

    # 3) Arrow cache layout fallback (HuggingFace cache stores IPC stream files)
    arrow_candidates = glob.glob(os.path.join(base_dir, '**', '*test*.arrow'), recursive=True)
    if not arrow_candidates:
        arrow_candidates = glob.glob(os.path.join(base_dir, '**', '*.arrow'), recursive=True)

    if arrow_candidates:
        arrow_candidates.sort(key=lambda p: (0 if 'test' in os.path.basename(p) else 1, len(p)))
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc

            table = ipc.open_stream(pa.memory_map(arrow_candidates[0], 'r')).read_all()
            return table.to_pylist()
        except Exception as e:
            raise RuntimeError(
                f"Failed to read local arrow dataset file {arrow_candidates[0]}: {e}"
            )

    raise FileNotFoundError(
        f"Could not load MBPP+ dataset from {base_dir}. "
        "Expected a HF dataset dir, a load_from_disk dir, or .arrow files."
    )


class HSD_MBPPPlus():
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = self.args.draft_model
        print(f'target model: {self.target_model_name}')
        print(f'draft model: {self.draft_model_name}')
        self.model_size = infer_model_size_tag(self.target_model_name)
        print(f'model size: {self.model_size}')
        if float(self.model_size[:-1]) > 3:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        print(f'using device: {self.device}')

        # Set random seed for reproducibility
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        # =======================Loading MBPP+ dataset=======================
        self.dataset_name = 'mbppplus'
        # Load MBPP+ dataset from HuggingFace
        dataset = load_dataset('evalplus/mbppplus', split='test')
        self.mbppplus_test = dataset

        # Use --num_samples if provided, otherwise use all test samples
        total_available = len(self.mbppplus_test)
        if self.args.num_samples is not None:
            self.num_samples = min(self.args.num_samples, total_available)
        else:
            self.num_samples = total_available

        print(f"Total available samples: {total_available}")
        print(f"num_samples to evaluate: {self.num_samples}")

        # =======================Model setup=================================
        self.model_setup()
        self.sd = self.test_setup()

        self.final_result = {'Block Efficiency': None, 'Decoding Speed': None}

    def extract_function_name_from_tests(self, test_list):
        """
        Extract the expected function name from test assertions.

        Args:
            test_list: List of assertion strings like
                       ['assert sort_matrix([[1, 2, 3], ...]) == ...', ...]

        Returns:
            The function name or None if not found
        """
        if not test_list:
            return None

        # Get first test and extract function name
        first_test = test_list[0] if isinstance(test_list, list) else str(test_list)

        # Pattern: assert function_name(...)
        match = re.search(r'assert\s+(\w+)\s*\(', first_test)
        if match:
            return match.group(1)

        return None

    def extract_code_from_response(self, response):
        """
        Extract Python code from model response.
        Handles markdown code blocks and raw code.

        Args:
            response: The full model response string

        Returns:
            Extracted Python code string
        """
        # Pattern 1: ```python ... ``` or ```Python ... ```
        python_block_pattern = r'```[pP]ython\s*\n(.*?)```'
        matches = re.findall(python_block_pattern, response, re.DOTALL)
        if matches:
            # Return the longest match (most likely the complete function)
            return max(matches, key=len).strip()

        # Pattern 2: ``` ... ``` (generic code block)
        generic_block_pattern = r'```\s*\n(.*?)```'
        matches = re.findall(generic_block_pattern, response, re.DOTALL)
        if matches:
            # Filter for Python-like code (contains def, import, etc.)
            python_matches = [m for m in matches if
                             re.search(r'^\s*(def |import |from |class )', m, re.MULTILINE)]
            if python_matches:
                return max(python_matches, key=len).strip()
            if matches:
                return max(matches, key=len).strip()

        # Pattern 3: Find function definitions directly in the response
        # Look for complete function definitions
        func_pattern = r'((?:import\s+\w+.*?\n)*(?:from\s+\w+.*?\n)*\s*def\s+\w+\s*\([^)]*\)\s*:.*?)(?=\n(?:def\s|\Z|```|[A-Z][a-z]+:|\n\n[A-Z]))'
        matches = re.findall(func_pattern, response, re.DOTALL)
        if matches:
            # Combine all function definitions and imports
            return '\n\n'.join(m.strip() for m in matches)

        # Pattern 4: Just extract everything that looks like Python code
        # This is a fallback - extract lines starting from first 'import' or 'def'
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if re.match(r'^(import |from |def |class |@)', line):
                in_code = True
            if in_code:
                # Stop if we hit non-code content
                if re.match(r'^[A-Z][a-z]+.*:$', line) and not line.strip().endswith('"""'):
                    break
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

        # Fallback: return the original response
        return response

    def test_answer(self, code, test_list, test_imports=None, timeout=10):
        """
        Execute the code and run test assertions to check correctness.

        Args:
            code: The generated code string to test (may include explanation text)
            test_list: List of assertion strings
            test_imports: List of import statements needed for testing
            timeout: Maximum execution time in seconds (to handle infinite loops)

        Returns:
            Tuple of (passed: bool, num_passed: int, total_tests: int)
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        try:
            # Extract code from model response
            extracted_code = self.extract_code_from_response(code)

            # Create a clean namespace for execution with common imports
            exec_globals = {
                '__builtins__': __builtins__,
            }

            # Add common imports that might be needed
            common_imports = """
import math
import heapq
import itertools
import collections
import functools
import operator
import re
import sys
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import permutations, combinations, product
from functools import reduce, lru_cache
from typing import List, Dict, Tuple, Optional, Set
"""
            try:
                exec(common_imports, exec_globals)
            except:
                pass  # Ignore import errors

            # Set up timeout handler (Unix only, skip on Windows)
            old_handler = None
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            try:
                # Execute test imports if provided (MBPP+ specific)
                if test_imports:
                    for imp in test_imports:
                        if imp and imp.strip():
                            try:
                                exec(imp, exec_globals)
                            except:
                                pass  # Skip failed imports

                # Execute the extracted code to define functions/classes
                exec(extracted_code, exec_globals)

                # Run each test assertion and count passes
                num_passed = 0
                total_tests = len(test_list) if test_list else 0
                for test in test_list:
                    try:
                        exec(test, exec_globals)
                        num_passed += 1
                    except AssertionError:
                        pass  # Test assertion failed
                    except Exception:
                        pass  # Other runtime error

                # All tests passed
                all_passed = (num_passed == total_tests and total_tests > 0)
                return all_passed, num_passed, total_tests

            finally:
                # Cancel the alarm
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError:
            # Code took too long (likely infinite loop)
            return False, 0, len(test_list) if test_list else 0
        except Exception as e:
            # Code has syntax error or runtime error
            if self.args.debug:
                print(f"Test execution error: {type(e).__name__}: {e}")
            return False, 0, len(test_list) if test_list else 0

    def parse_test_list_string(self, test_str):
        """
        Parse a test list string back to a Python list.
        The string is in format: "A_testlist:\n['assert ...', 'assert ...', ...]"
        """
        import ast
        try:
            # Remove the prefix and clean up
            test_str = test_str.strip()
            if test_str.startswith('A_testlist:'):
                test_str = test_str[len('A_testlist:'):].strip()

            # Try to parse as literal Python list
            test_list = ast.literal_eval(test_str)
            if isinstance(test_list, list):
                return test_list
        except:
            pass

        # Fallback: try to extract assertions manually
        assertions = re.findall(r'assert\s+.+?(?=(?:assert|\]|$))', test_str, re.DOTALL)
        if assertions:
            return [a.strip().rstrip(',').strip("'\"") for a in assertions]

        return []

    def parse_pred_ans(self, filename):
        """
        Parse the prediction/answer file and calculate accuracy.

        MBPP+ format:
        Q_task_id: ...
        Q_prompt: ...
        A_model: ...
        A_testlist: ...
        A_test_imports: ...
        A_passed: ...
        A_num_passed: ...
        A_total_tests: ...
        """
        with open(filename) as fd:
            content = fd.read()

        # Split by question blocks
        blocks = re.split(r'(?=Q_task_id: )', content)
        blocks = [b.strip() for b in blocks if b.strip() and b.strip().startswith('Q_task_id:')]

        num_q = 0
        acc = 0
        total_tests_passed = 0
        total_tests_overall = 0

        for block in blocks:
            num_q += 1

            # Parse each field
            passed_match = re.search(r'A_passed:\n(.*?)(?=A_num_passed:|$)', block, re.DOTALL)
            num_passed_match = re.search(r'A_num_passed:\n(.*?)(?=A_total_tests:|$)', block, re.DOTALL)
            total_tests_match = re.search(r'A_total_tests:\n(.*?)$', block, re.DOTALL)

            # Check if we have cached pass result
            if passed_match:
                passed_str = passed_match.group(1).strip()
                if passed_str.lower() == 'true':
                    acc += 1

            if num_passed_match:
                try:
                    total_tests_passed += int(num_passed_match.group(1).strip())
                except:
                    pass

            if total_tests_match:
                try:
                    total_tests_overall += int(total_tests_match.group(1).strip())
                except:
                    pass

        accuracy = float(acc / num_q) if num_q > 0 else 0.0
        test_pass_rate = float(total_tests_passed / total_tests_overall) if total_tests_overall > 0 else 0.0
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, accuracy))
        print('total_tests_passed %d total_tests_overall %d test_pass_rate %.4f' % (total_tests_passed, total_tests_overall, test_pass_rate))
        return accuracy, test_pass_rate

    def speculative_decoding(self, input_ids):

        outputs, counts = self.target_model.generate(input_ids, max_new_tokens=4000, do_sample=True,
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

        self.total_counts["draft_eval"].append(counts.get("draft_eval", []))
        self.total_counts["sample_length"].append(counts.get("sample_length", []))
        self.total_counts["target_eval"].append(counts.get("target_eval", []))
        self.total_counts["p_i"].append(counts.get("p_i", []))
        self.total_counts["q_i"].append(counts.get("q_i", []))
        self.total_counts["step_back_probs"].append(counts.get("step_back_probs", []))
        self.total_counts["total_step"].append(counts.get("total_step", []))
        self.total_counts["ids"].append(counts.get("ids", []))
        self.total_counts["draft_raw_logits"].append(counts.get("draft_raw_logits", []))
        self.total_counts["target_raw_logits"].append(counts.get("target_raw_logits", []))
        self.total_counts["draft"].append(to_jsonable(counts.get("draft", [])))
        self.total_counts["draft_probs"].append(to_jsonable(counts.get("draft_probs", [])))
        self.total_counts["target"].append(to_jsonable(counts.get("target", [])))
        self.total_counts["target_probs"].append(to_jsonable(counts.get("target_probs", [])))
        self.total_counts["n_matched"].append(to_jsonable(counts.get("n_matched", [])))

        return outputs

    def efficiency_analysis(self, file_path):
        """
        Analyze block efficiency and decoding speed from the efficiency log file.

        Block Efficiency (BE): Average number of tokens accepted per verification step
        Decoding Speed (DS): Tokens generated per second
        """
        gamma = self.args.gamma  # Use configured gamma instead of hardcoded 10

        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            count = orjson.loads(mm[:])   # mm[:] gives you a bytes object
            mm.close()

        # Check if we have speculative decoding data
        if not count.get("draft_eval") or len(count["draft_eval"]) == 0:
            # Non-speculative mode: calculate simple decoding speed
            total_time = sum(float(t) for t in count.get("time", []))
            total_tokens = 0  # We don't track output tokens in non-speculative mode

            print("Non-speculative mode: Block Efficiency not applicable")
            print(f"Total time: {total_time:.2f}s")

            self.final_result['Block Efficiency'] = "N/A"
            self.final_result['Decoding Speed'] = "N/A"
            self.final_result['Total Time (s)'] = f"{total_time:.2f}"
            return

        draft = 0
        target = 0
        step = 0
        sample = 0
        time_ = 0
        len_ = 0

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

        # Calculate per-step averages (same as results_analysis.py)
        draft_eval = draft / len_
        target_eval = target / len_
        total_step = step / len_
        sample_length = sample / len_  # block efficiency
        DS = len_ / time_ * gamma  # decoding speed (tokens/s)

        # Also calculate total tokens and tokens/second
        total_tokens_generated = sample
        tokens_per_second = total_tokens_generated / time_ if time_ > 0 else 0

        print(f"Block Efficiency (BE): {sample_length:.2f}")
        print(f"Decoding Speed (DS): {DS:.2f} tokens/s")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Total time: {time_:.2f}s")
        print(f"Effective tokens/s: {tokens_per_second:.2f}")
        print('---')

        self.final_result['Block Efficiency'] = f"{sample_length:.2f}"
        self.final_result['Decoding Speed'] = f"{DS:.2f}"
        self.final_result['Total Tokens'] = int(total_tokens_generated)
        self.final_result['Total Time (s)'] = f"{time_:.2f}"
        self.final_result['Tokens/s'] = f"{tokens_per_second:.2f}"

    def __call__(self):
        if self.args.debug:
            self.debug()
        else:
            self.total_counts = {"draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [],
                                 "step_back_probs": [], "p_i": [], "q_i": [], "time": [], "ids": [],
                                 "draft_raw_logits": [], "target_raw_logits": [],
                                 "draft": [], "draft_probs": [], "target": [], "target_probs": [],
                                 "n_matched": []}

            print("Starting MBPP+ evaluation")
            self.BW = effective_bandwidth_Bps()

            # Determine output base directory
            if self.args.results_dir:
                out_base = self.args.results_dir
            else:
                out_base = f"results/{self.args.name}"

            # Generate output directories
            os.makedirs(f"{out_base}/outputs/accuracy/", exist_ok=True)
            os.makedirs(f"{out_base}/outputs/efficiency/", exist_ok=True)
            os.makedirs(f"{out_base}/outputs/final_result/", exist_ok=True)

            acc_file = f'{out_base}/outputs/accuracy/{self.sd}.txt'

            self.progress = 0
            self.correct_count = 0
            self.total_tests_passed = 0
            self.total_tests_overall = 0

            with open(acc_file, 'w') as fd:
                for idx in tqdm(range(self.num_samples), total=self.num_samples):
                    sample = self.mbppplus_test[idx]

                    task_id = sample['task_id']
                    prompt = sample['prompt']
                    test_list = sample['test_list']
                    test_imports = sample.get('test_imports', [])

                    print(f"progress: {self.progress}/{self.num_samples}")
                    self.progress += 1

                    print(f"task_id: {task_id}")
                    print(f"prompt: {prompt[:100]}...")

                    # Extract expected function name from test cases
                    expected_func_name = self.extract_function_name_from_tests(test_list)

                    # Build prompt with function name hint
                    if expected_func_name:
                        user_prompt = f"{prompt}\n\nThe function should be named `{expected_func_name}`."
                    else:
                        user_prompt = prompt

                    # Use chat template to avoid generating strange strings with repetition penalty
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a code generation model. Output only the Python function definition without any explanation or markdown formatting."},
                        {"role": "user", "content": user_prompt}
                    ]

                    input_text = self.tokenizer2.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
                    input_length = input_ids.shape[1]  # Store input length to extract only generated tokens

                    embedding_device = self.draft_model.model.embed_tokens.weight.device
                    input_ids = input_ids.to(embedding_device)

                    start = time.time()

                    if self.args.speculative:
                        outputs = self.speculative_decoding(input_ids)
                    else:
                        if self.args.model == "target":
                            outputs = self.target_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                                                                 tokenizer=self.tokenizer2,
                                                                 eta_cutoff=self.args.eta_cutoff,
                                                                 min_p=self.args.min_p
                                                                 )
                            used_model = self.target_model
                        else:
                            outputs = self.draft_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                                                                tokenizer=self.tokenizer1,
                                                                eta_cutoff=self.args.eta_cutoff,
                                                                min_p=self.args.min_p
                                                                )
                            used_model = self.draft_model

                    end = time.time()
                    self.total_counts["time"].append(end - start)

                    # Only decode the generated tokens (skip input tokens)
                    generated_tokens = outputs[0][input_length:]
                    ans_ = self.tokenizer1.decode(generated_tokens, skip_special_tokens=True)

                    test_passed, num_passed, total_tests = self.test_answer(ans_, test_list, test_imports)
                    self.total_tests_passed += num_passed
                    self.total_tests_overall += total_tests

                    if test_passed:
                        self.correct_count += 1
                        print("All tests passed")
                        print(f"Task ID: {task_id}")
                        print(f"Answer: {ans_[:200]}...")
                    else:
                        print(f"Tests failed: {num_passed}/{total_tests} passed")
                        print(f"Task ID: {task_id}")
                        print(f"Answer: {ans_[:200]}...")

                    print(f"Running accuracy: {self.correct_count}/{self.progress} = {self.correct_count / self.progress:.4f}")
                    print(f"Running test pass rate: {self.total_tests_passed}/{self.total_tests_overall} = {self.total_tests_passed / self.total_tests_overall:.4f}")

                    fd.write('Q_task_id: %s\nQ_prompt: %s\nA_model:\n%s\nA_testlist:\n%s\nA_test_imports:\n%s\nA_passed:\n%s\nA_num_passed:\n%d\nA_total_tests:\n%d\n\n' % (
                        task_id, prompt, ans_, test_list, test_imports, test_passed, num_passed, total_tests))
                    fd.flush()  # Force write to disk immediately

                    if self.progress % 10 == 0:  # Save every 10 iterations
                        efficiency_file = f"{out_base}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                        with open(efficiency_file, "w") as f:
                            json.dump(to_jsonable(self.total_counts), f)

                # Save final efficiency file
                efficiency_file = f"{out_base}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                print(f'saving to {efficiency_file}')
                with open(efficiency_file, "w") as f:
                    json.dump(to_jsonable(self.total_counts), f)

            accuracy, test_pass_rate = self.parse_pred_ans(acc_file)
            self.efficiency_analysis(efficiency_file)
            self.final_result.update({
                'accuracy': accuracy,
                'test_pass_rate': test_pass_rate,
                'num_samples': self.num_samples,
                'dataset': self.dataset_name,
                'target_model': self.target_model_name,
                'draft_model': self.draft_model_name,
                'gamma': self.args.gamma,
                'speculative': self.args.speculative,
            })

            # Generate dynamic date string (MMDD format)
            final_result_file = f"{out_base}/outputs/final_result/{self.sd}_final_result.json"
            with open(final_result_file, "w") as f:
                json.dump(self.final_result, f, indent=2)

            # Print final summary
            print("\n" + "=" * 60)
            print("FINAL RESULTS SUMMARY - MBPP+")
            print("=" * 60)
            print(f"Dataset: {self.dataset_name}")
            print(f"Number of samples: {self.num_samples}")
            print(f"Target model: {self.target_model_name}")
            print(f"Draft model: {self.draft_model_name}")
            print(f"Gamma: {self.args.gamma}")
            print("-" * 60)
            print(f"Accuracy (all tests pass): {accuracy:.4f} ({int(accuracy * self.num_samples)}/{self.num_samples})")
            print(f"Block Efficiency: {self.final_result.get('Block Efficiency', 'N/A')}")
            print(f"Decoding Speed: {self.final_result.get('Decoding Speed', 'N/A')} tokens/s")
            print("=" * 60)
            print(f"Results saved to: {final_result_file}")

    def debug(self):
        """Debug mode: run on a small subset of samples for testing."""
        print("Running in DEBUG mode")
        self.total_counts = {"draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [],
                             "step_back_probs": [], "p_i": [], "q_i": [], "time": [], "ids": [],
                             "draft": [], "draft_probs": [], "target": [], "target_probs": [],
                             "n_matched": []}

        # Only run on first 5 samples in debug mode
        debug_samples = min(5, self.num_samples)

        for idx in range(debug_samples):
            sample = self.mbppplus_test[idx]
            task_id = sample['task_id']
            prompt = sample['prompt']
            test_list = sample['test_list']
            test_imports = sample.get('test_imports', [])

            print(f"\n{'=' * 50}")
            print(f"DEBUG Sample {idx + 1}/{debug_samples}")
            print(f"Task ID: {task_id}")
            print(f"Prompt: {prompt}")
            print(f"Test imports: {test_imports}")
            print(f"Test list ({len(test_list)} tests):")
            for t in test_list[:3]:  # Show first 3 tests
                print(f"  - {t[:100]}...")
            print(f"{'=' * 50}")

    def model_setup(self):
        draft_model_source = resolve_model_source(self.draft_model_name)
        target_model_source = resolve_model_source(self.target_model_name)

        print(f"resolved draft source: {draft_model_source}")
        print(f"resolved target source: {target_model_source}")

        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_source,
                                                                 device_map={"": self.device} if int(self.model_size[:-1]) < 32 else None)

        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_source,
                                                                  device_map={"": self.device} if int(self.model_size[:-1]) < 32 else None)

        # Set generation config
        self.draft_model.generation_config.num_assistant_tokens = self.args.gamma
        self.draft_model.generation_config.assistant_confidence_threshold = 0
        self.draft_model.generation_config.temperature = self.args.temperature
        self.draft_model.generation_config.top_k = self.args.top_k
        self.draft_model.generation_config.top_p = self.args.top_p

        self.target_model.generation_config.num_assistant_tokens = self.args.gamma
        self.target_model.generation_config.assistant_confidence_threshold = 0
        self.target_model.generation_config.temperature = self.args.temperature
        self.target_model.generation_config.top_k = self.args.top_k
        self.target_model.generation_config.top_p = self.args.top_p

        vocab_size = min(self.draft_model.config.vocab_size, self.target_model.config.vocab_size)
        self.draft_model.config.vocab_size = vocab_size
        self.target_model.config.vocab_size = vocab_size
        self.same_tokenizer = self.target_model.config.get_text_config().vocab_size == self.draft_model.config.get_text_config().vocab_size

        # Manually resize lm_head if needed
        if hasattr(self.draft_model, "lm_head"):
            old_lm_head = self.draft_model.lm_head
            dtype = old_lm_head.weight.dtype
            self.draft_model.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)
            self.draft_model.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

        if hasattr(self.target_model, "lm_head"):
            old_lm_head = self.target_model.lm_head
            dtype = old_lm_head.weight.dtype

            new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)

            with torch.no_grad():
                new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
                    old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]

            self.target_model.lm_head = new_lm_head

        # Redistribute for multi-gpu
        if torch.cuda.is_available() and int(self.model_size[:-1]) > 14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model, same_device_for_input_output=False)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1, offload_dir=None)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2, offload_dir=None)

        print("dispatch model finished")

        self.draft_model.eval()
        self.target_model.eval()

        # Fix rotary embedding buffers that may still be on CPU
        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

        # Make draft model output uniform distribution if requested
        if self.args.uniform_draft:
            self.make_draft_model_uniform()

        # Load tokenizers
        self.tokenizer1 = AutoTokenizer.from_pretrained(draft_model_source)
        self.tokenizer2 = AutoTokenizer.from_pretrained(target_model_source)

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
        if self.args.parallel:
            sd += "_parallel"
        if self.args.temperature:
            sd += f"_t{self.args.temperature}"
        if self.args.top_p:
            sd += f"_topp_{self.args.top_p}"
        if self.args.lenience < 1:
            sd += f"_lenience_{self.args.lenience}"
        if self.args.eta_cutoff:
            sd += f"_eta_cutoff_{self.args.eta_cutoff}"
        if self.args.min_p:
            sd += f"_min_p_{self.args.min_p}"
        if self.args.cascade:
            sd += "_cascade"
        if self.args.uniform_draft:
            sd += "_uniform"
        if self.args.eta_spd:
            sd += f"_eta_spd_{self.args.eta_spd}"
        if self.args.min_p_spd:
            sd += f"_min_p_spd_{self.args.min_p_spd}"
        if self.args.cos_lambda:
            sd += f"ABS_B_coslambda_{self.args.cos_lambda}"

        sd += f"_mbppplus"
        sd += f'{self.args.name}'
        return sd


def main():
    hsd = HSD_MBPPPlus()
    hsd()


if __name__ == "__main__":
    main()
