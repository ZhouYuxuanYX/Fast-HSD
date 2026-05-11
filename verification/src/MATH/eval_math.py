import re
import os
import glob
from tqdm import tqdm
from torch import LongTensor, FloatTensor, eq, device
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
import datasets
from accelerate import dispatch_model
from torch import nn
import numpy as np
import torch
import json
import argparse
import numpy as np
import time
import random
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

import orjson
import numpy as np
import mmap
import time
import os
from typing import List, Tuple, Dict, Optional
import json
from fractions import Fraction
import sympy
from sympy import simplify, nsimplify, Rational, latex
from sympy.parsing.latex import parse_latex


def argparse_setup():

    parser = argparse.ArgumentParser(prog='eval_math')
    parser.add_argument('--backward', action='store_true', default=False)
    parser.add_argument('--clever', action='store_true', default=False)
    parser.add_argument('--multidraft', type=int, default=1)

    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--gamma',  default=10, type=int, help='number of assisted tokens')
    parser.add_argument('--lenience',  default=1, type=float, help='lenience factor')
    parser.add_argument("--fast", action='store_true', default=False)

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--target-model', default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='target model')
    parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', help='draft model')
    parser.add_argument('--dataset', default="math", help='dataset name')

    parser.add_argument('--model', help='must be target or draft', default="target")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='additional name to distinguish different runs')
    parser.add_argument('--cascade', action='store_true', default=False)
    parser.add_argument('--uniform-draft', action='store_true', default=False, help='Make draft model output uniform distribution')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='max new tokens to generate')
    parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples for evaluation')
    parser.add_argument('--sample_start', type=int, default=0, help='0-based inclusive start index for evaluation subset')
    parser.add_argument('--sample_end', type=int, default=None, help='0-based exclusive end index for evaluation subset')
    parser.add_argument('--k', type=int, default=1, help='Number of sampled evaluation rounds per problem')
    parser.add_argument('--eta_cutoff', type=float, default=None, help='eta cutoff for logits processor')
    parser.add_argument('--min_p', type=float, default=None, help='min p for logits processor')
    parser.add_argument('--eta_spd', type=float, default=None, help='eta cutoff for spd')
    parser.add_argument('--min_p_spd', type=float, default=None, help='min p for spd')
    parser.add_argument('--cos_lambda', type=float, default=None, help='COS lambda for spd')
    parser.add_argument('--cos_mu', type=float, default=None, help='COS mu for spd')
    args = parser.parse_args()
    print(args)
    return args


def move_rotary_emb_to_device(model):
    try:
        device = model.model.embed_tokens.weight.device
        if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
            model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    except AttributeError:
        print("Could not move rotary_emb.inv_freq — structure may be different")


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
        key = f"{prefix}.{i}"
        device_map[key] = f"cuda:{gpu_id}"

    if same_device_for_input_output:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = 'cuda:0'
    else:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = f"cuda:{n_gpus - 1}"

    device_map[norm_key] = f"cuda:{n_gpus - 1}"
    return device_map


def effective_bandwidth_Bps():
    import torch, time
    if not torch.cuda.is_available():
        return float("nan")
    try:
        x = torch.empty((1024, 1024, 4000), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = x.clone()
        torch.cuda.synchronize()
        t1 = time.time()
        return x.numel() * x.element_size() / (t1 - t0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("Skipping effective bandwidth benchmark due to CUDA OOM")
        return float("nan")


def parse_model_size(model_name: str) -> Tuple[str, float]:
    """
    Parse model size from either a hub id or local path.
    Examples:
      - Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8 -> ("72B", 72.0)
      - /path/to/Qwen2.5-0.5B-Instruct-GPTQ-Int8 -> ("0.5B", 0.5)
    """
    candidate = os.path.basename(model_name.rstrip("/"))
    match = re.search(r"-(\d+(?:\.\d+)?)B(?:-|$)", candidate)
    if not match:
        match = re.search(r"-(\d+(?:\.\d+)?)B(?:-|$)", model_name)

    if match:
        size_num = float(match.group(1))
        return f"{match.group(1)}B", size_num

    return "0B", 0.0


def resolve_model_source(model_name: str) -> str:
    if not model_name or not os.path.isdir(model_name):
        return model_name

    if os.path.exists(os.path.join(model_name, "config.json")):
        return model_name

    refs_main = os.path.join(model_name, "refs", "main")
    snapshots_dir = os.path.join(model_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_name

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


class MathEvaluator():
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = self.args.draft_model
        print(f'target model: {self.target_model_name}')
        print(f'draft model: {self.draft_model_name}')
        self.model_size, self.model_size_num = parse_model_size(self.target_model_name)
        print(f'model size: {self.model_size}')
        if self.model_size_num > 3:
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

        # Load MATH dataset
        self.load_dataset()

        print(f"num_samples: {self.num_samples}")

        # Model setup
        self.model_setup()
        self.sd = self.test_setup()

        self.final_result = {'Block Efficiency': None, 'Decoding Speed': None}

    def load_dataset(self):
        """Load and sample the MATH dataset."""
        dataset = load_dataset('PATH_TO_MATH_DATASET', split='train')
        self.dataset_name = 'math'
        
        total_samples = len(dataset)
        self.num_samples = min(self.args.num_samples, total_samples)
        
        # Random sampling with seed for reproducibility
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        sampled_indices = all_indices[:self.num_samples]
        
        # Store sampled data
        self.problems = [dataset[i]['problem'] for i in sampled_indices]
        self.solutions = [dataset[i]['solution'] for i in sampled_indices]
        self.levels = [dataset[i]['level'] for i in sampled_indices]
        self.types = [dataset[i]['type'] for i in sampled_indices]
        
        # Extract ground truth answers
        self.ground_truths = [self.extract_boxed_answer(sol) for sol in self.solutions]
        
        print(f"Loaded {self.num_samples} samples from MATH dataset")

    def extract_boxed_answer(self, solution: str) -> str:
        """
        Extract answer from \\boxed{} in the solution.
        Handles nested braces properly.
        """
        # Find \boxed{ and extract content with proper brace matching
        boxed_pattern = r'\\boxed\{'
        matches = list(re.finditer(boxed_pattern, solution))
        
        if not matches:
            # Try alternative patterns
            # Pattern: \boxed answer (without braces)
            alt_match = re.search(r'\\boxed\s*(\S+)', solution)
            if alt_match:
                return alt_match.group(1)
            return ""
        
        # Get the last \boxed{} (usually the final answer)
        last_match = matches[-1]
        start = last_match.end()
        
        # Count braces to find matching closing brace
        brace_count = 1
        pos = start
        while pos < len(solution) and brace_count > 0:
            if solution[pos] == '{':
                brace_count += 1
            elif solution[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            return solution[start:pos-1].strip()
        
        return ""

    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract the answer from the model's response.
        Tries multiple patterns to find the answer.
        """
        # Pattern 1: \boxed{...}
        boxed_answer = self.extract_boxed_answer(response)
        if boxed_answer:
            return boxed_answer
        
        # Pattern 2: "The answer is X" or "the answer is X"
        answer_pattern = r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\.\n]+)'
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: "= X" at the end of a line (common in math solutions)
        equals_pattern = r'=\s*([^\n=]+?)\s*$'
        matches = re.findall(equals_pattern, response, re.MULTILINE)
        if matches:
            # Return the last match
            return matches[-1].strip()
        
        # Pattern 4: Look for $...$ at the end
        dollar_pattern = r'\$([^$]+)\$\s*\.?\s*$'
        match = re.search(dollar_pattern, response)
        if match:
            return match.group(1).strip()
        
        # Pattern 5: Bold or emphasized answer **X**
        bold_pattern = r'\*\*([^*]+)\*\*\s*\.?\s*$'
        match = re.search(bold_pattern, response)
        if match:
            return match.group(1).strip()
        
        return ""

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison.
        Handles fractions, decimals, LaTeX formatting, etc.
        """
        if not answer:
            return ""
        
        # Remove common LaTeX formatting
        answer = answer.strip()
        answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\$', '', answer)
        answer = re.sub(r'\\,', '', answer)
        answer = re.sub(r'\\;', '', answer)
        answer = re.sub(r'\\!', '', answer)
        answer = re.sub(r'\\quad', ' ', answer)
        answer = re.sub(r'\\qquad', ' ', answer)
        
        # Normalize negative fractions: \frac{-a}{b} or -\frac{a}{b} -> -a/b
        # First handle \frac{-a}{b} -> -\frac{a}{b}
        answer = re.sub(r'\\frac\{-([^{}]*)\}\{([^{}]*)\}', r'-\\frac{\1}{\2}', answer)
        answer = re.sub(r'\\dfrac\{-([^{}]*)\}\{([^{}]*)\}', r'-\\dfrac{\1}{\2}', answer)
        
        # Handle \frac{a}{b} -> a/b
        while r'\frac' in answer:
            frac_match = re.search(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', answer)
            if frac_match:
                num, den = frac_match.groups()
                answer = answer[:frac_match.start()] + f"({num})/({den})" + answer[frac_match.end():]
            else:
                break
        
        # Handle \dfrac similarly
        while r'\dfrac' in answer:
            frac_match = re.search(r'\\dfrac\{([^{}]*)\}\{([^{}]*)\}', answer)
            if frac_match:
                num, den = frac_match.groups()
                answer = answer[:frac_match.start()] + f"({num})/({den})" + answer[frac_match.end():]
            else:
                break
        
        # Handle \sqrt{x}
        answer = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', answer)
        
        # Remove remaining backslashes for simple commands
        answer = re.sub(r'\\([a-zA-Z]+)', r'\1', answer)
        
        # Normalize whitespace around commas and inside parentheses
        answer = re.sub(r'\s*,\s*', ',', answer)  # Remove spaces around commas
        answer = re.sub(r'\(\s+', '(', answer)    # Remove space after (
        answer = re.sub(r'\s+\)', ')', answer)    # Remove space before )
        
        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        # Remove trailing periods
        answer = answer.rstrip('.')
        
        return answer

    def extract_elements_from_answer(self, answer: str) -> List[str]:
        """
        Extract individual elements from a comma-separated answer.
        Handles tuples like (1,3) and lists like -1, 2, 7
        """
        # Remove outer parentheses if present
        answer = answer.strip()
        if (answer.startswith('(') and answer.endswith(')')) or \
           (answer.startswith('{') and answer.endswith('}')):
            answer = answer[1:-1]
        
        # Split by comma
        elements = [e.strip() for e in answer.split(',')]
        return elements

    def normalize_element(self, elem: str) -> Optional[float]:
        """Normalize a single element to a comparable value."""
        elem = elem.strip()
        
        # Try to parse as fraction
        frac_match = re.match(r'^(-?)\(?(-?\d+)\)?/\(?(\d+)\)?$', elem)
        if frac_match:
            sign = -1 if frac_match.group(1) == '-' else 1
            num = int(frac_match.group(2))
            den = int(frac_match.group(3))
            return sign * num / den
        
        # Handle percentage
        pct_match = re.match(r'^(-?[\d.]+)\s*%$', elem)
        if pct_match:
            return float(pct_match.group(1)) / 100
        
        # Try to parse as float
        try:
            return float(elem)
        except:
            pass
        
        return None

    def compare_answers(self, pred: str, gt: str) -> bool:
        """
        Compare predicted answer with ground truth.
        Returns True if they match.
        """
        if not pred or not gt:
            return False
        
        # Check if ground truth has multiple equivalent forms (e.g., "0.56 = 56%")
        if '=' in gt:
            gt_alternatives = [g.strip() for g in gt.split('=')]
            for gt_alt in gt_alternatives:
                if self.compare_answers(pred, gt_alt):
                    return True
        
        # Normalize both answers
        pred_norm = self.normalize_answer(pred)
        gt_norm = self.normalize_answer(gt)
        
        # Direct string comparison (case-insensitive)
        if pred_norm.lower() == gt_norm.lower():
            return True
        
        # Check if both are comma-separated lists/tuples (order-independent comparison)
        if ',' in pred_norm or ',' in gt_norm:
            pred_elements = self.extract_elements_from_answer(pred_norm)
            gt_elements = self.extract_elements_from_answer(gt_norm)
            
            if len(pred_elements) == len(gt_elements) and len(pred_elements) > 1:
                # Try order-independent comparison
                # First try string matching
                if sorted([e.lower() for e in pred_elements]) == sorted([e.lower() for e in gt_elements]):
                    return True
                
                # Try numerical comparison for each element
                pred_nums = [self.normalize_element(e) for e in pred_elements]
                gt_nums = [self.normalize_element(e) for e in gt_elements]
                
                if None not in pred_nums and None not in gt_nums:
                    # Sort and compare
                    pred_sorted = sorted(pred_nums)
                    gt_sorted = sorted(gt_nums)
                    if len(pred_sorted) == len(gt_sorted):
                        all_match = True
                        for p, g in zip(pred_sorted, gt_sorted):
                            if abs(p - g) > 1e-6:
                                all_match = False
                                break
                        if all_match:
                            return True
        
        # Try numerical comparison for single values
        try:
            pred_val = self.parse_numeric(pred_norm)
            gt_val = self.parse_numeric(gt_norm)
            if pred_val is not None and gt_val is not None:
                # Check if they're approximately equal
                if abs(pred_val - gt_val) < 1e-6:
                    return True
                # For fractions, also check exact equality
                if isinstance(pred_val, Fraction) and isinstance(gt_val, Fraction):
                    return pred_val == gt_val
        except:
            pass
        
        # Try symbolic comparison using sympy
        try:
            pred_sym = self.parse_symbolic(pred_norm)
            gt_sym = self.parse_symbolic(gt_norm)
            if pred_sym is not None and gt_sym is not None:
                diff = simplify(pred_sym - gt_sym)
                if diff == 0:
                    return True
        except:
            pass
        
        return False

    def parse_numeric(self, s: str) -> Optional[float]:
        """Try to parse a string as a number or fraction."""
        s = s.strip()
        
        # Handle fractions like "1/4" or "(1)/(4)"
        frac_match = re.match(r'^\(?(-?\d+)\)?/\(?(\d+)\)?$', s)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            return Fraction(num, den)
        
        # Handle simple integers
        try:
            return float(s)
        except:
            pass
        
        return None

    def parse_symbolic(self, s: str):
        """Try to parse a string as a sympy expression."""
        try:
            # Replace common patterns
            s = s.replace('^', '**')
            s = s.replace('sqrt', 'sympy.sqrt')
            s = s.replace('pi', 'sympy.pi')
            
            # Try eval with sympy
            result = eval(s, {"sympy": sympy, "__builtins__": {}})
            return result
        except:
            pass
        
        try:
            # Try parsing as LaTeX
            return parse_latex(s)
        except:
            pass
        
        return None

    def test_answer(self, response: str, ground_truth: str) -> Tuple[bool, str, str]:
        """
        Test if the model's response contains the correct answer.
        
        Returns:
            (is_correct, extracted_answer, ground_truth)
        """
        extracted = self.extract_answer_from_response(response)
        is_correct = self.compare_answers(extracted, ground_truth)
        return is_correct, extracted, ground_truth

    def speculative_decoding(self, input_ids):
        outputs, counts = self.target_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
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
                                cos_mu=self.args.cos_mu,
                                )

        self.total_counts["draft_eval"].append(counts["draft_eval"])
        self.total_counts["sample_length"].append(counts["sample_length"])
        self.total_counts["target_eval"].append(counts["target_eval"])
        self.total_counts["p_i"].append(counts["p_i"])
        self.total_counts["q_i"].append(counts["q_i"])
        self.total_counts["step_back_probs"].append(counts["step_back_probs"])
        self.total_counts["total_step"].append(counts["total_step"])
        self.total_counts["ids"].append(counts["ids"])

        return outputs

    def efficiency_analysis(self, file_path):
        """Analyze block efficiency and decoding speed."""
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

            draft += draft_list[draft_list==gamma].sum()
            target += target_list[draft_list==gamma].sum()
            step += step_list[draft_list==gamma].sum()
            sample += sample_list[draft_list==gamma].sum()
            time_ += float(count["time"][n])
            len_ += len(sample_list[draft_list==gamma])

        if len_ == 0:
            print("Warning: No valid samples found for efficiency analysis")
            self.final_result['Block Efficiency'] = "N/A"
            self.final_result['Decoding Speed'] = "N/A"
            return

        sample_length = sample/len_
        DS = len_ / time_ * gamma
        total_tokens_generated = sample
        tokens_per_second = total_tokens_generated / time_ if time_ > 0 else 0

        print(f"Block Efficiency (BE): {sample_length:.2f}")
        print(f"Decoding Speed (DS): {DS:.2f} tokens/s")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Total time: {time_:.2f}s")
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
            self.total_counts = {"draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                "step_back_probs":[], "p_i":[], "q_i":[], "time":[], "ids":[]}
            
            print("Starting MATH evaluation")
            self.BW = effective_bandwidth_Bps()
            
            os.makedirs(f"results/{self.args.name}/outputs/accuracy/", exist_ok=True)
            os.makedirs(f"results/{self.args.name}/outputs/efficiency/", exist_ok=True)
            os.makedirs(f"results/{self.args.name}/outputs/final_result/", exist_ok=True)

            acc_file = f'results/{self.args.name}/outputs/accuracy/{self.sd}.txt'

            self.progress = 0
            self.correct_count = 0
            
            with open(acc_file, 'w') as fd:
                for problem, solution, ground_truth, level, prob_type in tqdm(
                    zip(self.problems, self.solutions, self.ground_truths, self.levels, self.types),
                    total=self.num_samples
                ):
                    print(f"\nprogress: {self.progress}/{self.num_samples}")
                    self.progress += 1

                    print(f"Problem: {problem[:100]}...")
                    print(f"Ground truth: {ground_truth}")

                    # Build prompt for math problem
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful math assistant. Solve the problem step by step and provide the final answer in \\boxed{} format."},
                        {"role": "user", "content": problem}
                    ]
                    
                    input_text = self.tokenizer2.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
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
                            outputs = self.target_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                                                            tokenizer=self.tokenizer2,
                                                            eta_cutoff=self.args.eta_cutoff,
                                                            min_p=self.args.min_p
                                                            )
                        else:
                            outputs = self.draft_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, do_sample=True,
                                                            tokenizer=self.tokenizer1,
                                                            eta_cutoff=self.args.eta_cutoff,
                                                            min_p=self.args.min_p
                                                            )

                    end = time.time()
                    self.total_counts["time"].append(end-start)

                    # Only decode generated tokens
                    generated_tokens = outputs[0][input_length:]
                    response = self.tokenizer1.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Test the answer
                    is_correct, extracted_answer, gt = self.test_answer(response, ground_truth)
                    
                    if is_correct:
                        self.correct_count += 1
                        print("✓ Correct")
                    else:
                        print("✗ Incorrect")
                    
                    print(f"Extracted: {extracted_answer}")
                    print(f"Expected: {gt}")
                    print(f"Running accuracy: {self.correct_count}/{self.progress} = {self.correct_count/self.progress:.4f}")

                    # Save to file
                    fd.write('='*60 + '\n')
                    fd.write(f'Problem: {problem}\n')
                    fd.write(f'Level: {level}\n')
                    fd.write(f'Type: {prob_type}\n')
                    fd.write(f'Ground Truth: {ground_truth}\n')
                    fd.write(f'Model Response:\n{response}\n')
                    fd.write(f'Extracted Answer: {extracted_answer}\n')
                    fd.write(f'Correct: {is_correct}\n')
                    fd.write('\n')
                    fd.flush()
                    
                    if self.progress % 10 == 0:
                        efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                        with open(efficiency_file, "w") as f:
                            json.dump(self.total_counts, f)

                efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                print(f'Saving to {efficiency_file}')
                with open(efficiency_file, "w") as f:
                    json.dump(self.total_counts, f)

            accuracy = self.correct_count / self.num_samples if self.num_samples > 0 else 0.0
            self.efficiency_analysis(efficiency_file)
            
            self.final_result.update({
                'accuracy': accuracy,
                'correct': self.correct_count,
                'total': self.num_samples,
                'num_samples': self.num_samples,
                'dataset': self.dataset_name,
                'target_model': self.target_model_name,
                'draft_model': self.draft_model_name,
                'gamma': self.args.gamma,
                'speculative': self.args.speculative,
                'seed': self.args.seed,
            })
            
            final_result_file = f"results/{self.args.name}/outputs/final_result/{self.sd}_final_result.json"
            with open(final_result_file, "w") as f:
                json.dump(self.final_result, f, indent=2)
            
            # Print final summary
            print("\n" + "="*60)
            print("FINAL RESULTS SUMMARY")
            print("="*60)
            print(f"Dataset: {self.dataset_name}")
            print(f"Number of samples: {self.num_samples}")
            print(f"Random seed: {self.args.seed}")
            print(f"Target model: {self.target_model_name}")
            print(f"Draft model: {self.draft_model_name}")
            print(f"Gamma: {self.args.gamma}")
            print("-"*60)
            print(f"Accuracy: {accuracy:.4f} ({self.correct_count}/{self.num_samples})")
            print(f"Block Efficiency: {self.final_result.get('Block Efficiency', 'N/A')}")
            print(f"Decoding Speed: {self.final_result.get('Decoding Speed', 'N/A')} tokens/s")
            print("="*60)
            print(f"Results saved to: {final_result_file}")

    def model_setup(self):
        draft_model_source = resolve_model_source(self.draft_model_name)
        target_model_source = resolve_model_source(self.target_model_name)

        print(f"load draft model: {draft_model_source}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_source,
            device_map={"": self.device} if self.model_size_num < 32 else None)

        print(f"load target model: {target_model_source}")
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_source,
            device_map={"": self.device} if self.model_size_num < 32 else None)

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

        if torch.cuda.is_available() and self.model_size_num > 14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1, offload_dir=None)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2, offload_dir=None)

        print("Dispatch model finished")

        self.draft_model.eval()
        self.target_model.eval()

        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

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
        if self.args.k and self.args.k > 1:
            sd += f"_k_{self.args.k}"
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
            sd += f"_coslambda_{self.args.cos_lambda}"
        if self.args.cos_mu:
            sd += f"_cos_mu_{self.args.cos_mu}"
        
        sd += f"_seed{self.args.seed}"
        sd += f'{self.args.name}'
        return sd


def main():
    evaluator = MathEvaluator()
    evaluator()


if __name__ == "__main__":
    main()
