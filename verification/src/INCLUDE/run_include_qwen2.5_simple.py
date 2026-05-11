# Simplified INCLUDE evaluation with speculative decoding
# Evaluates multilingual questions from INCLUDE dataset (44 languages)
# Samples questions from each language and evaluates as a whole

import json
import time
import random
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import re
import os
import glob
import warnings

from tqdm import tqdm
from torch import LongTensor, FloatTensor, eq, device
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
import datasets
from accelerate import dispatch_model
from torch import nn
import numpy as np
import torch
import argparse
from huggingface_hub import snapshot_download
import cProfile, pstats, io

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# parse arguments
def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backward', action='store_true', default=False)  # hsd framework
    parser.add_argument('--clever', action='store_true', default=False)  # lossless
    parser.add_argument('--multidraft', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)  # lossy without cap
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--gamma', default=10, type=int, help='number of assited tokens')
    parser.add_argument('--lenience', default=1, type=float, help='lenience factor')
    parser.add_argument("--fast", action='store_true', default=False)  # lossy with cap
    parser.add_argument('--model', help='must be target or draft', default="target")
    parser.add_argument('--prompt', default='original', help='must be complex or original')
    parser.add_argument('--target-model', default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='target model name')
    parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', help='draft model path or HF model name')
    parser.add_argument('--dataset-dir', default=None, help='local INCLUDE dataset directory')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='additional name to distinguish different runs')
    parser.add_argument('--cascade', action='store_true', default=False)
    parser.add_argument('--ntrain', type=int, default=3, help='number of few-shot examples')
    parser.add_argument('--max_model_length', type=int, default=4096, help='max model context length')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='max new tokens to generate')
    parser.add_argument('--samples_per_language', type=int, default=5, help='number of questions to sample per language')
    parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
    parser.add_argument('--eta_cutoff', type=float, default=None, help='eta cutoff for logits processor')
    parser.add_argument('--min_p', type=float, default=None, help='min p for logits processor')
    parser.add_argument('--eta_spd', type=float, default=None, help='eta cutoff for spd')
    parser.add_argument('--min_p_spd', type=float, default=None, help='min p for spd')
    parser.add_argument('--cos_lambda', type=float, default=None, help='COS lambda for spd')
    parser.add_argument('--cos_mu', type=float, default=None, help='COS mu for spd')
    args = parser.parse_args()
    return args


# INCLUDE has 4 options (A-D)
choices = ["A", "B", "C", "D"]

# Languages in INCLUDE dataset
INCLUDE_LANGUAGES = [
    "Albanian", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", 
    "Bengali", "Bulgarian", "Chinese", "Croatian", "Dutch", "Estonian", 
    "Finnish", "French", "Georgian", "German", "Greek", "Hebrew", "Hindi", 
    "Hungarian", "Indonesian", "Italian", "Japanese", "Kazakh", "Korean", 
    "Lithuanian", "Malay", "Malayalam", "Nepali", "North Macedonian", "Persian", 
    "Polish", "Portuguese", "Russian", "Serbian", "Spanish", "Tagalog", 
    "Tamil", "Telugu", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese"
]


def preprocess_include(example):
    """Convert INCLUDE format to a standardized format"""
    options = [
        example.get("option_a", ""),
        example.get("option_b", ""),
        example.get("option_c", ""),
        example.get("option_d", "")
    ]
    # Filter out empty options
    options = [opt for opt in options if opt and opt.strip()]
    
    return {
        "question": example["question"],
        "options": options,
        "answer_index": example["answer"],  # 0-3 index
        "language": example.get("language", "Unknown"),
        "domain": example.get("domain", "Unknown"),
        "subject": example.get("subject", "Unknown"),
        "country": example.get("country", "Unknown"),
        "level": example.get("level", "Unknown"),
    }


def format_cot_example(example, including_answer=True):
    """Format a single CoT example for INCLUDE"""
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        if i < len(choices):
            prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        answer_idx = example.get("answer_index", 0)
        answer = choices[answer_idx] if answer_idx < len(choices) else "A"
        prompt += f"Answer: Let's think step by step. The answer is ({answer}).\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_examples, curr, k):
    """Generate CoT prompt with k few-shot examples"""
    # Initial prompt for multilingual evaluation
    prompt = "The following are multiple choice questions (with answers). "
    prompt += "Think step by step and then finish your answer with \"the answer is (X)\" "
    prompt += "where X is the correct letter choice (A, B, C, or D).\n\n"
    
    # Use first k validation examples as few-shot
    for example in val_examples[:k]:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    """
    Extract answer from model output for INCLUDE.
    INCLUDE expects single letter answers from A-D.
    """
    # First try: "answer is (X)" or "answer is X"
    pattern = r"answer is \(?([A-D])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Second try: "Answer: X"
    match = re.search(r'[aA]nswer:\s*\(?([A-D])\)?', text)
    if match:
        return match.group(1).upper()
    
    # Third try: find last occurrence of a letter A-D
    pattern = r"\b([A-D])\b(?!.*\b[A-D]\b)"
    match = re.search(pattern, text.upper(), re.DOTALL)
    if match:
        return match.group(1)
    
    # Silently return None if extraction fails
    return None


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
    candidate = os.path.basename(model_name.rstrip("/")) if model_name else ""
    match = re.search(r'(\d+(?:\.\d+)?[BM])', candidate, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'(\d+(?:\.\d+)?[BM])', model_name or "", re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "72B"


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


def resolve_include_dataset_dir(dataset_dir: str) -> str:
    if dataset_dir and os.path.isdir(dataset_dir):
        return dataset_dir

    fallbacks = [
        'PATH_TO_INCLUDE_DATASET',
        'PATH_TO_INCLUDE_DATASET',
    ]
    for candidate in fallbacks:
        if os.path.isdir(candidate):
            return candidate

    return dataset_dir


class HSD():
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = self.args.draft_model
        self.dataset_dir = resolve_include_dataset_dir(self.args.dataset_dir)
        self.model_size = infer_model_size_tag(self.target_model_name)
        self.total_counts = {"draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                  "step_back_probs":[], "p_i":[], "q_i":[], "time":[], "ids":[]}
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

        # =======================Loading INCLUDE dataset=======================
        print(f"Loading INCLUDE dataset from {self.dataset_dir}...")
        self.load_include_dataset()
        
        # =======================Model setup=================================
        self.model_setup()
        self.sd = self.test_setup()

    def load_include_dataset(self):
        """Load and sample from INCLUDE dataset - treating all splits as one pool"""
        # Available language configs in INCLUDE dataset
        available_languages = [
            'Albanian', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian',
            'Bengali', 'Bulgarian', 'Chinese', 'Croatian', 'Dutch', 'Estonian',
            'Finnish', 'French', 'Georgian', 'German', 'Greek', 'Hebrew', 'Hindi',
            'Hungarian', 'Indonesian', 'Italian', 'Japanese', 'Kazakh', 'Korean',
            'Lithuanian', 'Malay', 'Malayalam', 'Nepali', 'North Macedonian', 'Persian',
            'Polish', 'Portuguese', 'Russian', 'Serbian', 'Spanish', 'Tagalog',
            'Tamil', 'Telugu', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese'
        ]
        # Note: Skipping 'Dutch - Flemish' and 'Dutch-Flemish' as they may be duplicates
        
        print(f"Available language configs: {len(available_languages)}")
        
        all_data = []
        samples_per_lang = self.args.samples_per_language
        
        print(f"\nSampling {samples_per_lang} questions from each language (seed={self.args.seed})...")
        
        for lang in available_languages:
            try:
                # Local dataset layout: <dataset_dir>/<lang>/0.0.0/<hash>/include-base-44-*.arrow
                import pyarrow as pa
                import pyarrow.ipc as ipc

                lang_root = os.path.join(self.dataset_dir, lang, "0.0.0")
                hash_dirs = sorted([d for d in glob.glob(os.path.join(lang_root, "*")) if os.path.isdir(d)])
                if not hash_dirs:
                    raise FileNotFoundError(f"No cached dataset directory found under {lang_root}")

                language_cache_dir = hash_dirs[-1]

                # Combine validation + test into one pool
                lang_data = []
                for split_name in ["validation", "test"]:
                    arrow_file = os.path.join(language_cache_dir, f"include-base-44-{split_name}.arrow")
                    if not os.path.exists(arrow_file):
                        continue

                    table = ipc.open_stream(pa.memory_map(arrow_file, "r")).read_all()
                    for ex in table.to_pylist():
                        lang_data.append(preprocess_include(ex))
                
                if len(lang_data) == 0:
                    print(f"  {lang}: skipped (no data found)")
                    continue
                
                # Sample from combined data
                if len(lang_data) > samples_per_lang:
                    sampled = random.sample(lang_data, samples_per_lang)
                else:
                    sampled = lang_data
                
                all_data.extend(sampled)
                print(f"  {lang}: sampled {len(sampled)}/{len(lang_data)} questions")
                
            except Exception as e:
                print(f"  {lang}: error loading - {str(e)[:50]}")
                continue
        
        # Shuffle the combined data
        random.shuffle(all_data)
        
        self.test_df = all_data
        # Use first few examples from each language as few-shot examples
        self.val_df = all_data[:min(50, len(all_data))]
        
        print(f"\nTotal sampled: {len(self.test_df)} questions from {len(available_languages)} languages")

    def speculative_decoding(self, input_ids):
        outputs, counts = self.target_model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, 
                                                     do_sample=True, 
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

    def __call__(self):
        """
        Evaluate the model on INCLUDE dataset (all languages together).
        """
        print('='*80)
        print('Evaluating INCLUDE (all languages together)')
        print('='*80)
        
        correct = 0
        wrong = 0
        results = []
        
        # Track per-language stats for analysis
        lang_stats = {}
        
        # Output file for all results
        output_file = f'final_results/test_Qwen2.5-{self.model_size}_{self.sd}_include_results.json'

        # print out generation config
        print(self.target_model.generation_config)
        print(self.draft_model.generation_config)
        
        for i in tqdm(range(len(self.test_df)), desc="Evaluating"):
            curr = self.test_df[i]
            lang = curr.get("language", "Unknown")
            
            # Generate prompt with few-shot examples
            k = self.args.ntrain
            prompt_length_ok = False
            prompt = None
            
            while not prompt_length_ok and k >= 0:
                prompt = generate_cot_prompt(self.val_df, curr, k)
                # Tokenize to check length
                input_ids_test = self.tokenizer2(prompt, return_tensors="pt").input_ids
                length = len(input_ids_test[0])
                
                if length < self.args.max_model_length - self.args.max_new_tokens:
                    prompt_length_ok = True
                else:
                    k -= 1
            
            if k < 0:
                k = 0
                prompt = generate_cot_prompt(self.val_df, curr, k)
            
            # Use chat template
            messages = [
                {"role": "system",
                 "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            input_text = self.tokenizer2.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
            
            # Move to correct device
            embedding_device = self.draft_model.model.embed_tokens.weight.device
            input_ids = input_ids.to(embedding_device)
            
            # Generate
            start = time.time()
            
            if self.args.speculative:
                outputs = self.speculative_decoding(input_ids)
            else:
                if self.args.model == "target":
                    outputs = self.target_model.generate(
                        input_ids, 
                        max_new_tokens=self.args.max_new_tokens, 
                        do_sample=True,  
                        tokenizer=self.tokenizer2,
                        eta_cutoff=self.args.eta_cutoff,
                        min_p=self.args.min_p
                    )
                else:
                    outputs = self.draft_model.generate(
                        input_ids, 
                        max_new_tokens=self.args.max_new_tokens, 
                        do_sample=True, 
                        tokenizer=self.tokenizer1,
                        eta_cutoff=self.args.eta_cutoff,
                        min_p=self.args.min_p
                    )
            
            end = time.time()
            self.total_counts["time"].append(end - start)
            
            # Decode answer
            ans_model = self.tokenizer1.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            
            # Extract answer
            pred = extract_answer(ans_model)
            
            # Get ground truth
            answer_index = curr["answer_index"]
            answer = choices[answer_index] if answer_index < len(choices) else "A"
            
            # Store result
            curr["pred"] = pred
            curr["model_outputs"] = ans_model
            results.append(curr)
            
            # Initialize language stats if needed
            if lang not in lang_stats:
                lang_stats[lang] = {"correct": 0, "wrong": 0}
            
            # Update stats
            if not pred:
                # Random guess if extraction failed
                print(f"Random guess for question {i+1} (language: {lang})")
                x = random.randint(0, len(curr["options"]) - 1)
                if x == answer_index:
                    correct += 1
                    lang_stats[lang]["correct"] += 1
                else:
                    wrong += 1
                    lang_stats[lang]["wrong"] += 1
            elif pred == answer:
                correct += 1
                lang_stats[lang]["correct"] += 1
            else:
                wrong += 1
                lang_stats[lang]["wrong"] += 1
            
            # Print progress every 100 questions
            if (i + 1) % 100 == 0:
                current_acc = correct / (correct + wrong)
                avg_time = sum(self.total_counts["time"][-100:]) / min(100, len(self.total_counts["time"]))
                print(f"\n[Progress] {i+1}/{len(self.test_df)} | Acc: {current_acc:.4f} | Avg time: {avg_time:.2f}s")
        
        # Save results
        with open(output_file, "w") as fo:
            fo.write(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Calculate final metrics
        total_questions = len(self.total_counts["time"])
        accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0
        avg_time_per_q = sum(self.total_counts["time"]) / total_questions if total_questions > 0 else 0
        
        # Save metrics
        save_path = f"final_results/{self.sd}_include_total_counts.json"
        with open(save_path, "w") as f:
            json.dump(self.total_counts, f, indent=2)
        
        # Save language stats
        lang_stats_path = f"final_results/{self.sd}_include_lang_stats.json"
        with open(lang_stats_path, "w") as f:
            json.dump(lang_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\n{'ACCURACY':^80}")
        print("-"*80)
        print(f"  Correct:   {correct}")
        print(f"  Wrong:     {wrong}")
        print(f"  Accuracy:  {accuracy:.4f}")
        
        print(f"\n{'PERFORMANCE':^80}")
        print("-"*80)
        print(f"  Total questions:          {total_questions}")
        print(f"  Average time per question: {avg_time_per_q:.2f}s")
        print(f"  Total time:               {sum(self.total_counts['time'])/60:.2f} minutes")
        
        if self.args.speculative:
            # Filter for steps where draft_eval == gamma (full draft rounds)
            gamma = self.args.gamma
            total_time_seconds = sum(self.total_counts["time"])
            
            # Calculate metrics filtering for steps where draft == gamma
            filtered_draft_evals = 0
            filtered_target_evals = 0
            filtered_total_steps = 0
            filtered_sample_length = 0
            filtered_steps_count = 0  # Number of steps where draft == gamma
            
            for n in range(len(self.total_counts["draft_eval"])):
                draft_array = np.array(self.total_counts["draft_eval"][n])
                target_array = np.array(self.total_counts["target_eval"][n])
                step_array = np.array(self.total_counts["total_step"][n])
                sample_array = np.array(self.total_counts["sample_length"][n])
                
                # Filter for steps where draft == gamma
                mask = (draft_array == gamma)
                filtered_draft_evals += draft_array[mask].sum()
                filtered_target_evals += target_array[mask].sum()
                filtered_total_steps += step_array[mask].sum()
                filtered_sample_length += sample_array[mask].sum()
                filtered_steps_count += mask.sum()
            
            # Block efficiency (BE) = average tokens accepted per filtered step
            block_efficiency = filtered_sample_length / filtered_steps_count if filtered_steps_count > 0 else 0
            
            # Decoding speed (DS) = (filtered_steps / time) * gamma
            decoding_speed = (filtered_steps_count / total_time_seconds * gamma) if total_time_seconds > 0 else 0
            
            # Additional metrics
            avg_draft_per_step = filtered_draft_evals / filtered_steps_count if filtered_steps_count > 0 else 0
            acceptance_rate = filtered_sample_length / filtered_draft_evals if filtered_draft_evals > 0 else 0
            
            # Total metrics (unfiltered, for reference)
            total_draft_evals = sum([sum(x) for x in self.total_counts["draft_eval"]])
            total_target_evals = sum([sum(x) for x in self.total_counts["target_eval"]])
            total_accepted = sum([sum(x) for x in self.total_counts["sample_length"]])
            total_steps = sum([len(x) for x in self.total_counts["sample_length"]])
            
            print(f"\n{'SPECULATIVE DECODING METRICS':^80}")
            print("-"*80)
            print(f"  Total draft tokens (all):     {total_draft_evals}")
            print(f"  Total accepted tokens (all):  {total_accepted}")
            print(f"  Total steps (all):            {total_steps}")
            print(f"  Filtered steps (draft==γ):    {filtered_steps_count}")
            print(f"  Total time:                   {total_time_seconds:.2f}s")
            print(f"")
            print(f"  Block efficiency (BE):        {block_efficiency:.2f} tokens/step")
            print(f"  Decoding speed (DS):          {decoding_speed:.2f} tokens/second")
            print(f"  Acceptance rate:              {acceptance_rate:.4f}")
            print(f"  Avg draft tokens/step:        {avg_draft_per_step:.2f}")
        
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  - {output_file}")
        print(f"  - {save_path}")
        print(f"  - {lang_stats_path}")
        print("="*80)

    def model_setup(self):
        draft_model_source = resolve_model_source(self.draft_model_name)
        target_model_source = resolve_model_source(self.target_model_name)

        print(f"load draft model: {draft_model_source}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_source,
                                                                device_map={"": self.device} if float(
                                                                    self.model_size[:-1]) < 32 else None)

        print(f"load target model: {target_model_source}")
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_source,
                                                                 device_map={"": self.device} if float(
                                                                     self.model_size[:-1]) < 32 else None)

        self.draft_model.generation_config.num_assistant_tokens = self.args.gamma
        self.draft_model.generation_config.assistant_confidence_threshold = 0

        self.target_model.generation_config.num_assistant_tokens = self.args.gamma
        self.target_model.generation_config.assistant_confidence_threshold = 0

        if self.args.temperature is not None:
            self.draft_model.generation_config.temperature = self.args.temperature
            self.target_model.generation_config.temperature = self.args.temperature
        if self.args.top_p is not None:
            self.draft_model.generation_config.top_p = self.args.top_p
            self.target_model.generation_config.top_p = self.args.top_p
        if self.args.top_k is not None:
            self.draft_model.generation_config.top_k = self.args.top_k
            self.target_model.generation_config.top_k = self.args.top_k

        vocab_size = min(self.draft_model.config.vocab_size, self.target_model.config.vocab_size)
        self.draft_model.config.vocab_size = vocab_size
        self.target_model.config.vocab_size = vocab_size
        self.same_tokenizer = self.target_model.config.get_text_config().vocab_size == self.draft_model.config.get_text_config().vocab_size

        # Manually resize lm_head if needed
        if hasattr(self.draft_model, "lm_head"):
            old_lm_head = self.draft_model.lm_head
            dtype = old_lm_head.weight.dtype
            self.draft_model.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(
                old_lm_head.weight.device, dtype=dtype)
            self.draft_model.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

        if hasattr(self.target_model, "lm_head"):
            old_lm_head = self.target_model.lm_head
            dtype = old_lm_head.weight.dtype
            new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device,
                                                                                        dtype=dtype)
            with torch.no_grad():
                new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
                    old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]
            self.target_model.lm_head = new_lm_head

        if torch.cuda.is_available() and int(self.model_size[:-1]) > 14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1, offload_dir=None)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2, offload_dir=None)

        print("dispatch model finished")

        self.draft_model.eval()
        self.target_model.eval()

        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

        # load tokenizers
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
                sd+="_fast"
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
        if self.args.lenience<1:
            sd += f"_lenience_{self.args.lenience}"
        if self.args.eta_cutoff:
            sd += f"_eta{self.args.eta_cutoff}"
        if self.args.min_p:
            sd += f"_minp_{self.args.min_p}"
        if self.args.cascade:
            sd += "_cascade"
        if self.args.eta_spd:
            sd += f"_eta_spd{self.args.eta_spd}"
        if self.args.min_p_spd:
            sd += f"_minp_spd{self.args.min_p_spd}"
        if self.args.cos_lambda:
            sd += f"_coslambda_{self.args.cos_lambda}"
        if self.args.cos_mu:
            sd += f"cos_mu{self.args.cos_mu}"
        sd += f'{self.args.name}'
        return sd


def main():
    """
    Main function to run INCLUDE evaluation with speculative decoding.
    """
    start_time = time.time()
    hsd = HSD()
    
    # Run evaluation on all questions (no language separation)
    hsd()
    
    end_time = time.time()
    print(f"\nTotal run time: {(end_time - start_time)/60:.2f} minutes")
    return 


if __name__ == '__main__':
    main()

