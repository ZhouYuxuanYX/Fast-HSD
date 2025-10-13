import re
import os
# !!!!!!!!!!!!must set the environment variable before importing transformers, otherwise it won't work!!!!!!!!
######### use the local cache on haicore
# os.environ['HF_HOME'] = '/root/autodl-tmp/cache'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
import cProfile, pstats, io

import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time
import pandas as pd
import os
import glob
from typing import List, Tuple, Dict
import json


def argparse_setup():

    parser = argparse.ArgumentParser(prog='myprogram')
    parser.add_argument('--backward', action='store_true', default=False) # hsd framework
    parser.add_argument('--clever', action='store_true', default=False) # lossless
    parser.add_argument('--multidraft', type=int, default=1)

    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False) # lossy without cap
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--gamma',  default=10, type=int, help='number of assited tokens')
    parser.add_argument('--lenience',  default=1, type=float, help='lenience factor')
    parser.add_argument("--fast", action='store_true', default=False) # lossy with cap

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    

    parser.add_argument('--target-model', default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='must be complex or original')
    parser.add_argument('--draft-model', default='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', help='must be complex or original')
    parser.add_argument('--dataset', default="gsarti/flores_101", help='must be GPQA or MMLU')

    parser.add_argument('--model', help='must be target or draft', default="target")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='additional name to distinguish different runs')
    parser.add_argument('--cascade', action='store_true', default=False)
    
    # FLORES+ specific arguments
    parser.add_argument('--src-lang', type=str, default='eng', help='Source language code (e.g., eng, deu, fra)')
    parser.add_argument('--tgt-lang', type=str, default='deu', help='Target language code (e.g., deu, fra, spa)')
    parser.add_argument('--num-samples', type=int, default=400, help='Number of samples to evaluate')
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
    H   = getattr(cfg, "num_attention_heads")
    H_kv = getattr(cfg, "num_key_value_heads", H)  # GQA-aware
    D   = cfg.hidden_size
    Dh  = D // H
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
    import torch, time
    if not torch.cuda.is_available():
        # No CUDA: return NaN so you don't divide by zero; skip KV split on CPU/MPS
        return float("nan")
    x = torch.empty((1024,1024,512), device="cuda", dtype=torch.float16)  # ~1GB
    torch.cuda.synchronize(); t0 = time.time()
    _ = x.clone(); torch.cuda.synchronize(); t1 = time.time()
    return x.numel()*x.element_size()/(t1 - t0)

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

def compute_basic_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute basic translation metrics without external dependencies.
    """
    metrics = {}
    
    # Character-level accuracy (simple approximation)
    total_chars = 0
    correct_chars = 0
    
    # Word-level metrics
    total_words = 0
    correct_words = 0
    
    # Sentence-level exact matches
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        # Clean and normalize
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()
        
        # Exact match
        if pred_clean == ref_clean:
            exact_matches += 1
        
        # Character level
        total_chars += len(ref_clean)
        for i, char in enumerate(ref_clean):
            if i < len(pred_clean) and pred_clean[i] == char:
                correct_chars += 1
        
        # Word level
        pred_words = set(pred_clean.split())
        ref_words = set(ref_clean.split())
        total_words += len(ref_words)
        correct_words += len(pred_words.intersection(ref_words))
    
    metrics['exact_match_ratio'] = exact_matches / len(predictions) if predictions else 0
    metrics['character_accuracy'] = correct_chars / total_chars if total_chars > 0 else 0
    metrics['word_overlap_ratio'] = correct_words / total_words if total_words > 0 else 0
    
    return metrics

def parse_translation_results(filename: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse the FLORES translation result file and extract clean translations.
    
    Returns:
        Tuple of (source_sentences, predicted_translations, reference_translations)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by Q: to get individual examples
    examples = content.split('Q: ')[1:]  # Skip the first empty part
    
    source_sentences = []
    predicted_translations = []
    reference_translations = []
    
    for example in examples:
        lines = example.strip().split('\n')
        
        # Extract source sentence (first line after Q:)
        source = lines[0].strip()
        source_sentences.append(source)
        
        # Find the assistant's response (after "assistant")
        prediction = ""
        reference = ""
        
        in_assistant = False
        in_reference = False
        
        for line in lines:
            line = line.strip()
            
            if line == "assistant":
                in_assistant = True
                continue
            elif line.startswith("A:"):
                in_assistant = False
                in_reference = True
                reference = line[2:].strip()  # Remove "A:" prefix
                continue
            elif in_assistant and line and not line.startswith("A:"):
                prediction = line.strip()
                in_assistant = False
            elif in_reference and line and not line.startswith("Q:"):
                reference += " " + line.strip()
        
        predicted_translations.append(prediction)
        reference_translations.append(reference.strip())
    
    return source_sentences, predicted_translations, reference_translations

def compute_bleu_simple(predictions: List[str], references: List[str]) -> float:
    """
    Simple BLEU-like score implementation (1-gram precision).
    """
    total_precision = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        if not pred_words:
            continue
            
        matches = 0
        for word in pred_words:
            if word in ref_words:
                matches += 1
        
        precision = matches / len(pred_words) if pred_words else 0
        total_precision += precision
    
    return total_precision / len(predictions) if predictions else 0


class HSD():
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = self.args.draft_model
        print(f'target model: {self.target_model_name}')
        print(f'draft model: {self.draft_model_name}')
        self.model_size = self.target_model_name.split("/")[1].split("-")[1]
        print(f'model size: {self.model_size}')
        if float(self.model_size[:-1])>3:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        print(f'using device: {self.device}')

        # =======================Loading GPQA dataset=======================
        
        if self.args.dataset == 'fingertap/GPQA-Diamond':
            dataset = load_dataset(self.args.dataset)
            self.dataset_name = 'GPQA'
            self.dataset = dataset['test']
            self.num_samples = len(self.dataset['question']) //2 if int(self.model_size[:-1])>3 else len(self.dataset['question'])

        elif self.args.dataset == 'princeton-nlp/SWE-bench_Verified':
            dataset = load_dataset(self.args.dataset)
            self.dataset_name = 'SWE'
            self.dataset = dataset['test']
            self.num_samples = len(self.dataset['question']) //5 if int(self.model_size[:-1])>3 else len(self.dataset['question'])
            
        elif self.args.dataset == 'Maxwell-Jia/AIME_2024':
            dataset = load_dataset(self.args.dataset)
            self.dataset_name = 'AIME'
            self.num_samples = len(self.dataset['problem']) if int(self.model_size[:-1])>3 else len(self.dataset['question'])

        elif self.args.dataset == 'gsarti/flores_101':
            self.dataset_name = 'FLORES'
            
            # First try the standard FLORES-101 dataset (more reliable)
            print("Loading FLORES-101 dataset...")
            ds_all = load_dataset("gsarti/flores_101", name="all", split="devtest")
            src, tgt = self.args.src_lang, self.args.tgt_lang
            pairs = ds_all.map(lambda ex: {
                "src": ex[f"sentence_{src}"],
                "ref": ex[f"sentence_{tgt}"]
            }, remove_columns=[c for c in ds_all.column_names if not c.startswith("sentence_")])
            print(f"Successfully loaded FLORES-101 for {src} -> {tgt}")
        
            
            self.dataset = pairs
            # Use command line argument for number of samples
            self.num_samples = min(self.args.num_samples, len(pairs))
            print(f"Dataset loaded with {len(pairs)} total examples, using {self.num_samples} samples")

        else:
            print(f'Are you sure you want to run on other datasets? {self.args.dataset}')
            dataset = load_dataset(self.args.dataset)
            self.dataset_name = 'others'
            self.num_samples = len(self.dataset['question']) // 10 if int(self.model_size[:-1])>3 else len(self.dataset['question'])

        print(f"num_samples:{self.num_samples}")

        # =======================Model setup=================================
        self.model_setup()
        self.sd = self.test_setup()

        self.final_result = {'Block Efficiency': None, 'Decoding Speed': None}


    def speculative_decoding(self, input_ids):

        outputs, counts = self.target_model.generate(input_ids, max_new_tokens=512, do_sample=True,
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
                                cascade = self.args.cascade        
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
        gamma = 10

        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            count = orjson.loads(mm[:])   # mm[:] gives you a bytes object
            mm.close()

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
            len_ += len(sample_list)

        # Calculate per-step averages (same as results_analysis.py)
        draft_eval = draft/len_
        target_eval = target/len_
        total_step = step/len_
        sample_length = sample/len_  # block efficiency
        times = time_/len_

        print("Total decoding times:", times)

        # Fix speed calculation to match results_analysis.py
        # Speed should be: total_sample_length / total_time
        speed = sample / time_

        print(f"BE:{sample_length:.2f}")
        print(f"Speed:{speed:.2f}")
        print('---')
        self.final_result['Block Efficiency'] = f"{sample_length:.2f}"
        self.final_result['Decoding Speed'] = f"{speed:.2f}"


    def __call__(self):
        if self.args.debug:
            self.debug()
        else:
            self.total_counts = {"draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                "step_back_probs":[], "p_i":[], "q_i":[], "time":[], "ids":[]}
            
            print("start training")
            self.BW = effective_bandwidth_Bps()
            acc_file = f'results/{self.args.name}/outputs/accuracy/{self.sd}.txt'
            # num_samples

            self.progress = 0
            
            # Language mapping for better prompt generation
            lang_map = {
                'eng': 'English', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 
                'ita': 'Italian', 'por': 'Portuguese', 'rus': 'Russian', 'zho': 'Chinese',
                'jpn': 'Japanese', 'kor': 'Korean', 'ara': 'Arabic', 'hin': 'Hindi'
            }
            src_lang_name = lang_map.get(self.args.src_lang, self.args.src_lang)
            tgt_lang_name = lang_map.get(self.args.tgt_lang, self.args.tgt_lang)
            
            with open(acc_file, 'w') as fd:
                for src_text, ref_text in tqdm(zip(self.dataset['src'][:self.num_samples], self.dataset['ref'][:self.num_samples]),
                                total=self.num_samples):
                    print(f"progress: {self.progress}/{self.num_samples}")
                    self.progress+=1

                    prompt_q = f'''Please translate the following {src_lang_name} sentence into {tgt_lang_name}. Provide only the {tgt_lang_name} translation without any explanations or additional text.
{src_lang_name}: {src_text}
{tgt_lang_name}:'''

                    # use chat template to avoid generating strange strings with repetition penalty
                    messages = [
                        {"role": "system", "content": "You are a professional translator. Translate text accurately and concisely. Provide only the translation without explanations, comments, or additional formatting."},
                        {"role": "user", "content": prompt_q}
                    ]
                    input_text = self.tokenizer2.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids

                    embedding_device = self.draft_model.model.embed_tokens.weight.device
                    input_ids = input_ids.to(embedding_device)

                    start = time.time()

                    if self.args.speculative:
                        outputs = self.speculative_decoding(input_ids)
                    else:
                        if self.args.model == "target":
                            outputs = self.target_model.generate(input_ids, max_new_tokens=512, do_sample=True,
                                                            tokenizer=self.tokenizer2
                                                            )
                            used_model = self.target_model
                        else:
                            outputs = self.draft_model.generate(input_ids, max_new_tokens=512, do_sample=True,
                                                            tokenizer=self.tokenizer1
                                                            )
                            used_model = self.draft_model

                    end = time.time()
                    self.total_counts["time"].append(end-start)

                    ans_ = self.tokenizer1.decode(outputs[0], skip_special_tokens=True)
                    fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (src_text, ans_, ref_text))
                    fd.flush()  # Force write to disk immediately
                    if self.progress % 10 == 0:  # Save every 10 iterations
                        efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                        with open(efficiency_file, "w") as f:
                            json.dump(self.total_counts, f)
                    # print('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))
                efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts.json"
                print(f'saving to {efficiency_file}')
                with open(efficiency_file, "w") as f:
                    json.dump(self.total_counts, f)

            # Pass language codes for proper COMET evaluation
            src_lang_code = self.args.src_lang if hasattr(self.args, 'src_lang') else 'en'
            tgt_lang_code = self.args.tgt_lang if hasattr(self.args, 'tgt_lang') else 'de'

            sources, predictions, references = parse_translation_results(acc_file)
            metrics = compute_basic_metrics(predictions, references)
            metrics['BLEU_simple'] = compute_bleu_simple(predictions, references)

            self.efficiency_analysis(efficiency_file)
            self.final_result.update(metrics)


            final_result_file = f"results/{self.args.name}/outputs/final_result/{self.sd}_final_result.json"
            with open(final_result_file, "w") as f:
                json.dump(self.final_result, f)
                


    def model_setup(self):

        self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_name,
            device_map={"": self.device} if int(self.model_size[:-1])<32 else None)

        self.target_model = AutoModelForCausalLM.from_pretrained(self.target_model_name,
            device_map={"": self.device} if int(self.model_size[:-1])<32 else None)

        # @yx: use default qwen settings, uncomment when conducting ablation study.
        self.draft_model.generation_config.num_assistant_tokens = self.args.gamma
        # otherwise the draft length will change dynamically
        self.draft_model.generation_config.assistant_confidence_threshold = 0
        self.draft_model.generation_config.temperature = self.args.temperature
        self.draft_model.generation_config.top_k = self.args.top_k
        self.draft_model.generation_config.top_p = self.args.top_p

        self.target_model.generation_config.num_assistant_tokens = self.args.gamma
        # otherwise the draft length will change dynamically
        self.target_model.generation_config.assistant_confidence_threshold = 0
        self.target_model.generation_config.temperature = self.args.temperature
        self.target_model.generation_config.top_k = self.args.top_k
        self.target_model.generation_config.top_p = self.args.top_p

        vocab_size = min(self.draft_model.config.vocab_size, self.target_model.config.vocab_size)
        self.draft_model.config.vocab_size = vocab_size
        self.target_model.config.vocab_size = vocab_size
        self.same_tokenizer =  self.target_model.config.get_text_config().vocab_size == self.draft_model.config.get_text_config().vocab_size


        # just changing the config.vocab_size is not enough, RuntimeError: The size of tensor a (152064) must match the size of tensor b (151936) at non-singleton dimension 2
        # change output size too
        # Manually resize lm_head if needed
        if hasattr(self.draft_model, "lm_head"):
            old_lm_head = self.draft_model.lm_head
            dtype = old_lm_head.weight.dtype  # preserve dtype, likely torch.float16 or torch.int8 (for GPTQ)
            self.draft_model.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)
            self.draft_model.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

        if hasattr(self.target_model, "lm_head"):
            old_lm_head = self.target_model.lm_head
            dtype = old_lm_head.weight.dtype  # preserve dtype, likely torch.float16 or torch.int8 (for GPTQ)

            # Create new lm_head with correct dtype and device
            new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)

            # Copy existing weights if within bounds
            with torch.no_grad():
                new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
                    old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]

            self.target_model.lm_head = new_lm_head
        # redistribute after changing the layer, otherwise it won't work using "balanced" device map for multi-gpu context
        # Get a recommended device map first
        # device_map = infer_auto_device_map(model1, max_memory={i: "40GiB" for i in range(torch.cuda.device_count())})


        if torch.cuda.is_available() and int(self.model_size[:-1])>14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1, offload_dir=None)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2, offload_dir=None)

        print("dispatch model finished")

        self.draft_model.eval()
        self.target_model.eval()

        # Fix rotary embedding buffers that may still be on CPU
        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

        # load tokenizers
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.draft_model_name)
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.target_model_name)


        print('draft_model.config:')
        print(self.draft_model.config.to_json_string())                 # human-readable
        print('---')

        print('draft_model.generation_config:')
        print(self.draft_model.generation_config.to_json_string())
        print('---')

        print('draft_model.hf_device_map:')
        print(self.draft_model.hf_device_map)
        print('---')

        print('draft_model.parameters().dtype, draft_model.parameters().device:')
        print(next(self.draft_model.parameters()).dtype, next(self.draft_model.parameters()).device)
        print('---')

        print('tokenizer1.init_kwargs, tokenizer1.special_tokens_map:')
        print(self.tokenizer1.init_kwargs)
        print(self.tokenizer1.special_tokens_map)

        print('---')

        print('target_model.config:')
        print(self.target_model.config.to_json_string())
        print('---')

        print('target_model.generation_config:')
        print(self.target_model.generation_config.to_json_string())
        print('---')

        print('target_model.hf_device_map:')
        print(self.target_model.hf_device_map)
        print('---')

        print('target_model.parameters().dtype, target_model.parameters().device:')
        print(next(self.target_model.parameters()).dtype, next(self.target_model.parameters()).device)
        print('---')

        print('tokenizer2.init_kwargs, tokenizer2.special_tokens_map:')
        print(self.tokenizer2.init_kwargs)
        print(self.tokenizer2.special_tokens_map)

        print('---')


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
        if self.args.temperature<1:
            sd += f"_t{self.args.temperature}"
        if self.args.top_p <10:
            sd += f"_topp_{self.args.top_p}"
        if self.args.lenience<1:
            sd += f"_lenience_{self.args.lenience}"
        if self.args.cascade:
            sd += "_cascade"
        sd += f'{self.args.name}'
        return sd

    def debug(self):
        # Debug method placeholder
        pass


def main():
    hsd = HSD()
    hsd()

if __name__ == "__main__":
    main()
