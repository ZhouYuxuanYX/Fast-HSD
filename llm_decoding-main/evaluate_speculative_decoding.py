import json
import sys
import torch
import os
# must set before importing transformers
os.environ['HF_HOME'] = '/root/autodl-tmp/cache'
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import math
import time
import json
from torch import nn
from tqdm import tqdm
from accelerate import dispatch_model
from fsd import (
    fsd_vec_decoding,
    topp_decoding,
    magic_decoding,
    fsd_decoding,
    contrastive_decoding3,
    mirostat_decoding,
    dola,
)
from multiprocessing import Process
import pandas as pd
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
import numpy as np

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):
        self.token_id_list = token_id_list
        self.stop_tag = None

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for _ in range(len(self.token_id_list)):
            stop_states = [
                np.array_equal(
                    self.token_id_list[_],
                    input_ids[i][-len(self.token_id_list[_]) :].detach().cpu().numpy(),
                )
                for i in range(input_ids.size(0))
            ]
            if self.stop_tag is None:
                self.stop_tag = stop_states
            else:
                self.stop_tag = [
                    self.stop_tag[i] or stop_states[i]
                    for i in range(len(self.stop_tag))
                ]
            if all(self.stop_tag):
                self.stop_tag = None
                return True
        return False


def args_parse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--infile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument(
        "--outfile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument('--backward', action='store_true', default=False)
    parser.add_argument('--clever', action='store_true', default=False)
    parser.add_argument('--blockwise', action='store_true', default=False)
    parser.add_argument('--recursive', action='store_true', default=False)
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--gamma', default=10, type=int, help='number of assisted tokens')
    parser.add_argument('--lenience',  default=1, type=float, help='lenience factor')
    parser.add_argument("--approxi", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--decoding_method", type=str, default="topp")
    parser.add_argument("--gpus_per_model", type=int, default=2)
    parser.add_argument(
        "--model_name_or_path", default="decapoda-research/llama-7b-hf", type=str
    )

    parser.add_argument(
        "--student_model_name_or_path",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )

    parser.add_argument("--fsd_alpha", default=0.4, type=float)
    parser.add_argument("--fsd_k", default=5, type=int)
    parser.add_argument("--fsd_n", default=2, type=int)

    parser.add_argument(
        "--fsd_d_alpha", default=0.6, type=float
    )  # 0.6 is not good, maybe lower alpha
    parser.add_argument("--fsd_d_k", default=5, type=int)
    parser.add_argument("--fsd_d_n", default=3, type=int)

    parser.add_argument("--magic_alpha", default=0.4, type=float)
    parser.add_argument("--magic_p", default=0.95, type=float)
    parser.add_argument("--magic_k", default=5, type=int)
    parser.add_argument("--magic_n", default=2, type=int)

    parser.add_argument("--cs_alpha", default=0.6, type=float)
    parser.add_argument("--cs_k", default=5, type=int)

    parser.add_argument(
        "--cd_alpha", default=0.1, type=float
    )  # same notation as original paper
    parser.add_argument("--cd_tt", default=1.0, type=float, help="teacher temperature")
    parser.add_argument("--cd_st", default=0.5, type=float, help="student temperature")
    parser.add_argument("--cd_ignore_prefix", default="yes", type=str)

    parser.add_argument(
        "--cd2_alpha", default=0.1, type=float
    )  # algorithm 1 in the paper cd improves...
    parser.add_argument("--cd2_tt", default=1.0, type=float, help="teacher temperature")
    parser.add_argument("--cd2_st", default=0.5, type=float, help="student temperature")

    parser.add_argument("--cd3_alpha", default=0.1, type=float)
    parser.add_argument("--cd3_beta", default=0.5, type=float)
    parser.add_argument(
        "--cd3_tt", default=1.0, type=float, help="teacher temperature"
    )  # algorithm 2 in the paper cd improves...
    parser.add_argument("--cd3_st", default=1.0, type=float, help="student temperature")

    parser.add_argument(
        "--topp_p", default=0.95, type=float, help="used for topp and topp2"
    )
    parser.add_argument("--typical_p", default=0.95, type=float)
    parser.add_argument("--epsilon_cutoff", default=3e-4, type=float)
    parser.add_argument("--eta_cutoff", default=3e-4, type=float)

    parser.add_argument("--beam_n", default=4, type=int)
    parser.add_argument("--diverse_beam_n", default=4, type=int)
    parser.add_argument("--diverse_beam_groups", default=2, type=int)
    parser.add_argument("--topk_k", default=1, type=int)

    parser.add_argument("--mirostat_tau", default=3.0, type=float)

    parser.add_argument("--dola_early_exit_layers", default=0, type=str)
    parser.add_argument("--dola_mature_layer", default=32, type=int)
    
    parser.add_argument("--begin_gpu", default=0, type=int)
    args = parser.parse_args()
    return args


def out_file(outfile_path, generation_lst):
    with open(outfile_path, "w", encoding="utf-8") as f:
        json.dump(generation_lst, f, indent=4)

    print(f"written to {outfile_path}")


def generate(rank, args):
    visible_devices = [
        str(rank * args.gpus_per_model + i + args.begin_gpu) for i in range(args.gpus_per_model)
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    model2_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"

    tokenizer2 = AutoTokenizer.from_pretrained(
        model2_name,
        # trust_remote_code=True
    )
    tokenizer1 = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        # trust_remote_code=True
    )

    tokenizer1.padding_side = "left"
    if tokenizer1.pad_token_id is None and tokenizer1.eos_token_id is not None:
        tokenizer1.pad_token_id = tokenizer1.eos_token_id
    else:
        tokenizer1.add_special_tokens({"pad_token": "<|endoftext|>"})

    if tokenizer1.eos_token_id is None and tokenizer1.pad_token_id is not None:
        tokenizer1.eos_token_id = tokenizer1.pad_token_id

    tokenizer2.padding_side = "left"
    if tokenizer2.pad_token_id is None and tokenizer2.eos_token_id is not None:
        tokenizer2.pad_token_id = tokenizer2.eos_token_id
    else:
        tokenizer2.add_special_tokens({"pad_token": "<|endoftext|>"})

    if tokenizer2.eos_token_id is None and tokenizer2.pad_token_id is not None:
        tokenizer2.eos_token_id = tokenizer2.pad_token_id



    model2 = AutoModelForCausalLM.from_pretrained(
        model2_name,
        device_map=None,
        # trust_remote_code=True,
    )


    model1 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
          device_map=None,
          # trust_remote_code=True,
          )

    model1.generation_config.num_assistant_tokens = args.gamma
    # otherwise the draft length will change dynamically
    model1.generation_config.assistant_confidence_threshold = 0
    model1.generation_config.temperature = 1
    model1.generation_config.top_k = 0
    model1.generation_config.top_p = 1

    model2.generation_config.num_assistant_tokens = args.gamma
    # otherwise the draft length will change dynamically
    model2.generation_config.assistant_confidence_threshold = 0
    model2.generation_config.temperature = 1
    model2.generation_config.top_k = 0
    model2.generation_config.top_p = 1

    vocab_size = min(model1.config.vocab_size, model2.config.vocab_size)
    model1.config.vocab_size = vocab_size
    model2.config.vocab_size = vocab_size
    same_tokenizer = model2.config.get_text_config().vocab_size == model1.config.get_text_config().vocab_size

    # just changing the config.vocab_size is not enough, RuntimeError: The size of tensor a (152064) must match the size of tensor b (151936) at non-singleton dimension 2
    # change output size too
    # Manually resize lm_head if needed
    if hasattr(model1, "lm_head"):
        old_lm_head = model1.lm_head
        model1.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device)
        model1.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

    if hasattr(model2, "lm_head"):
        old_lm_head = model2.lm_head
        dtype = old_lm_head.weight.dtype  # preserve dtype, likely torch.float16 or torch.int8 (for GPTQ)

        # Create new lm_head with correct dtype and device
        new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device,
                                                                                    dtype=dtype)

        # Copy existing weights if within bounds
        with torch.no_grad():
            new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
                old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]

        model2.lm_head = new_lm_head

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

    device_map1 = manual_device_map(model1)
    device_map2 = manual_device_map(model2)

    model1.eval()
    model2.eval()

    model1 = dispatch_model(model1, device_map=device_map1, offload_dir=None)
    model2 = dispatch_model(model2, device_map=device_map2, offload_dir=None)

    # Fix rotary embedding buffers that may still be on CPU
    def move_rotary_emb_to_device(model):
        # Get the device of the layer where rotary embedding is applied
        try:
            device = model.model.embed_tokens.weight.device
            if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
                model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
        except AttributeError:
            print("Could not move rotary_emb.inv_freq — structure may be different")

    move_rotary_emb_to_device(model1)
    move_rotary_emb_to_device(model2)


    prompt_lst = []

    with open(args.infile) as f:
        idx = 0
        for line in f.readlines():
            d = json.loads(line.strip())
            d["idx"] = idx
            prompt_lst.append(d)
            idx += 1

    print(f"the total number of prompts: {len(prompt_lst)}")
    prompt_lst = prompt_lst[rank :: args.num_processes]
    print(f"the total number of prompts for rank {rank}: {len(prompt_lst)}")
    #if os.path.exists(args.outfile + f"{rank}"):
    if False:
        generated = pd.read_json(args.outfile + f"{rank}", lines=True)
        remove_list = []
        for _ in range(len(prompt_lst)):
            #print(prompt_lst[_].keys())
            #print("check key")
            #print(generated.keys())
            #exit()

            if prompt_lst[_]["idx"] in generated["idx"].values and _ not in remove_list:
                remove_list.append(_)
        prompt_lst = [
            prompt_lst[_] for _ in range(len(prompt_lst)) if _ not in remove_list
        ]
    print(f"the total number of prompts for rank {rank} to generate: {len(prompt_lst)}")

    # generation_res = []
    
    #print("check prompt list")
    #print(prompt_lst)
    #exit()

    s = time.time()
    max_new_tokens = args.max_new_tokens


    total_counts = {"draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                  "step_back_probs":[], "p_i":[], "q_i":[], "hist_lengths": [], "time":[], "ids":[]}


    for start in tqdm(range(0, 10, args.batch_size), disable=rank != 0):
        stopping_criteria = StoppingCriteriaList()
        if start % 20 == 0 and rank == 0:
            print(f"rank {rank} has generated {start} prompts")
        cur_prompt_lst = prompt_lst[start : start + args.batch_size]
        prompt_text = [f"{x['instructions']}" for x in cur_prompt_lst]

        # use chat template to avoid generating strange strings with repetition penalty
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt_text[0]}
        ]
        input_text = tokenizer2.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer2(
            input_text, padding=True, add_special_tokens=True, return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"].to(model2.device)
        attention_mask = model_inputs["attention_mask"].to(model2.device)
        prompt_len = input_ids.size(1)
        args.max_new_tokens = min(max_new_tokens, args.max_length - prompt_len)
        if args.max_new_tokens < 0:
            generation_text = [""] * len(cur_prompt_lst)
            for prompt, generation in zip(cur_prompt_lst, generation_text):
                json_str = json.dumps(
                    {
                        "idx": prompt["idx"],
                        # "instructions": prompt["instructions"],
                        "completion": generation.strip(),
                    }
                )
                with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                    f.write(json_str + "\n")
            continue


        # outputs = model2.generate(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=args.max_new_tokens,
        #     top_k=args.topk_k,
        #     do_sample=True,
        #     stopping_criteria=stopping_criteria,
        # )
        start = time.time()
        outputs, counts = model2.generate(input_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=args.max_new_tokens,
                                          do_sample=True,
                                          assistant_model=model1,
                                          stopping_criteria=stopping_criteria,
                                          assistant_confidence_threshold=0,
                                          backward=args.backward,
                                          # recursive=args.recursive,
                                          assistant_tokenizer=tokenizer1 if not same_tokenizer else None,
                                          tokenizer=tokenizer1,
                                          return_probs=args.backward or args.blockwise,
                                          blockwise=args.blockwise,
                                          clever=args.clever,
                                          approxi=args.approxi,
                                          lenience=self.args.lenience
                                          )

        total_counts["draft_eval"].append(counts["draft_eval"])
        total_counts["sample_length"].append(counts["sample_length"])
        total_counts["target_eval"].append(counts["target_eval"])
        total_counts["p_i"].append(counts["p_i"])
        total_counts["q_i"].append(counts["q_i"])
        total_counts["hist_lengths"].append(counts["hist_lengths"])
        total_counts["step_back_probs"].append(counts["step_back_probs"])
        total_counts["total_step"].append(counts["total_step"])
        total_counts["ids"].append(counts["ids"])

        end = time.time()
        total_counts["time"].append(start - end)


        generation_text = tokenizer2.batch_decode(
            outputs[:, prompt_len:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )

        #print(generation_text)
        #exit()
        for prompt, generation in zip(cur_prompt_lst, generation_text):
            json_str = json.dumps(
                {
                    "idx": prompt["idx"],
                    # "instructions": prompt["instructions"],
                    "completion": generation.strip(),
                }
            )
            with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                f.write(json_str + "\n")

    t = time.time()

    sd = ""
    if args.blockwise:
        sd += "blockwise"
    else:
        sd += "backward" if args.backward else "naive"
        if args.recursive:
            sd += "_recursive"
        elif args.clever:
            sd += "_clever"
            if args.approxi:
                sd += "_approxi"


    sd += f"_gamma_{args.gamma}"

    task = args.infile.split("/")[2]
    print("time used: ", t - s)
    with open(f"{task}_{sd}_total_counts.json", "w") as f:
        json.dump(total_counts, f)


if __name__ == "__main__":
    args = args_parse()
    args.early_stop = True
    print(args)
    assert args.world_size % args.gpus_per_model == 0
    args.num_processes = args.world_size // args.gpus_per_model
    if os.path.exists(args.outfile):
        try:
            all_ret = pd.read_json(args.outfile, lines=True)
            all_ret = all_ret.drop_duplicates(subset=["idx"], keep="first")
            all_ret.reset_index(drop=True, inplace=True)
            all_ret = all_ret[all_ret["completion"] != ""]
            all_ret.reset_index(drop=True, inplace=True)
            input_file = pd.read_json(args.infile, lines=True)
            if len(all_ret) == len(input_file):
                print(f"{args.outfile} already generated.")
                sys.exit(0)
            else:
                print("some prompts are not generated, regenerate them.")
                for _ in range(args.num_processes):
                    if os.path.exists(args.outfile + f"{_}"):
                        os.remove(args.outfile + f"{_}")
                for _ in range(len(all_ret)):
                    to_write_id = all_ret.iloc[_]["idx"] % args.num_processes
                    with open(
                        args.outfile + f"{to_write_id}", "a", encoding="utf-8"
                    ) as f:
                        json_str = json.dumps(
                            {
                                "idx": int(all_ret.iloc[_]["idx"]),
                                # "instructions": all_ret.iloc[_]["instructions"],
                                "completion": all_ret.iloc[_]["completion"],
                            }
                        )
                        f.write(json_str + "\n")
        except:
            print("bad output file")
            sys.exit(0)

    process_list = []
    for i in range(args.num_processes):
        p = Process(target=generate, args=(i, args))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    all_ret = pd.DataFrame()
    for rank in range(args.num_processes):
        with open(args.outfile + f"{rank}", "r", encoding="utf-8") as f:
            all_ret = pd.concat(
                [all_ret, pd.read_json(f, lines=True)], ignore_index=True
            )
    print(all_ret)
    all_ret.sort_values(by="idx", inplace=True)
    all_ret.to_json(args.outfile, orient="records", lines=True, force_ascii=False)
    for rank in range(args.num_processes):
        os.remove(args.outfile + f"{rank}")
