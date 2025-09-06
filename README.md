# Quick Start

## A. Make sure you have
1. chain-of-thought-hub-for-collaborators
2. llm_decoding-main
3. clever_single-draft
4. requirement.txt


## B. Environment building
### 1. Key requirement
CUDA 11.8
Python 3.8
Pytorch 2.1.0

```
conda create -n xxx python=3.8
```
```
conda activate xxx
```

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
pip install transformers==4.46.3
```

```
pip install datasets==2.21.0
```

```
pip install llmcompressor
```

```
pip install optimum
```

```
pip install auto-gptq
```

Other requirements, please refer to requirement.txt

### 2. Replace essential files in transformers 
i. Make sure your transformers version is 4.46.3
    example file path: miniconda3/envs/vla/lib/python3.8/site-packages/transformers
ii. Then, replace the file under transformer packages.

    |__transformers
        |__generation
            |__candidate_generator.py
            |__utils.py   
        |__cache_utils.py

### 3. Quick Start on GSM8K Experiments

```
cd autodl-tmp/chain-of-thought-hub-for-collaborators/gsm8k
```

If you want to test the default setting (multi GPU).


**Tokenwise**
```
CUDA_VISIBLE_DEVICES=0 python3 eval_speculative_decoding_llm.py  --speculative --gamma 1.0 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee tokenwise_gamma1_tmp1_top1_leni1.txt
```

**Blockwise**
```
CUDA_VISIBLE_DEVICES=1 python3 eval_speculative_decoding_llm.py  --speculative --blockwise --gamma 1.0 --temperature 1.0 --top_p 1.0 2>&1 | tee blockwise_gamma1_tmp1_top1.txt
```

**Naive HSD**
```
CUDA_VISIBLE_DEVICES=2 python3 eval_speculative_decoding_llm.py --speculative --backward --gamma 1.0 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee hsd_gamma1_tmp1_top1_leni1.txt
```

**Fast HSD**
```
CUDA_VISIBLE_DEVICES=3 python3 eval_speculative_decoding_llm.py  --speculative --backward --clever --approxi -gamma 1.0 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee fasthsd_gamma1_tmp1_top1_leni1.txt
```

**🔧Evaluation**
```
python compute_speculative_stats.py
```


### 4. Quick Start on CNN Daily and HumanEval Experiments

```
cd /root/autodl-tmp/llm_decoding-main
```



**Tokenwise**
```
bash gen_speculative_cnndailymail.sh
```

**Blockwise**
```
gen_speculative_blockwise_cnndailymail.sh
```

**Naive HSD**
```
bash gen_speculative_naivehsd_cnndailymail.sh
```

**Fast HSD**
```
bash gen_speculative_fashsd_human_eval.sh
```
```
CUDA_VISIBLE_DEVICES=0
model_names=(Qwen_72B)
tasks=(cnndailymail)
method=speculative_fasthsd

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=${model_name}
        
        python3 evaluate_speculative_decoding.py \
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${method}_lenience_1.jsonl\
                --model ${model_path}\
                --gpus_per_model 4\
                --world_size 4\
                --batch_size 1\
		        --max_new_tokens 512 \
                --speculative \
                --backward \
                --clever \
                --approxi \
                --lenience 1.0
    done
done
```


**🔧Evaluation**
```
python compute_speculative_stats.py
```

