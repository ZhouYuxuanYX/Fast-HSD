### 1. Environment Setup
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

You may comment out these lines if you use default download link cache for huggingface.

```
os.environ['HF_HOME'] = '/root/autodl-tmp/cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
```


If you want to test the default setting (multi GPU).

**Tokenwise**
```
CUDA_VISIBLE_DEVICES=0 python3 eval_speculative_decoding_llm.py  --speculative --gamma 10 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee tokenwise_gamma10_tmp1_top1_leni1.txt
```

**Blockwise**
```
CUDA_VISIBLE_DEVICES=1 python3 eval_speculative_decoding_llm.py  --speculative --blockwise --gamma 10 --temperature 1.0 --top_p 1.0 2>&1 | tee blockwise_gamma10_tmp1_top1.txt
```

**Naive HSD**
```
CUDA_VISIBLE_DEVICES=2 python3 eval_speculative_decoding_llm.py --speculative --backward --gamma 10 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee hsd_gamma10_tmp1_top1_leni1.txt
```

**Fast HSD**
```
CUDA_VISIBLE_DEVICES=3 python3 eval_speculative_decoding_llm.py  --speculative --backward --clever --approxi --gamma 10 --temperature 1.0 --top_p 1.0 --lenience 1.0 2>&1 | tee fasthsd_gamma10_tmp1_top1_leni1.txt
```

**🔧 Evaluation**
```
python compute_speculative_stats.py
```

**Variables**

Line 46 to change the target model size (14B, 32B, 72B)
```
    parser.add_argument('--target-model',  default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='must be complex or original')
```

### 4. Quick Start on CNN Daily and HumanEval Experiments

```
cd /root/autodl-tmp/llm_decoding-main
```

You may comment out this line if you use default cache for huggingface.

```
os.environ['HF_HOME'] = '/root/autodl-tmp/cache'
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
bash gen_speculative_fasthsd_human_eval.sh
```

```
CUDA_VISIBLE_DEVICES=0
model_names=(Qwen_72B)
tasks=(human_eval)
lenience=1.0
method=speculative_fasthsd

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=${model_name}
        
        python3 evaluate_speculative_decoding.py \
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${method}_lenience_${lenience}.jsonl\
                --model ${model_path}\
                --gpus_per_model 4\
                --world_size 4\
                --batch_size 1\
		        --max_new_tokens 512\
                --speculative\
                --backward \
                --clever \
                --approxi \
                --lenience ${lenience}
    done
done

```


**🔧Evaluation**
```
python compute_speculative_stats.py
```

