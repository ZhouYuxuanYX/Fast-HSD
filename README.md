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

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

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

### Quick Start on GSM8K Experiments

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




