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

### Quick Start on experiments

```
cd autodl-tmp/chain-of-thought-hub-for-collaborators/gsm8k
```
        

        




