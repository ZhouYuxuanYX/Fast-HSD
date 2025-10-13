#!/bin/bash
#SBATCH -J gsm8k-fastHSD
#SBATCH -Agts-rg1002                  # Charge account
#SBATCH -N1                              # Number of nodes and cores per node required
#SBATCH --gres=gpu:h200:1                # GPU type (A100) and number of GPUs
#SBATCH --cpus-per-task=8  # 8 cores per task
#SBATCH --time=7:00:00
#SBATCH -q embers
#SBATCH -o /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA/results/reportsReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=twang730@gatech.edu # E-mail address for notifications 
#SBATCH -o /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA/results/out/new_gsm8k-fastHSD-%j.out

source ~/.bashrc
conda activate /storage/home/hcoda1/8/twang730/p-rg1002-0/envs/spd_clean

# Go to repo
cd /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA

# Ensure output folders exist (if scripts expect them)

srun python eval_speculative_decoding_llm_GPQA.py \
    --name gpqa_QW25_72vs05_clever_default_lenience10_final \
    --model target \
    --clever \
    --backward \
    --lenience 1.0 \
    --speculative \
    --target-model Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8 \
    --draft-model Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 \
    --dataset "fingertap/GPQA-Diamond" \
    --temperature 0.7 \
    --top_p 0.8 \
    2>&1 | tee /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA/results/logs/gpqa_QW25_72vs05_clever_default_lenience10_final.txt