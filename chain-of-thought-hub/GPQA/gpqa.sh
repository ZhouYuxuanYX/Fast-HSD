

# Go to repo
cd /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA

# Ensure output folders exist (if scripts expect them)

python eval_speculative_decoding_llm_GPQA.py \
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
    2>&1 | tee results/logs/gpqa_QW25_72vs05_clever_default_lenience10_final.txt