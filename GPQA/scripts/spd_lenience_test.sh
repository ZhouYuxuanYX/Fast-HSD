

#!/bin/bash

# SPD lenience [0.2, 0.4, 0.6, 0.8, 1.0]
# using Qwen default settings
# target model: Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8
# draft model: Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8
# dataset: fingertap/GPQA-Diamond
# sample size: 99

cd /storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA

name="SPD_lenience_gpqa"

mkdir -p results/$name/logs
mkdir -p results/$name/outputs/efficiency
mkdir -p results/$name/outputs/accuracy
mkdir -p results/$name/outputs/final_result

for lenience in 0.2 0.4 0.6 0.8 1.0; do
    echo "Running experiment with lenience=$lenience"
    python eval_speculative_decoding_llm_GPQA.py \
        --name "${name}" \
        --lenience $lenience \
        --speculative \
        --gamma 10 \
        2>&1 | tee results/$name/logs/${lenience}_exp.txt
    echo "Completed lenience=$lenience"
done

echo "All experiments completed!"