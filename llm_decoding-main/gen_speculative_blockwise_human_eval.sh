#/bin/bash
model_names=(Qwen_72B)
tasks=(human_eval)
method=speculative_blockwise

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=${model_name}
        
        python3 evaluate_speculative_decoding.py \
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${method}_gamma_10.jsonl\
                --model ${model_path}\
                --gpus_per_model 4\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --speculative\
       		    --blockwise	
    done
done
