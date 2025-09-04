model_names=(meta-llama/Llama-2-7b-hf)
tasks=(mbpp)
decoding_method=eta
eta_cutoffs=(0.512 0.023 0.002)
huggingface-cli login --token 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 --add-to-git-credential


for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=${model_name}
        for eta_cutoff in ${eta_cutoffs[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${eta_cutoff}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 1\
                --batch_size 1\
                --max_new_tokens 512\
                --eta_cutoff ${eta_cutoff}
        done
    done
done
