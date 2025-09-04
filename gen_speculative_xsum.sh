#/bin/bash
model_names=(Qwen_72B)
tasks=(xsum)
method=speculative
huggingface-cli login --token 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 --add-to-git-credential

cd /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/llm_decoding-main
echo "speculative"
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
source ~/.bashrc
source activate base
conda activate speculative
conda info
pip install hf_xet

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
                --speculative 
    done
done
