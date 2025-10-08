#!/bin/bash

# Directory containing the input jsonl files
INPUT_DIR="/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp3/humaneval"

# Directory where the script is located (where ev.py is)
SCRIPT_DIR="/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "Starting evaluation of all jsonl files in $INPUT_DIR"
echo "============================================================"

# Loop through all .jsonl files in the input directory
for input_file in "$INPUT_DIR"/*.jsonl; do
    # Check if files exist (in case no .jsonl files are found)
    if [ ! -f "$input_file" ]; then
        echo "No .jsonl files found in $INPUT_DIR"
        exit 1
    fi
    
    # Extract the filename without path and extension
    filename=$(basename "$input_file" .jsonl)

    # plus task name human_eval
    filename="${filename}_human_eval"
    
    # Create output filename by appending "_results.jsonl"
    output_file="${filename}_results.jsonl"
    
    echo "----------------------------------------"
    echo "Processing: $filename.jsonl"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    # Run the evaluation command
    python3 ev.py \
        --task_name human_eval \
        --load_generations_path "$input_file" \
        --metric_output_path "$output_file" \
        --allow_code_execution
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $filename.jsonl"
    else
        echo "✗ Error processing $filename.jsonl"
    fi
    
    echo ""
done

echo "============================================================"
echo "All evaluations completed!"
echo "Results saved in the current directory: $SCRIPT_DIR"

# List the generated result files
echo ""
echo "Generated result files:"
ls -la *_results.jsonl 2>/dev/null || echo "No result files found."
