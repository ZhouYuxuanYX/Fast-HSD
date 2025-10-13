import re
import os

folder_name = '/storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA/results/eval/select'
for file in os.listdir(folder_name):
    if file.endswith('.txt'):
        file_name = os.path.join(folder_name, file)
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for i, line in enumerate(lines):
        if "MODEL_ANSWER" in line:
            model_match = re.search(r"MODEL_ANSWER:\s*([A-D])", line)
            model_ans = model_match.group(1) if model_match else None
            
            # Check for the "A:" line and the next line after that
            true_ans = None
            if i + 2 < len(lines) and re.match(r"^\s*A:\s*$", lines[i + 1]):
                match = re.match(r"^\s*([A-D])\s*$", lines[i + 2])
                if match:
                    true_ans = match.group(1)
                else:
                    continue
            
            results.append((model_ans, true_ans))
    # Print them nicely
    count = 0
    total_count = 0
    for model, true in results:
        if model is not None and true is not None and model == true:
            count += 1
        if model is not None and true is not None:
            total_count += 1

    score = count / total_count

    print(f"file name: {file} ||score: {score*100 :.2f} ")