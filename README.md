### 1. Environment Setup
```
conda create -n xxx python=3.8
```
```
conda activate xxx
```

```
pip install transformers==4.46.3, datasets, optimum, auto-gptq
```



### 2. Replace essential files in transformers 
i. Make sure your transformers version is 4.46.3
    example file path: miniconda3/envs/vla/lib/python3.8/site-packages/transformers
ii. Then, replace the file under transformer packages.

    |__transformers
        |__generation
            |__candidate_generator.py
            |__utils.py   
            |__logits_process.py   
        |__cache_utils.py

### 3. Quick Start on Verification Experiments

All commands must be run from the **project root** (`Fast-HSD/`). Two GPUs are required per run (`CUDA_VISIBLE_DEVICES=0,1`). In each command below, replace `METHOD_FLAG`, `VALUE`, and `RUN_NAME` using the **method table** at the end of this section.

---

#### MATH

```bash
CUDA_VISIBLE_DEVICES=0,1 python verification/src/MATH/eval_math.py \
    --speculative \
    --METHOD_FLAG VALUE \
    --name "RUN_NAME"
```

---

#### MBPP+

```bash
CUDA_VISIBLE_DEVICES=0,1 python verification/src/mbppplus/eval_mbppplus.py \
    --speculative \
    --METHOD_FLAG VALUE \
    --name "RUN_NAME"
```

---

#### INCLUDE

`--dataset-dir` must point to a local copy of the [CohereLabs/include-base-44](https://huggingface.co/datasets/CohereLabs/include-base-44) dataset.

```bash
CUDA_VISIBLE_DEVICES=0,1 python verification/src/INCLUDE/run_include_qwen2.5_simple.py \
    --dataset-dir PATH_TO_INCLUDE_DATASET \
    --speculative \
    --temperature 0.7 \
    --METHOD_FLAG VALUE \
    --name "RUN_NAME"
```

---

#### BFCL

```bash
CUDA_VISIBLE_DEVICES=0,1 python verification/src/bfcl/eval_bfcl.py \
    --data-dir verification/src/bfcl/bfcl_data \
    --speculative \
    --METHOD_FLAG VALUE \
    --name "RUN_NAME"
```

---

#### Method argument table

Run once per `VALUE` in the listed set. `RUN_NAME` is a free-form label used in the output filename.

| Method | `METHOD_FLAG` | `VALUE` set |
|--------|--------------|-------------|
| SD Baseline (lossless) | *(omit — `--lenience` defaults to `1.0`)* | — |
| Lenience | `--lenience` | `0.2, 0.4, 0.6, 0.8` |
| CoS | `--cos_lambda` | `0.2, 0.4, 0.6, 0.8` |
| SpecCascade | `--min_p_spd` | `0.1, 0.3, 0.5, 0.7, 0.9` |
| Typical Sampling + SD | `--eta_spd` | `0.05, 0.10, 0.15, 0.20, 0.25` |
| Min-p Smpl. + SD | `--min_p` | `0.1, 0.3, 0.5, 0.7, 0.9` |
| η Sampl. + SD | `--eta_cutoff` | `0.05, 0.10, 0.15, 0.20, 0.25` |

### 4. Quick Start on EAGLE Experiments

All commands must be run from the **`EAGLE/`** directory (the root that contains the `eagle/` package and `scripts/`):
```
cd Fast-HSD/EAGLE
```

---

#### Step 1: Dataset Preparation

Run once before any experiments. Each script downloads the dataset from HuggingFace and writes a `question.jsonl` file into `eagle/data/<bench>/`.

| Benchmark | Command | Output | Size |
|-----------|---------|--------|------|
| **MATH** | `python scripts/prepare_competition_math.py` | `eagle/data/competition_math/question.jsonl` | 500 problems |
| **MBPP+** | `python scripts/prepare_mbppplus.py` | `eagle/data/mbppplus/question.jsonl` | 378 problems |
| **INCLUDE** | `python scripts/prepare_include.py --samples-per-language 5 --seed 42` | `eagle/data/include/question.jsonl` | 220 problems (5 × 44 languages) |
| **BFCL** | `python scripts/prepare_bfcl_questions.py` | `eagle/data/bfcl/question.jsonl` | 200 problems (`parallel_multiple` category) |

---

#### Step 2: Run Experiments

Each command writes one `.jsonl` per run to `<bench_name>/` (e.g. `competition_math/`, `mbppplus/`, `include/`, `bfcl/`) relative to the EAGLE root. Output filenames follow the pattern `<model_id>-temperature-<T>_<name>_<timestamp>.jsonl`. Run one command per hyperparameter value; replace `BENCH` with the desired benchmark and `VALUE` with each entry in the listed set.

**EAGLE3 Baseline** (`--lenience 1.0`):

```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --lenience 1.0 \
    --name "eagle_lenience1.0"
```

**EAGLE3 + Lossy Verification**:

*Lenience* (`--lenience` ∈ `{0.2, 0.4, 0.6, 0.8}`):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --lenience VALUE \
    --name "eagle_lenienceVALUE"
```

*SpecCascade* (`--min-p` ∈ `{0.1, 0.3, 0.5, 0.7, 0.9}`):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --min-p VALUE \
    --name "eagle_minpVALUE"
```

*Typical Acceptance* (`--eta` ∈ `{0.05, 0.10, 0.15, 0.20, 0.25}`):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --eta VALUE \
    --name "eagle_etaVALUE"
```

**EAGLE3 + Truncation-based Sampling** (draft-side truncation; EAGLE3 verification is unmodified):

*Min-p Smpl. + SD* (`--min-p-baseline` ∈ `{0.1, 0.3, 0.5, 0.7, 0.9}`):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --min-p-baseline VALUE \
    --name "eagle_minp_baselineVALUE"
```

*eta Sampling* (`--eta-baseline` ∈ `{0.05, 0.10, 0.15, 0.20, 0.25}`):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --use_eagle3 \
    --bench-name BENCH \
    --temperature 0.7 \
    --eta-baseline VALUE \
    --name "eagle_eta_baselineVALUE"
```

---

#### Step 3: Evaluation

Run from the **`EAGLE/`** directory. Pass a glob of the generated `.jsonl` files; each script prints per-file BE, decoding speed, and accuracy, plus a summary table when multiple files are provided.

**MATH**
```bash
python scripts/results_analysis.py competition_math/*.jsonl \
    --question-file eagle/data/competition_math/question.jsonl
```

**MBPP+** (executes generated code against the test suite)
```bash
python scripts/eval_mbppplus.py mbppplus/*.jsonl
```

**INCLUDE** (multiple-choice letter matching; reads `eagle/data/include/question.jsonl` by default)
```bash
python scripts/eval_include.py include/*.jsonl
```

**BFCL** (AST-based function-call matching; reads `eagle/data/bfcl/question.jsonl` by default)
```bash
python scripts/eval_bfcl_eagle.py bfcl/*.jsonl
```

Each evaluation script reports:
- **BE** — average accepted draft tokens per step (block efficiency)
- **DS** — decoding speed in tokens/s
- **Acc / Pass@1** — task accuracy (exact-match for MATH/INCLUDE, code execution for MBPP+, AST match for BFCL)

### Results

#### EAGLE-3 Results
EAGLE-3 speculative decoding results across four benchmarks (LLaMA-3.1 8B, temperature = 0.7, block size = 7).
**Bold** = best per column across all methods; <u>underline</u> = second best.

| Method | Param | MATH BE | MATH DS | MATH Acc (%) | MBPP+ BE | MBPP+ DS | MBPP+ Pass@1 (%) | INCLUDE BE | INCLUDE DS | INCLUDE Acc (%) | BFCL BE | BFCL DS | BFCL Acc (%) |
|--------|-------|--------:|--------:|-------------:|---------:|---------:|-----------------:|-----------:|-----------:|----------------:|--------:|--------:|-------------:|
| Baseline | --- | 3.76 | 140.21 | 73.20 | 4.70 | 167.83 | 59.30 | 0.67 | 45.72 | <u>35.50</u> | 2.57 | 85.27 | 86.00 |
| Lenience | 0.2 | 4.49 | 161.29 | 71.20 | 5.17 | <u>181.95</u> | <u>61.11</u> | 0.86 | 50.61 | 30.91 | 2.63 | **87.31** | 83.50 |
| Lenience | 0.4 | 4.34 | 157.03 | 76.20 | 5.02 | 177.13 | 59.79 | 0.80 | 48.93 | 32.73 | 2.61 | <u>86.14</u> | 85.50 |
| Lenience | 0.6 | 4.19 | 152.85 | 76.00 | 4.96 | 175.43 | 59.52 | 0.73 | 47.03 | 34.55 | 2.59 | 85.77 | 86.00 |
| Lenience | 0.8 | 4.07 | 149.39 | 75.20 | 4.92 | 174.25 | 59.79 | 0.72 | 46.73 | 33.64 | 2.59 | 85.56 | 84.00 |
| SpecCascade | 0.1 | 4.47 | 159.83 | 73.00 | 5.16 | 180.81 | 59.79 | 0.92 | 51.76 | 30.91 | 2.62 | 85.82 | 84.50 |
| SpecCascade | 0.3 | 4.28 | 154.42 | <u>77.40</u> | 5.03 | 176.79 | **61.38** | 0.79 | 48.26 | 31.82 | 2.59 | 85.18 | 84.50 |
| SpecCascade | 0.5 | 4.18 | 151.56 | **77.60** | 4.99 | 175.66 | 60.85 | 0.76 | 47.43 | 31.82 | 2.58 | 85.03 | 86.50 |
| SpecCascade | 0.7 | 4.12 | 149.71 | 76.20 | 4.93 | 173.70 | **61.38** | 0.73 | 46.74 | **36.82** | 2.58 | 84.99 | 85.50 |
| SpecCascade | 0.9 | 4.06 | 148.20 | 76.60 | 4.89 | 172.60 | 60.32 | 0.72 | 46.45 | 35.00 | 2.58 | 85.00 | 86.00 |
| Min-p Smpl. + SD | 0.1 | 3.89 | 140.72 | 75.20 | 4.84 | 166.79 | 61.90 | 0.54 | 45.34 | <u>40.00</u> | 2.58 | 84.45 | 86.00 |
| Min-p Smpl. + SD | 0.3 | 3.96 | 142.85 | **77.60** | 4.89 | 168.29 | 60.32 | 0.55 | 45.65 | 36.82 | 2.58 | 84.13 | 86.00 |
| Min-p Smpl. + SD | 0.5 | 3.98 | 143.42 | <u>77.40</u> | 4.91 | 168.98 | 60.85 | 0.56 | 45.91 | 35.91 | 2.58 | 84.09 | 86.50 |
| Min-p Smpl. + SD | 0.7 | <u>4.00</u> | <u>143.88</u> | <u>77.40</u> | <u>4.92</u> | <u>169.23</u> | 62.43 | 0.56 | <u>45.96</u> | 35.91 | 2.59 | 84.21 | **87.50** |
| Min-p Smpl. + SD | 0.9 | **4.05** | **145.24** | 76.80 | **4.93** | **169.58** | 63.49 | **0.57** | **46.13** | 36.36 | 2.59 | 84.42 | <u>87.00</u> |
| η Sampl. + SD | 0.05 | 3.85 | 137.92 | 75.20 | 4.84 | 164.79 | 59.79 | 0.55 | 45.42 | 35.00 | 2.58 | 83.63 | **87.50** |
| η Sampl. + SD | 0.10 | 3.90 | 139.33 | 77.00 | 4.86 | 165.32 | 61.11 | 0.54 | 45.06 | **41.82** | 2.59 | 83.92 | 86.00 |
| η Sampl. + SD | 0.15 | 3.93 | 140.21 | 77.20 | 4.88 | 165.95 | 61.90 | 0.54 | 45.13 | 35.00 | 2.59 | 83.91 | 86.50 |
| η Sampl. + SD | 0.20 | 3.96 | 140.97 | 76.00 | 4.89 | 166.23 | **64.29** | 0.54 | 45.14 | 34.09 | 2.57 | 83.54 | 85.00 |
| η Sampl. + SD | 0.25 | 3.99 | 141.83 | 75.20 | 4.88 | 165.77 | <u>63.76</u> | <u>0.57</u> | 45.92 | 37.73 | 2.58 | 63.85 | 86.50 |
| Typical Sampling | 0.05 | **4.82** | **168.77** | 66.00 | **5.29** | **182.86** | 55.29 | **1.24** | **59.80** | 29.55 | **2.66** | 86.11 | 78.00 |
| Typical Sampling | 0.10 | <u>4.66</u> | <u>164.19</u> | 68.20 | <u>5.21</u> | 180.54 | 56.08 | <u>1.15</u> | <u>57.20</u> | 29.55 | <u>2.65</u> | 85.85 | 79.50 |
| Typical Sampling | 0.15 | 4.58 | 161.86 | 72.20 | 5.16 | 178.87 | 57.41 | 1.05 | 54.76 | 25.91 | 2.64 | 85.65 | 83.00 |
| Typical Sampling | 0.20 | 4.54 | 160.65 | 69.80 | 5.16 | 178.94 | 58.47 | 0.98 | 52.73 | 27.73 | 2.63 | 85.40 | 83.50 |
| Typical Sampling | 0.25 | 4.46 | 158.17 | 73.00 | 5.17 | 179.31 | 57.14 | 1.00 | 53.21 | 26.82 | 2.62 | 85.05 | 83.00 |


#### Verification Results
Speculative decoding verification results across four benchmarks (Qwen2.5 0.5B draft → 72B target, temperature = 0.7). All entries are mean ± std over three seeds. **Bold** = best per column; <u>underline</u> = second best.

| Method | Param | MATH BE | MATH DS | MATH Acc (%) | MBPP+ BE | MBPP+ DS | MBPP+ Pass@1 (%) | INCLUDE BE | INCLUDE DS | INCLUDE Acc (%) | BFCL BE | BFCL DS | BFCL Acc (%) |
|--------|-------|--------:|--------:|-------------:|---------:|---------:|-----------------:|-----------:|-----------:|----------------:|--------:|--------:|-------------:|
| SD Baseline | — | 7.98±0.02 | 4.67±0.02 | 76.47±1.53 | 5.47±0.06 | 4.06±0.03 | 75.84±0.40 | 3.40±0.03 | 4.63±0.01 | 68.18±0.00 | 8.73±0.01 | 6.86±0.00 | 88.17±0.58 |
| Min-p Smpl. + SD | 0.1 | 7.94±0.02 | 4.65±0.02 | 76.27±1.53 | 5.42±0.08 | 3.89±0.14 | 75.57±0.40 | 3.34±0.04 | 4.63±0.01 | 68.79±0.69 | 8.73±0.01 | 6.97±0.18 | 88.17±0.58 |
| Min-p Smpl. + SD | 0.3 | 7.99±0.05 | 4.64±0.01 | 76.67±0.83 | 5.56±0.03 | 3.90±0.14 | 76.19±0.46 | 3.42±0.04 | 4.62±0.01 | 68.18±2.08 | 8.85±0.01 | 6.94±0.21 | 88.33±1.04 |
| Min-p Smpl. + SD | 0.5 | 8.03±0.02 | 4.64±0.01 | 76.07±1.17 | 5.45±0.07 | 3.89±0.14 | 75.57±0.31 | 3.44±0.02 | 4.61±0.02 | 68.48±1.46 | 8.88±0.02 | 6.97±0.19 | 87.67±0.29 |
| Min-p Smpl. + SD | 0.7 | 8.08±0.04 | 4.64±0.01 | 76.27±0.42 | 5.59±0.02 | 3.88±0.13 | 76.01±0.31 | 3.47±0.05 | 4.63±0.01 | 66.52±1.14 | 8.89±0.02 | 6.96±0.19 | 87.67±0.29 |
| Min-p Smpl. + SD | 0.9 | 8.08±0.10 | 4.65±0.01 | 77.27±1.27 | 5.62±0.04 | 3.88±0.14 | 76.01±0.15 | 3.47±0.04 | 4.62±0.01 | 66.97±1.31 | 8.89±0.01 | 6.97±0.18 | 88.00±0.00 |
| Cascade | 0.1 | 8.46±0.06 | 4.53±0.22 | 76.87±0.46 | 5.64±0.03 | 3.89±0.14 | 76.46±0.26 | 3.57±0.04 | 4.63±0.01 | 64.85±1.14 | 8.87±0.04 | 6.83±0.05 | 89.00±0.50 |
| Cascade | 0.3 | 8.36±0.04 | 4.65±0.01 | 75.47±0.70 | 5.59±0.04 | 3.89±0.14 | 75.40±0.26 | 3.50±0.03 | 4.60±0.01 | 66.67±3.44 | 8.89±0.01 | 6.81±0.02 | 88.67±0.29 |
| Cascade | 0.5 | 8.12±0.05 | 4.66±0.01 | 74.87±0.42 | 5.51±0.03 | 3.90±0.12 | 74.87±0.70 | 3.40±0.08 | 4.61±0.02 | 68.33±3.03 | 8.85±0.03 | 6.88±0.03 | 88.00±0.00 |
| Cascade | 0.7 | 8.01±0.06 | 4.66±0.01 | 75.47±0.70 | 5.48±0.01 | 3.90±0.14 | 76.46±0.53 | 3.30±0.04 | 4.60±0.01 | 68.33±1.05 | 8.84±0.03 | 6.82±0.03 | 87.67±0.29 |
| Cascade | 0.9 | 7.84±0.07 | 4.66±0.01 | 75.47±1.40 | 5.41±0.01 | 3.89±0.14 | 76.19±0.26 | 3.25±0.03 | 4.62±0.01 | 65.91±0.45 | 8.84±0.03 | 5.28±0.04 | 88.17±0.58 |
| η Smpl. + SD | 0.05 | 7.92±0.04 | 4.63±0.01 | 76.67±1.85 | 5.42±0.08 | 3.89±0.14 | 75.57±0.40 | 3.34±0.05 | 4.61±0.02 | 67.73±1.82 | 8.73±0.01 | 6.95±0.18 | 88.17±0.58 |
| η Smpl. + SD | 0.10 | 7.94±0.03 | 4.64±0.01 | 76.47±1.03 | 5.49±0.06 | 3.88±0.15 | 76.54±0.61 | 3.31±0.02 | 4.62±0.02 | 68.79±0.26 | 8.74±0.01 | 6.82±0.02 | 88.17±0.58 |
| η Smpl. + SD | 0.15 | 7.96±0.03 | 4.64±0.02 | 75.87±2.21 | 5.44±0.02 | 3.89±0.15 | 76.01±0.31 | 3.39±0.05 | 4.60±0.03 | 67.27±0.45 | 8.77±0.01 | 6.79±0.05 | 88.17±0.58 |
| η Smpl. + SD | 0.20 | 8.11±0.29 | 4.51±0.21 | 76.27±0.70 | 5.57±0.05 | 3.88±0.12 | 76.01±0.15 | 3.42±0.05 | 4.61±0.01 | 68.33±1.84 | 8.80±0.07 | 6.80±0.02 | 88.17±0.58 |
| η Smpl. + SD | 0.25 | 8.03±0.06 | 4.50±0.21 | 75.40±1.60 | 5.57±0.04 | 3.87±0.14 | 75.13±0.26 | 3.41±0.02 | 4.63±0.03 | 68.79±1.60 | 8.88±0.05 | 6.72±0.07 | 88.17±0.58 |
| Medusa | 0.05 | 8.49±0.01 | 4.67±1.93 | 76.60±2.08 | 5.64±0.03 | 3.89±0.14 | 76.46±1.06 | 3.58±0.03 | 4.62±0.01 | 66.52±1.31 | 8.87±0.04 | 6.84±0.02 | 89.00±0.50 |
| Medusa | 0.10 | 8.47±0.05 | 4.66±0.00 | 75.20±1.11 | 5.64±0.03 | 3.89±0.13 | 76.46±1.06 | 3.63±0.03 | 4.61±0.01 | 66.52±1.05 | 8.87±0.04 | 6.85±0.01 | 89.00±0.50 |
| Medusa | 0.15 | 8.21±0.24 | 4.65±0.03 | 75.73±2.12 | 5.66±0.03 | 3.88±0.14 | 75.66±0.53 | 3.59±0.01 | 4.62±0.02 | 66.06±1.31 | 8.87±0.04 | 6.85±0.03 | 89.00±0.50 |
| Medusa | 0.20 | 8.39±0.01 | 4.65±0.00 | 74.67±1.55 | 5.63±0.05 | 3.90±0.14 | 75.40±0.26 | 3.54±0.03 | 4.60±0.01 | 68.79±0.69 | 8.87±0.04 | 6.84±0.02 | 89.00±0.50 |
| Medusa | 0.25 | 8.30±0.05 | 4.67±0.01 | 75.53±0.31 | 5.55±0.08 | 3.90±0.14 | 75.04±0.15 | 3.46±0.01 | 4.63±0.01 | 66.06±0.26 | 8.89±0.01 | 6.78±0.04 | 88.67±0.29 |
| Lenience | 0.2 | 8.49±0.02 | 4.68±0.05 | 74.47±0.58 | 5.65±0.03 | 4.06±0.02 | 75.13±0.26 | 3.59±0.04 | 4.52±0.04 | 67.42±1.39 | 8.87±0.01 | 6.90±0.02 | 88.67±0.29 |
| Lenience | 0.4 | 8.37±0.08 | 4.66±0.12 | 75.60±2.65 | 5.61±0.02 | 4.06±0.02 | 75.49±0.67 | 3.55±0.06 | 4.53±0.09 | 67.88±1.31 | 8.85±0.04 | 6.87±0.01 | 88.00±0.50 |
| Lenience | 0.6 | 8.26±0.05 | 4.66±0.14 | 78.00±1.73 | 5.52±0.07 | 4.06±0.03 | 75.57±0.40 | 3.48±0.02 | 4.50±0.18 | 68.64±1.82 | 8.83±0.02 | 6.84±0.02 | 88.17±0.29 |
| Lenience | 0.8 | 8.22±0.07 | 4.65±0.23 | 76.47±1.15 | 5.52±0.04 | 4.06±0.02 | 75.75±0.40 | 3.38±0.03 | 4.49±0.11 | 68.33±0.26 | 8.81±0.05 | 6.87±0.02 | 87.67±0.29 |
| CoS | 0.2 | 8.33±0.00 | 4.68±0.01 | 71.53±2.12 | 6.05±0.13 | 4.10±0.01 | 71.43±1.06 | 3.89±0.04 | 4.61±0.02 | 64.55±1.98 | 8.76±0.13 | 6.79±0.04 | 70.67±3.40 |
| CoS | 0.4 | 8.80±0.04 | 4.67±0.00 | 60.40±2.84 | 7.14±0.09 | 4.13±0.01 | 61.20±3.29 | 4.69±0.07 | 4.58±0.03 | 53.79±2.10 | 8.84±0.15 | 6.69±0.03 | 58.00±4.50 |
| CoS | 0.6 | 9.42±0.01 | 4.67±0.01 | 54.80±0.72 | 7.86±0.02 | 4.20±0.02 | 59.44±0.85 | 5.99±0.03 | 4.53±0.01 | 48.03±1.89 | 9.29±0.16 | 6.69±0.03 | 43.83±3.79 |
| CoS | 0.8 | 10.14±0.11 | 4.65±0.01 | 46.00±2.12 | 9.26±0.04 | 4.26±0.00 | 54.06±0.31 | 8.02±0.05 | 4.49±0.01 | 37.12±2.05 | 10.26±0.07 | 6.75±0.01 | 43.67±1.89 |