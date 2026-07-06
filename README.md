<div align="center">

# Fast-HSD

**Unifying Lossy Verification in Speculative Decoding: Underlying Mechanisms and Empirical Pitfalls**

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8.20](https://img.shields.io/badge/python-3.8.20-blue.svg)](https://www.python.org)

</div>

> **TL;DR** — The zoo of "lossy" speculative-decoding verifiers collapses into just **two families**, and one of them looks better than it is because it's measured against the wrong baseline. Fast-HSD turns that analysis into runnable code and a four-benchmark harness that *exposes* the speed–quality trade-off instead of hiding it.

## Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
  - [Patch the installed transformers](#patch-the-installed-transformers)
- [Quick Start](#-quick-start)
  - [CLI flags](#cli-flags)
  - [Method argument reference](#method-argument-reference)
  - [Output layout](#output-layout)
  - [Benchmarks](#benchmarks)
- [Using the patches from your own code](#-using-the-patches-from-your-own-code)
- [Repository layout](#-repository-layout)
- [Testing](#-testing)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [News](#-news)
- [License](#-license)

## Overview

Fast-HSD is a **diagnostic framework for lossy verification in speculative decoding (SD)**.

Speculative decoding accelerates LLM inference by having a small *draft* model propose
tokens that a large *target* model then verifies. "Lossy" verification relaxes the
acceptance rule to accept more draft tokens — trading a little output quality for a lot
of speed. A growing zoo of such methods has appeared, each with its own framing and
hyperparameters, making them hard to compare.

## Highlights

- **One package, one CLI.** `pip install -e .` then `fast-hsd-eval --benchmark math --method lenience --param 0.4 ...` — no more copy-files-into-site-packages dance.
- **Acceptance rules as importable functions.** `fast_hsd.core.collaborative_verification` and `fast_hsd.core.truncation_verification` expose the paper's math as ~20 lines of NumPy/PyTorch each, ready to be cross-validated against your own implementation.
- **Auto symlink sync of vendored transformers patches.** `fast-hsd-eval` symlinks the four modified `transformers==4.46.3` files into your active env's `site-packages` at startup. Edits to the vendored copies are instantly live in every process, including downstream consumers (SGLang, Ray workers, SpecForge training) that bypass `install()`. Runtime monkey-patcher kept as a fallback.
- **Per-run output directory** with structured JSONL, human-readable responses, raw per-block SD telemetry, and a summary covering accuracy + block efficiency + decoding speed.
- **Reproduces every table in the paper from a single shell command.** `bash examples/reproduce_table_main_results.sh`.

## Installation

Fast-HSD pins `python==3.8.20` and `transformers==4.46.3` because the vendored
patches target that exact release. Both the symlink-sync script and the
runtime patcher refuse to run (with a clear warning) on any other version.

```bash
# 1. Create a fresh conda env.
conda create -n fsd python=3.8.20 -y
conda activate fsd

# 2. Install the package + dev tools.
git clone https://anonymous.4open.science/anonymize/Fast-HSD-E6AD
cd Fast-HSD
pip install -e ".[dev]"

# 3. GPTQ-quantized Qwen models need these (not pulled in by pyproject.toml
#    because the GPTQ stack is optional for the acceptance-math tests).
pip install optimum auto-gptq
```

To verify the install:

```bash
python -c "import transformers; print(transformers.__version__)"   # → 4.46.3
fast-hsd-eval --help
```

### Patch the installed transformers

The four modified transformers files under `transformers/` need to land inside
your env's `site-packages/transformers/`. There are three paths, in decreasing
order of preference:

**1. Symlink sync (default — recommended).** `fast-hsd-eval` calls
`fast_hsd.patches.sync` at startup, which symlinks the four vendored files
into the active env's `site-packages/transformers/`. Original files are saved
alongside as `<file>.fasthsd-orig` so `--restore` can undo cleanly. You can
also invoke the script directly:

```bash
# Target the env this python belongs to (default = sys.prefix).
python scripts/sync_transformers_patches.py

# Target a specific conda env.
python scripts/sync_transformers_patches.py --env /path/to/conda/envs/fsd

# Inspect status without changing anything.
python scripts/sync_transformers_patches.py --check

# Undo (restore the .fasthsd-orig backups).
python scripts/sync_transformers_patches.py --restore
```

Symlinks beat copying because edits to a vendored file (e.g. `transformers/
generation/utils.py`) are *immediately* live in every process — no need to
re-sync. Pass `--no-sync-patches` to `fast-hsd-eval` if you want to skip it
(e.g. read-only env).

**2. Runtime monkey-patch (fallback).** When sync can't write (read-only env,
permission denied, version mismatch), `fast-hsd-eval` falls back to
`fast_hsd.patches.install()`, which rebinds the patched symbols on the
already-imported `transformers` modules. Idempotent and version-checked.
Limitation: every process that uses the patched generation path has to call
`install()` itself, so downstream consumers (SGLang workers, Ray actors) must
import `fast_hsd.patches` before their first `model.generate(...)`.

**3. Manual copy (legacy).** Kept for reproducibility of older runs; not
recommended. Copy the four files yourself:

```
<env>/lib/python*/site-packages/transformers/generation/candidate_generator.py
<env>/lib/python*/site-packages/transformers/generation/logits_process.py
<env>/lib/python*/site-packages/transformers/generation/utils.py
<env>/lib/python*/site-packages/transformers/cache_utils.py
```

## 🚀 Quick Start

Every method ships with a small shell script in `examples/` that wraps the
unified `fast-hsd-eval` CLI with a representative hyperparameter value. Pick
the benchmark (`math`, `mbppplus`, `include`, `bfcl`) as the first positional
argument and the method hyperparameter as the second.

**Plain speculative decoding (Qwen2.5 72B / 0.5B):**

```bash
# Lossless baseline:
bash examples/sd_baseline.sh math

# Lossy collaborative verification:
bash examples/sd_lenience.sh       math   0.4    # lenience factor
bash examples/sd_cos.sh            math   0.4    # CoS lambda

# Lossy truncation-based verification:
bash examples/sd_speccascade.sh    math   0.5    # min-p threshold
bash examples/sd_typical_sampling.sh math 0.10   # eta cutoff (Medusa)

# True truncation baselines (target with truncation sampling + lossless SD):
bash examples/sd_min_p_sampling.sh math   0.5
bash examples/sd_eta_sampling.sh   math   0.10
```

**EAGLE-3 (Llama-3.1-8B + EAGLE3 draft, single GPU):**

```bash
bash examples/eagle3_baseline.sh         math
bash examples/eagle3_lenience.sh         math 0.4
bash examples/eagle3_speccascade.sh      math 0.5
bash examples/eagle3_typical_sampling.sh math 0.10
```

**Equivalent direct CLI form**, if you'd rather not go through the shell scripts:

```bash
fast-hsd-eval \
    --benchmark math \
    --method METHOD --param VALUE \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 --seed 0 \
    --gamma 10 \
    --max-new-tokens 2048 \
    --name "RUN_NAME"
```

### CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--benchmark` | (required) | `math` / `mbppplus` / `include` / `bfcl` |
| `--method` | (required) | `baseline` / `lenience` / `cos` / `speccascade` / `min_p_sampling` / `eta_sampling` / `typical_sampling` |
| `--param` | — | Method hyperparameter (see method-argument reference below) |
| `--config` | — | Optional `configs/methods/*.json`, overrides `--param` |
| `--target-model` | (required) | HF id or local path |
| `--draft-model` | (required) | HF id or local path |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-new-tokens` | `2048` | Hard cap on generated tokens per question |
| `--gamma` | `10` | Number of draft tokens per SD block; sets `num_assistant_tokens` and is the denominator for block-efficiency |
| `--num-samples` | all | Truncate the benchmark to the first *N* questions (smoke-test convenience) |
| `--question-file` | benchmark default | Override the JSONL path (`math` and `include` only) |
| `--seed` | `0` | Reproducibility — seeds `random`, `numpy`, `torch` |
| `--name` | (required) | Run name; used as the output sub-directory |
| `--output-dir` | `outputs` | Root output directory |
| `--use-eagle3` | off | Use EAGLE-3 draft (single-GPU pipeline) instead of plain SD |
| `--sync-patches` / `--no-sync-patches` | on | Symlink-sync the vendored transformers patches at startup |
| `--install-patches` / `--no-install-patches` | on | Fall back to runtime monkey-patcher if sync didn't run |

### Method argument reference

| Method (`--method`) | Hyperparameter (`--param`) | Sweep range in the paper |
|---|---|---|
| `baseline` | — | (omit `--param`) |
| `lenience` | lenience factor *l* ∈ (0, 1] | {0.2, 0.4, 0.6, 0.8} |
| `cos` | CoS lambda ∈ [0, 1] | {0.2, 0.4, 0.6, 0.8} |
| `speccascade` | min-p threshold ∈ [0, 1] | {0.1, 0.3, 0.5, 0.7, 0.9} |
| `min_p_sampling` | min-p threshold ∈ [0, 1] | {0.1, 0.3, 0.5, 0.7, 0.9} |
| `eta_sampling` | eta > 0 | {0.05, 0.10, 0.15, 0.20, 0.25} |
| `typical_sampling` | eta cutoff > 0 | {0.05, 0.10, 0.15, 0.20, 0.25} |

### Output layout

Each run lands in its own directory under `outputs/`:

```
outputs/
└── <benchmark>/
    └── <run_name>/
        ├── rows.jsonl       # one JSON row per question — structured records
        ├── responses.txt    # human-readable per-question dump (legacy parity)
        ├── efficiency.json  # raw per-block SD counts (legacy total_counts shape)
        └── summary.json     # accuracy + block efficiency + decoding speed
```

- **`rows.jsonl`** — fields: `question_id`, `prompt` (chat-templated), `response`, `gold`, `correct`, `extracted_answer`, `level`, `prob_type`; SD telemetry `accepted_tokens` / `proposed_tokens` / `decoding_seconds` / `output_tokens`; per-block lists `draft_eval_per_block` / `target_eval_per_block` / `sample_length_per_block` / `n_matched_per_block` / `total_step_per_block`.
- **`efficiency.json`** — `{"draft_eval": [[...per-block per-question...]], "target_eval": ..., "sample_length": ..., "total_step": ..., "n_matched": ..., "time": [...]}`. Same shape as the legacy `total_counts_checkpoint.json`, so offline analysis scripts written against the legacy artifact still work.
- **`summary.json`** — `accuracy`, `block_efficiency` (mean of per-block `sample_length` over full-gamma blocks), `decoding_speed` (`num_full_blocks / total_decode_time × gamma`), `tokens_per_second`, total wall/decode times, model + method metadata.

The final summary also gets printed to stdout at end of run:

```
============================================================
FINAL RESULTS SUMMARY — bfcl
============================================================
Run name        : baseline_bfcl_seed0
Method/param    : baseline/None
Target model    : Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8
Draft model     : Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8
Gamma           : 10
Seed / temp     : 0 / 0.7
Samples scored  : 177/200 (of 200)
------------------------------------------------------------
Accuracy (all-pass): 0.8850
Block efficiency   : 8.76 (avg accepted tokens per full-gamma block)
Decoding speed     : 6.89 tok/s (gamma-normalized)
Tokens/s           : 6.71
Total decode time  : 1945.47s
Total output tokens: 13046
Full-gamma blocks  : 1340
============================================================
Rows JSONL        : outputs/bfcl/baseline_bfcl_seed0/rows.jsonl
Summary JSON      : outputs/bfcl/baseline_bfcl_seed0/summary.json
============================================================
```

### Benchmarks

All four benchmarks are scored **in-process** — the CLI emits accuracy +
block efficiency + decoding speed at end of run; no offline pass needed.

| Benchmark | Default prompt source | Scoring | Notes |
|---|---|---|---|
| `math` | `EAGLE/eagle/data/competition_math/question.jsonl` (500 q's) | `\boxed{}` extraction + LaTeX-aware equivalence (`is_equiv`) | Math-assistant system prompt; gold pulled from `reference[0]` |
| `include` | `EAGLE/eagle/data/include/question.jsonl` (220 q's, multilingual MCQ) | Three-pass regex (`"answer is X"` → `"Answer: X"` → last `A-D`), random-guess on full miss (seeded from `--seed`) | Zero-shot CoT; bundled prompts already end with `"Answer: Let's think step by step."` |
| `bfcl` | `EAGLE/eagle/data/bfcl/question.jsonl` (200 q's) | `ast.parse` of the model's `[func(...)]` list + category-specific checker | Bundled prompt is split on `"\nQuestion:"` so BFCL rules + function schemas land in the chat-template *system* role; without the split, models add free-text preamble that breaks the AST parser |
| `mbppplus` | Prompts from bundled `EAGLE/eagle/data/mbppplus/question.jsonl`; `test_list` + `test_imports` fetched from HF `evalplus/mbppplus` on first `score()` (cached) | In-process `exec` sandbox with common imports pre-loaded; runs every assertion; 10-second SIGALRM timeout per problem | Set `--max-new-tokens 4000` (legacy default) for real runs |

## Using the patches from your own code

The Fast-HSD patches can be reused outside this repository — e.g. inside
SpecForge training pipelines or as a SGLang verification backend.

```python
# Path A: symlink the vendored files into your env once (preferred).
# Then ``import transformers`` everywhere gets the patched version with no
# per-process setup. The script is callable as both a CLI and a function.
from fast_hsd.patches.sync import sync
sync()  # default: sys.prefix; pass env="/path/to/env" to target another env.

import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(...)
draft = transformers.AutoModelForCausalLM.from_pretrained(...)
out = model.generate(
    **inputs,
    assistant_model=draft,
    lenience=0.4,                          # collaborative verification
    # or: min_p_spd=0.5, cascade=True      # truncation-based verification
    # or: cos_lambda=0.6
)
```

```python
# Path B: runtime monkey-patch — works without filesystem write access, but
# every process needs to call ``install()`` before its first ``generate()``.
from fast_hsd.patches import install
install()  # idempotent and version-checked

import transformers
...
```

For a Python-only check of the acceptance math (no GPU needed):

```python
from fast_hsd.core.collaborative_verification import lenience_accept_prob
from fast_hsd.core.truncation_verification import speccascade_accepts
```

## Repository layout

```
fast_hsd/
├── core/                       # Paper's acceptance rules (importable)
│   ├── collaborative_verification.py
│   ├── truncation_verification.py
│   └── acceptance.py           # unified dispatch
├── patches/                    # Patch installer
│   ├── sync.py                 # symlink-based (preferred)
│   └── __init__.py             # runtime monkey-patcher
├── benchmarks/                 # math, mbppplus, include, bfcl
│   ├── base.py                 # shared driving loop + output writers
│   ├── _math_scoring.py        # \boxed{} extraction + is_equiv
│   ├── _include_scoring.py     # 3-pass regex letter extraction
│   ├── _bfcl_scoring.py        # AST parser + category-specific checkers
│   └── _mbppplus_scoring.py    # in-process exec sandbox + Pass@1
├── eagle/                      # Thin shim over EAGLE/ fork
└── cli.py                      # `fast-hsd-eval` entry point

scripts/
├── sync_transformers_patches.py  # standalone wrapper around fast_hsd.patches.sync

configs/methods/*.json          # One per lossy-verification method
configs/models/*.json           # One per (target, draft) pair
examples/reproduce_*.sh         # One per paper table
tests/                          # Unit tests on acceptance rules (CPU-only)

EAGLE/                          # Vendored EAGLE fork (drafts the tokens)
transformers/                   # Source of the vendored patches
verification/                   # Legacy per-benchmark eval scripts (kept for reference)
```

## Testing

The acceptance-rule unit tests are CPU-only and run in seconds:

```bash
pytest tests/test_acceptance_rules.py -v
```

They verify the paper's mathematical content: that `lenience=1` reduces to the
lossless rule, that `cos_lambda=0` collapses to the draft, that SpecCascade with
threshold 0 accepts everything, and that the lenience overshoot ceiling holds.

## Acknowledgments

This codebase builds on [EAGLE](https://github.com/SafeAILab/EAGLE) (the
vendored draft-model implementation under `EAGLE/`) and is engineered in the
spirit of [SpecForge](https://github.com/sgl-project/SpecForge), the SGLang
team's training framework. We thank the authors of the methods we evaluate —
SpecCascade, Medusa, CoS, and the speculative-decoding lenience formulation —
for releasing high-quality reference implementations.


## License

Apache 2.0 — see [LICENSE](LICENSE).
