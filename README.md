<div align="center">

# Fast-HSD

**Unifying Lossy Verification in Speculative Decoding: Underlying Mechanisms and Empirical Pitfalls**

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/ZhouYuxuanYX/Fast-HSD/actions/workflows/lint.yaml/badge.svg)](https://github.com/ZhouYuxuanYX/Fast-HSD/actions/workflows/lint.yaml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

</div>

Fast-HSD is a diagnostic framework for lossy verification in speculative decoding (SD).
The accompanying paper shows that the prior zoo of methods collapses into two families
— *truncation-based verification* (SpecCascade, Medusa typical-acceptance) and
*collaborative verification* (CoS, Lenience) — and surfaces an overlooked pitfall:
truncation-based methods can degrade quality far more than reported when measured
against the *true* truncation-sampling baseline. Fast-HSD ships the analysis as
runnable code and a four-benchmark harness designed to expose, rather than hide,
the speed–quality trade-off.

## Highlights

- **One package, one CLI.** `pip install -e .` then `fast-hsd-eval --benchmark math --method lenience --param 0.4 ...` — no more copy-files-into-site-packages dance.
- **Acceptance rules as importable functions.** `fast_hsd.core.collaborative_verification` and `fast_hsd.core.truncation_verification` expose the paper's math as ~20 lines of NumPy/PyTorch each, ready to be cross-validated against your own implementation.
- **Runtime monkey-patcher.** `from fast_hsd.patches import install; install()` rebinds the four modified `transformers==4.46.3` symbols in-process. No manual file copying. Idempotent and version-checked.
- **Reproduces every table in the paper from a single shell command.** `bash examples/reproduce_table_main_results.sh`.

## Install

```bash
git clone https://github.com/ZhouYuxuanYX/Fast-HSD.git
cd Fast-HSD
pip install -e ".[dev]"
```

Fast-HSD pins `transformers==4.46.3` because the vendored patches under
`transformers/` target that exact release. The runtime patcher will refuse to
install (with a clear warning) if a different version is active.

## Quick start

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

Each invocation writes one JSONL row per question to
`outputs/<benchmark>/<name>.jsonl`. The `<name>` field follows
`<method>_<value>_<bench>_seed<n>` so you can sweep over hyperparameters
without overwriting files.

**Equivalent direct CLI form**, if you'd rather not go through the shell scripts:

```bash
fast-hsd-eval \
    --benchmark math \
    --method METHOD --param VALUE \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 --seed 0 \
    --name "RUN_NAME"
```

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

Aggregate per-run JSONLs into a single table with:

```bash
python scripts/results_analysis.py outputs/math/*.jsonl
```

## Using the patches from your own code

The Fast-HSD patches can be reused outside this repository — e.g. inside
SpecForge training pipelines or as a SGLang verification backend:

```python
import fast_hsd_install
fast_hsd_install.install()                # one-time, idempotent

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

For a Python-only check of the acceptance math (no GPU needed):

```python
from fast_hsd.core.collaborative_verification import lenience_accept_prob
from fast_hsd.core.truncation_verification import speccascade_accepts
```

## Layout

```
fast_hsd/
├── core/                       # Paper's acceptance rules (importable)
│   ├── collaborative_verification.py
│   ├── truncation_verification.py
│   └── acceptance.py           # unified dispatch
├── patches/                    # Runtime monkey-patch installer
├── benchmarks/                 # math, mbppplus, include, bfcl
├── eagle/                      # Thin shim over EAGLE/ fork
└── cli.py                      # `fast-hsd-eval` entry point

configs/methods/*.json          # One per lossy-verification method
configs/models/*.json           # One per (target, draft) pair
examples/reproduce_*.sh         # One per paper table
tests/                          # Unit tests on acceptance rules (CPU-only)

EAGLE/                          # Vendored EAGLE fork (drafts the tokens)
transformers/                   # Source of the vendored patches
verification/                   # Legacy per-benchmark eval scripts (deprecated)
```

## Manual install (legacy)

The pre-refactor instructions are kept here for reproducibility of older runs.
**Prefer the runtime patcher above** in any new work.

1. Verify your installed `transformers` is exactly 4.46.3:
   `python -c "import transformers; print(transformers.__version__)"`.
2. Copy the four vendored files on top of the installed package:

```
<env>/lib/python*/site-packages/transformers/generation/candidate_generator.py
<env>/lib/python*/site-packages/transformers/generation/logits_process.py
<env>/lib/python*/site-packages/transformers/generation/utils.py
<env>/lib/python*/site-packages/transformers/cache_utils.py
```

## Testing

The acceptance-rule unit tests are CPU-only and run in seconds:

```bash
pytest tests/test_acceptance_rules.py -v
```

They verify the paper's mathematical content: that `lenience=1` reduces to the
lossless rule, that `cos_lambda=0` collapses to the draft, that SpecCascade with
threshold 0 accepts everything, and that the lenience overshoot ceiling holds.

## Citation

```bibtex
@inproceedings{zhou2026unifying,
  title={Unifying Lossy Verification in Speculative Decoding: Underlying Mechanisms and Empirical Pitfalls},
  author={Zhou, Yuxuan and Wang, Tianyu and Wu, Qifeng and Wu, Fengyi and Li, Heng and Xiao, Zikai and Wang, Wenbin and Shang, Junyuan and Cheng, Zhi-Qi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments

This codebase builds on [EAGLE](https://github.com/SafeAILab/EAGLE) (the
vendored draft-model implementation under `EAGLE/`) and is engineered in the
spirit of [SpecForge](https://github.com/sgl-project/SpecForge), the SGLang
team's training framework. We thank the authors of the methods we evaluate —
SpecCascade, Medusa, CoS, and the speculative-decoding lenience formulation —
for releasing high-quality reference implementations.

## News

- **2026-05**: Public refactor (`refactor/full`) — SpecForge-style packagization, runtime patcher, unified CLI, CI.
- **2026-04**: Preprint submitted to NeurIPS 2026.
- **2026-03**: Initial anonymous release at `anonymous.4open.science/r/Fast-HSD-E6AD/`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
