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

```bash
# Reproduce the main Qwen2.5-72B/0.5B table (Table 1 of the paper):
bash examples/reproduce_table_main_results.sh

# Reproduce the EAGLE-3 + Llama-3.1-8B table:
bash examples/reproduce_eagle3.sh

# Reproduce Figure 1 (gap widens with task difficulty):
bash examples/reproduce_difficulty_trend.sh
```

Each script writes per-run JSONL files under `outputs/<benchmark>/`. Aggregate
into the paper's table format with:

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
