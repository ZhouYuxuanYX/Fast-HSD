# EAGLE-3 path fix (refactor/full)

## Symptom

The refactored `fast-hsd-eval --use-eagle3` produced EAGLE-3 results that diverged
from the legacy `EAGLE/eagle/evaluation/gen_ea_answer_*.py` pipeline (notably on
decoding speed and acceptance length — the EAGLE-specific metrics).

## Root cause

Two bugs in the refactor's EAGLE-3 path:

1. **`base.py` never called `eagenerate`.** `BenchmarkEvaluator.generate()`
   unconditionally ran the plain-SD path:

   ```python
   output = target.generate(**inputs, assistant_model=draft, **gen_kwargs)
   ```

   For `--use-eagle3`, `runner.build()` returns the `EaModel` wrapper as *both*
   `target` and `draft`, so this invoked HF universal-assisted-decoding —
   **bypassing EAGLE-3's tree drafting (`EaModel.eagenerate`) and the
   lossy-verification kwargs entirely.** The legacy pipeline calls
   `model.eagenerate(input_ids, ..., lenience=, min_p=, eta=)`.

2. **`runner.build()` dropped the tree hyperparameters.** It omitted
   `total_token` / `depth` / `top_k`, so `EaModel.from_pretrained` fell back to
   its own defaults (`depth=7`) instead of the legacy gen-script defaults
   (`total_token=60, depth=5, top_k=10`) — a different draft tree, hence
   different acceptance/speed.

## Fix

### `fast_hsd/eagle/runner.py`
- Pass `total_token` / `depth` / `top_k` to `EaModel.from_pretrained`
  (defaults `60 / 5 / 10`, matching legacy), read from the new CLI flags.
- Use `torch_dtype=torch.float16` and `model.eval()` (parity with legacy).

### `fast_hsd/benchmarks/base.py`
- `generate()` now branches to a new `_generate_eagle()` when `args.use_eagle3`.
- `_generate_eagle()`:
  - Builds the chat-templated prompt, tokenizes with `add_special_tokens=False`
    (the chat template already injects BOS/headers — matches legacy).
  - Calls `model.eagenerate(input_ids, temperature, max_new_tokens, log=True,
    is_llama3=<sniffed from target-model name>, **lossy_kwargs)`.
  - Records per-step `step_stats = (accept_length, block_size)` into the
    `BenchmarkRecord` per-block fields, so block-efficiency / speed are reported.
- Method → `eagenerate` kwargs map (mirrors the plain-SD mapping):

  | `--method` | eagenerate kwarg |
  |---|---|
  | `baseline` | `lenience=1.0` |
  | `lenience` | `lenience=<param>` |
  | `speccascade` | `min_p=<param>` |
  | `typical_sampling` | `eta=<param>` |
  | `min_p_sampling` | `min_p_baseline=<param>` |
  | `eta_sampling` | `eta_baseline=<param>` |
  | `cos` | *(unsupported on the EAGLE path)* |

- `_summarize()` gained an EAGLE branch: EAGLE tree blocks have a **variable**
  size, so the fixed-`gamma` filter used for plain SD doesn't apply. For EAGLE,
  `block_efficiency = mean tokens committed per step` (= mean `accept_length+1`)
  and `decoding_speed = raw tokens/s`.

### `fast_hsd/cli.py`
- New flags: `--eagle-total-token` (60), `--eagle-depth` (5), `--eagle-top-k` (10).

## Verification

BFCL, 30 questions, **Llama-3.1-8B-Instruct + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`**,
`--method baseline`, `temperature 0.7`, `total_token=60 / depth=5 / top_k=10`,
same A40 GPU. Legacy run via `gen_ea_answer_llama3chat.py` (env `llm`) + scored
with `eval_bfcl_eagle.py`; latest via `fast-hsd-eval --use-eagle3` (env `fsd`).

| Metric | Legacy (main / `llm`) | Latest (refactor / `fsd`) |
|---|---|---|
| Accuracy | 86.67% (26/30) | 90.00% (27/30) |
| **Speed tok/s (warm)** | **46.48** | **46.06** |
| Speed tok/s (incl. cold q0) | 46.48 | 25.77 |
| Avg accepted / step | 2.402 | 2.251 |
| Block efficiency (acc/slots) | 0.349 | 0.329 |

**Warm decoding speed is essentially identical (46.5 vs 46.1 tok/s)** — EAGLE-3
tree drafting is now actually running. Residual differences:

- The latest's *raw* tok/s (25.77) is dragged down by a one-time cold first
  question (q0 ≈ 20 s of CUDA/EAGLE init). The legacy warms up 3× before timing,
  so the **warm** number (excluding q0) is the fair comparison.
- The ±1-question accuracy difference is within sampling noise at `temperature
  0.7`, plus a known prompt-role difference: the legacy puts the BFCL rules in
  the *user* turn with a generic system prompt, while the refactor's `bfcl.py`
  splits the rules into the *system* role.

## How to reproduce

**Latest (refactor, env `fsd`):**
```bash
fast-hsd-eval --benchmark bfcl --use-eagle3 --method baseline \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model  yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --temperature 0.7 --max-new-tokens 1024 --num-samples 30 \
  --eagle-total-token 60 --eagle-depth 5 --eagle-top-k 10 \
  --name eagle_bfcl_latest
```

**Legacy (main, env `llm`)** — `gen_ea_answer_llama3chat.py --bench-name bfcl
--num-choices 2 ...` then `python EAGLE/scripts/eval_bfcl_eagle.py <answers>.jsonl`.

### Gotchas observed while reproducing
- The legacy `gen_ea_answer_*.py` needs `shortuuid` and `fastchat`, absent from
  the `llm` env. Do **not** install them there — shim via `PYTHONPATH`
  (faithful reimpls of `shortuuid.uuid` and
  `fastchat.llm_judge.common.load_questions`).
- Legacy off-by-one: its choice loop is `for i in range(1, num_choices)`, so
  `--num-choices 1` generates **zero** outputs. Use `--num-choices 2` for one
  choice per question.
- An A40 (48 GB) is sufficient for the 8B EAGLE-3 pipeline.

## Known limitations / follow-ups
- The stdout summary line still prints the plain-SD label *"avg accepted tokens
  per full-gamma block"* on the EAGLE path; the value is correct (mean tokens
  per step) but the label is cosmetic.
- `cos` (collaborative-overshoot) is not wired into `eagenerate`.
- For strict byte-level parity with the legacy BFCL prompt, the refactor would
  need to stop splitting the BFCL rules into the system role on the EAGLE path.
