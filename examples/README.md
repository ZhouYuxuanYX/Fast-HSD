# Example invocations

One short shell script per lossy-verification method, ready to run. Each script
exercises the unified ``fast-hsd-eval`` CLI with concrete hyperparameter values
drawn from the paper's sweeps. Edit the ``BENCH`` and ``VALUE`` variables at
the top of each file to try other settings — the script's comment block lists
the legal range for the method.

## Plain speculative decoding (Qwen2.5 72B / 0.5B pair)

| Script | Method |
|---|---|
| ``sd_baseline.sh`` | Lossless speculative decoding |
| ``sd_lenience.sh`` | Lenience-based collaborative verification |
| ``sd_cos.sh`` | Collaborative Decoding via Speculation (CoS) |
| ``sd_speccascade.sh`` | SpecCascade truncation-based verification |
| ``sd_min_p_sampling.sh`` | Min-p truncation sampling + lossless SD |
| ``sd_eta_sampling.sh`` | Eta truncation sampling + lossless SD |
| ``sd_typical_sampling.sh`` | Medusa-style typical acceptance |

## EAGLE-3 (Llama-3.1-8B + EAGLE3 draft)

| Script | Method |
|---|---|
| ``eagle3_baseline.sh`` | EAGLE-3 + lossless verification |
| ``eagle3_lenience.sh`` | EAGLE-3 + lenience |
| ``eagle3_speccascade.sh`` | EAGLE-3 + SpecCascade |
| ``eagle3_typical_sampling.sh`` | EAGLE-3 + typical acceptance |

## Conventions

Every script writes one JSONL row per question to
``outputs/<benchmark>/<name>.jsonl``. ``<name>`` is set inside the script and
follows the format ``<method>_<value>_<bench>`` so you can sweep without
overwriting files.

To run every script in this directory:

```bash
for f in examples/sd_*.sh examples/eagle3_*.sh; do bash "$f"; done
```
