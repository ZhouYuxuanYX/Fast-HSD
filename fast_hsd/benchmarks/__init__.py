"""Benchmark runners for the Fast-HSD paper.

Each module here exposes a single ``run(args, method_cfg)`` function that:

1. Loads the benchmark dataset (or expects a path passed via ``args``).
2. Builds the target + draft model pair (calling the EAGLE wrapper if
   ``args.use_eagle3`` is set).
3. Runs speculative decoding with the chosen lossy-verification method.
4. Writes one JSONL row per question to ``args.output_dir/<benchmark>/<name>.jsonl``.

The runners are thin: the actual algorithmic work lives in :mod:`fast_hsd.core`
and in the patched ``transformers`` module installed via :mod:`fast_hsd.patches`.
"""

from fast_hsd.benchmarks.base import BenchmarkEvaluator  # noqa: F401

__all__ = ["BenchmarkEvaluator"]
