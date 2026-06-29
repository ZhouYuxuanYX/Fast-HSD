"""Fast-HSD: Unified diagnostic framework for lossy verification in speculative decoding.

This package accompanies the paper "Unifying Lossy Verification in Speculative
Decoding: Underlying Mechanisms and Empirical Pitfalls" (NeurIPS 2026 preprint).

Public API
----------
- ``fast_hsd.core``: the paper's acceptance rules as importable functions.
- ``fast_hsd.patches.install()``: monkey-patches transformers==4.46.3 at runtime.
- ``fast_hsd.benchmarks``: shared Evaluator base class + per-benchmark drivers.
- ``fast_hsd.cli.main()``: ``fast-hsd-eval`` command-line entry point.
"""

__version__ = "0.1.0"

from fast_hsd import core, patches  # noqa: F401

__all__ = ["core", "patches", "__version__"]
