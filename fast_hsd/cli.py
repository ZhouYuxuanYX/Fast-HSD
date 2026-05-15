"""``fast-hsd-eval`` command-line entry point.

Replaces the previous workflow where users invoked one of five different
``verification/src/*/eval_*.py`` scripts depending on the benchmark. The new
shape is::

    fast-hsd-eval \\
        --benchmark math \\
        --method lenience --param 0.4 \\
        --target-model Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8 \\
        --draft-model Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 \\
        --name "lenience_0.4_seed0" \\
        --seed 0

The actual benchmark runners (loading data, running speculative decoding,
scoring) live in :mod:`fast_hsd.benchmarks`. This module is intentionally thin
so that adding a new benchmark only requires dropping a file in
``fast_hsd/benchmarks/`` — no CLI plumbing needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import import_module
from typing import Optional, Sequence

logger = logging.getLogger("fast_hsd")

# Method → (CLI flag name, default value if user just passes the method without
# a parameter). The grid sweeps used in the paper live in configs/methods/*.json.
METHODS = {
    "baseline": (None, None),
    "lenience": ("--param", 1.0),
    "cos": ("--param", 1.0),
    "speccascade": ("--param", 0.0),
    "min_p_sampling": ("--param", 0.0),
    "eta_sampling": ("--param", 0.0),
    "typical_sampling": ("--param", 0.0),
}

BENCHMARKS = ("math", "mbppplus", "include", "bfcl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast-hsd-eval",
        description=(
            "Run a lossy-verification benchmark from the paper "
            "'Unifying Lossy Verification in Speculative Decoding'."
        ),
    )
    p.add_argument(
        "--benchmark",
        required=True,
        choices=BENCHMARKS,
        help="Which benchmark to run.",
    )
    p.add_argument(
        "--method",
        required=True,
        choices=list(METHODS),
        help="Which lossy-verification method to use.",
    )
    p.add_argument(
        "--param",
        type=float,
        default=None,
        help="Method hyperparameter (e.g. lenience factor, cos lambda, min-p threshold).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config (e.g. configs/methods/lenience.json) — overrides --param.",
    )
    p.add_argument("--target-model", type=str, required=True)
    p.add_argument("--draft-model", type=str, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name. Used to construct the output filename.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory under which the per-run .jsonl file is written.",
    )
    p.add_argument(
        "--use-eagle3",
        action="store_true",
        help="Run the EAGLE-3 variant of the pipeline instead of plain SD.",
    )
    p.add_argument(
        "--install-patches",
        action="store_true",
        default=True,
        help="Call fast_hsd.patches.install() at startup. Default: enabled.",
    )
    p.add_argument(
        "--no-install-patches",
        dest="install_patches",
        action="store_false",
        help="Skip the runtime monkey-patch (use if you copied the patched "
        "transformers files manually).",
    )
    return p


def _load_benchmark(name: str):
    """Lazy-import the benchmark module so an unused benchmark's deps don't
    have to be installed."""
    return import_module(f"fast_hsd.benchmarks.{name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = build_parser().parse_args(argv)

    if args.install_patches:
        from fast_hsd.patches import install
        install(verbose=True)

    # Resolve method config.
    if args.config is not None:
        with open(args.config) as f:
            method_cfg = json.load(f)
    else:
        method_cfg = {"method": args.method, "param": args.param}

    logger.info("benchmark=%s method=%s cfg=%s", args.benchmark, args.method, method_cfg)

    bench = _load_benchmark(args.benchmark)
    rc = bench.run(args, method_cfg)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
