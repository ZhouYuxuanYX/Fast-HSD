"""EAGLE-3 build helper used by the unified benchmark CLI.

This module collapses the eight per-chat-template ``gen_ea_answer_*.py``
scripts under ``EAGLE/eagle/evaluation/`` into one parameterized entry point.
The user picks the chat template via ``args.chat_template`` (or via the model
name, which we sniff for a default).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure ``EAGLE/`` is importable so we can pull in the vendored ea_model.
_EAGLE_DIR = _REPO_ROOT / "EAGLE"
if str(_EAGLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EAGLE_DIR))


__all__ = ["build"]


def build(args):
    """Build a wrapped EAGLE-3 model triple ``(target, draft, tokenizer)``.

    Parameters are read off ``args``:

    - ``args.target_model``: base model HF id / path.
    - ``args.draft_model``: EAGLE draft model HF id / path.
    - ``args.chat_template``: one of ``{'llama2', 'llama3', 'vicuna', 'mixtral',
      'qwen3', 'ds'}``. If not set, we sniff it from ``args.target_model``.
    """
    import torch
    from eagle.model.ea_model import EaModel  # type: ignore

    # EAGLE tree-search hyperparameters. Defaults match the legacy
    # gen_ea_answer_*.py argparse (total_token=60, depth=5, top_k=10); without
    # passing these, EaModel.from_pretrained falls back to its own defaults
    # (depth=7), which silently changes the draft tree and hence the
    # acceptance/speed numbers.
    model = EaModel.from_pretrained(
        base_model_path=args.target_model,
        ea_model_path=args.draft_model,
        total_token=int(getattr(args, "eagle_total_token", 60)),
        depth=int(getattr(args, "eagle_depth", 5)),
        top_k=int(getattr(args, "eagle_top_k", 10)),
        use_eagle3=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    # EAGLE's `EaModel` exposes both target and draft inside the same wrapper;
    # we return `(wrapper, wrapper, tokenizer)` so the BenchmarkEvaluator's
    # call signature stays uniform across the EAGLE and plain-SD pathways.
    return model, model, tokenizer
