"""Shared base class for the four benchmark runners.

The pre-refactor layout had five copies of "argparse + load model + generate +
score" living in ``verification/src/*/eval_*.py``. This base class collapses
the shared scaffolding so each benchmark module only has to provide the
dataset-specific bits (loader, prompt formatter, scorer).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("fast_hsd.benchmarks")

__all__ = ["BenchmarkRecord", "BenchmarkEvaluator"]


@dataclasses.dataclass
class BenchmarkRecord:
    """One row of the per-run output JSONL."""

    question_id: str
    prompt: str
    response: str
    gold: Optional[str]
    correct: Optional[bool]
    # Speculative-decoding telemetry.
    accepted_tokens: int
    proposed_tokens: int
    decoding_seconds: float
    output_tokens: int

    @property
    def block_efficiency(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / max(1, (self.proposed_tokens // max(1, self.output_tokens)))

    @property
    def decoding_speed_tokens_per_sec(self) -> float:
        if self.decoding_seconds <= 0:
            return 0.0
        return self.output_tokens / self.decoding_seconds


class BenchmarkEvaluator:
    """Subclass-and-fill-in template for a benchmark runner.

    Subclasses override :meth:`load_questions`, :meth:`format_prompt`, and
    :meth:`score`. The driving loop (:meth:`run`) is concrete.
    """

    name: str = "unset"

    # ----- Methods subclasses must implement. ----------------------------------

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def format_prompt(self, question: Dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError

    def score(self, question: Dict[str, Any], response: str) -> Optional[bool]:  # pragma: no cover
        raise NotImplementedError

    # ----- Concrete driving loop. ----------------------------------------------

    def build_model(self, args):
        """Build and return ``(target_model, draft_model, tokenizer)``.

        The default implementation defers to ``fast_hsd.eagle.runner`` when
        ``args.use_eagle3`` is set, and to plain HF AutoModel construction
        otherwise. Subclasses can override for benchmark-specific quirks
        (e.g. INCLUDE's multilingual tokenizer config).
        """
        if args.use_eagle3:
            from fast_hsd.eagle import runner

            return runner.build(args)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        target = AutoModelForCausalLM.from_pretrained(args.target_model, trust_remote_code=True)
        draft = AutoModelForCausalLM.from_pretrained(args.draft_model, trust_remote_code=True)
        return target, draft, tokenizer

    def generate(self, target, draft, tokenizer, prompt: str, args, method_cfg) -> BenchmarkRecord:
        """Run speculative decoding for a single prompt.

        This is a *skeleton*. The actual call into the patched
        ``transformers.generation.utils`` happens here; in this refactor we
        defer to ``model.generate(...)`` and let the runtime patch handle the
        method-specific kwargs.
        """
        import torch

        method = method_cfg.get("method", "baseline")
        param = method_cfg.get("param")

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
        # Translate the unified --param into transformers' lossy-verification kwargs.
        # These names are the ones added by the patches; see transformers/generation/utils.py.
        if method == "lenience":
            gen_kwargs["lenience"] = float(param)
        elif method == "cos":
            gen_kwargs["cos_lambda"] = float(param)
        elif method == "speccascade":
            gen_kwargs["cascade"] = True
            gen_kwargs["min_p_spd"] = float(param)
        elif method == "min_p_sampling":
            gen_kwargs["min_p"] = float(param)
        elif method == "eta_sampling":
            gen_kwargs["eta_spd"] = float(param)
        elif method == "typical_sampling":
            gen_kwargs["eta_cutoff"] = float(param)

        inputs = tokenizer(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            output = target.generate(
                **inputs,
                assistant_model=draft,
                **gen_kwargs,
            )
        dt = time.perf_counter() - t0
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Telemetry (accepted/proposed) requires fields on the generation output;
        # the patched _speculative_sampling stores these on ``output`` when
        # ``return_dict_in_generate=True`` is set. The exact wiring is benchmark-
        # specific; this base class leaves the fields at -1 if unavailable.
        return BenchmarkRecord(
            question_id="",
            prompt=prompt,
            response=text,
            gold=None,
            correct=None,
            accepted_tokens=-1,
            proposed_tokens=-1,
            decoding_seconds=dt,
            output_tokens=int(output.shape[-1] - inputs["input_ids"].shape[1]),
        )

    def run(self, args, method_cfg) -> int:
        """Top-level entry point invoked by :func:`fast_hsd.cli.main`."""
        target, draft, tokenizer = self.build_model(args)
        questions = list(self.load_questions(args))
        logger.info("Loaded %d questions from %s", len(questions), self.name)

        out_dir = os.path.join(args.output_dir, self.name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.name}.jsonl")
        n_correct = 0
        n_total = 0

        with open(out_path, "w") as fh:
            for q in questions:
                prompt = self.format_prompt(q)
                rec = self.generate(target, draft, tokenizer, prompt, args, method_cfg)
                rec.question_id = str(q.get("question_id", q.get("id", n_total)))
                rec.gold = q.get("gold")
                rec.correct = self.score(q, rec.response)
                if rec.correct is not None:
                    n_correct += int(rec.correct)
                    n_total += 1
                fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                fh.flush()

        acc = n_correct / n_total if n_total > 0 else float("nan")
        logger.info("%s: wrote %s — accuracy %.2f%%", self.name, out_path, 100.0 * acc)
        return 0
