"""Shared base class for the four benchmark runners.

Per-run output layout (mirrors the legacy ``verification/src/*/eval_*.py``):

    outputs/<bench>/<name>.jsonl           # one row per question (structured)
    outputs/<bench>/<name>_responses.txt   # human-readable per-question dump
    outputs/<bench>/<name>_efficiency.json # raw per-block SD counts (for offline analysis)
    outputs/<bench>/<name>_summary.json    # accuracy + block efficiency + decoding speed

Block efficiency / decoding speed are computed with the legacy filter
(only blocks where ``draft_eval == gamma``), so numbers are comparable
across the old script and the new CLI.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    extracted_answer: Optional[str] = None
    level: Optional[str] = None
    prob_type: Optional[str] = None

    # Aggregated SD telemetry (sums across blocks).
    accepted_tokens: int = -1
    proposed_tokens: int = -1
    decoding_seconds: float = 0.0
    output_tokens: int = 0

    # Per-block SD telemetry — one entry per SD block. Names match the keys
    # the patched ``_speculative_sampling`` writes into ``counts``.
    draft_eval_per_block: List[int] = dataclasses.field(default_factory=list)
    target_eval_per_block: List[int] = dataclasses.field(default_factory=list)
    sample_length_per_block: List[int] = dataclasses.field(default_factory=list)
    total_step_per_block: List[int] = dataclasses.field(default_factory=list)
    n_matched_per_block: List[int] = dataclasses.field(default_factory=list)


class BenchmarkEvaluator:
    """Subclass-and-fill-in template for a benchmark runner.

    Subclasses override :meth:`load_questions`, :meth:`format_prompt`, and
    :meth:`score`. The driving loop (:meth:`run`) is concrete.

    Setting ``system_prompt`` causes :meth:`generate` to wrap each prompt
    with that system message via ``tokenizer.apply_chat_template``. Leave
    it ``None`` for raw-text prompting.
    """

    name: str = "unset"
    system_prompt: Optional[str] = None

    # ----- Methods subclasses must implement. ----------------------------------

    def load_questions(self, args) -> Iterable[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def format_prompt(self, question: Dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError

    def score(
        self, question: Dict[str, Any], response: str
    ) -> Optional[bool]:  # pragma: no cover
        """Return ``True``/``False``/``None`` for correctness, or a
        ``(correct, extracted_answer)`` tuple."""
        raise NotImplementedError

    # ----- Concrete driving loop. ----------------------------------------------

    def build_model(self, args):
        """Build and return ``(target_model, draft_model, tokenizer)``."""
        if args.use_eagle3:
            from fast_hsd.eagle import runner

            return runner.build(args)

        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, trust_remote_code=True, device_map="auto"
        )
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, trust_remote_code=True, device_map="auto"
        )

        # Clamp the two models to a shared vocabulary so universal-assisted
        # decoding's tokenizer check passes. Mirrors the legacy eval_*.py setup.
        vocab_size = min(target.config.vocab_size, draft.config.vocab_size)
        target.config.vocab_size = vocab_size
        draft.config.vocab_size = vocab_size
        for m in (target, draft):
            if hasattr(m, "lm_head") and m.lm_head.out_features != vocab_size:
                old = m.lm_head
                new = nn.Linear(old.in_features, vocab_size, bias=False).to(
                    old.weight.device, dtype=old.weight.dtype
                )
                with torch.no_grad():
                    new.weight[: min(old.out_features, vocab_size)] = old.weight[
                        : min(old.out_features, vocab_size)
                    ]
                m.lm_head = new

        gamma = int(getattr(args, "gamma", 10))
        for m in (target, draft):
            if hasattr(m, "generation_config"):
                m.generation_config.num_assistant_tokens = gamma
                m.generation_config.temperature = args.temperature

        target.eval()
        draft.eval()
        return target, draft, tokenizer

    def _wrap_with_chat_template(
        self, tokenizer, user_text: str, system_text: Optional[str] = None
    ) -> str:
        """Apply the model's chat template iff a system prompt is provided.

        ``system_text`` (when given) takes precedence over the class-level
        ``self.system_prompt``. Lets a benchmark pick a different system
        message per row (e.g. BFCL, where the system content varies with
        the per-row function schemas).
        """
        sys_msg = system_text if system_text is not None else self.system_prompt
        if not sys_msg:
            return user_text
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning("apply_chat_template failed (%s); falling back to raw text.", e)
            return user_text

    def generate(
        self,
        target,
        draft,
        tokenizer,
        prompt: str,
        args,
        method_cfg,
        system_prompt: Optional[str] = None,
    ) -> BenchmarkRecord:
        """Run speculative decoding for a single prompt.

        ``system_prompt`` overrides the class-level ``self.system_prompt``
        for this single call. Useful for benchmarks like BFCL where the
        system content varies per row.
        """
        import torch

        method = method_cfg.get("method", "baseline")
        param = method_cfg.get("param")

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
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

        full_prompt = self._wrap_with_chat_template(tokenizer, prompt, system_text=system_prompt)

        inputs = tokenizer(full_prompt, return_tensors="pt")
        device = next(target.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output = target.generate(
                **inputs,
                assistant_model=draft,
                **gen_kwargs,
            )
        dt = time.perf_counter() - t0

        # Patched generate returns ``(output_ids, counts)`` where ``counts``
        # is a dict of per-SD-block lists. Unpack defensively.
        counts: Optional[Dict[str, List[int]]] = None
        if isinstance(output, tuple):
            output, counts = output[0], output[1] if len(output) > 1 else None

        text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

        accepted = proposed = -1
        draft_eval_pb: List[int] = []
        target_eval_pb: List[int] = []
        sample_length_pb: List[int] = []
        total_step_pb: List[int] = []
        n_matched_pb: List[int] = []
        if isinstance(counts, dict):
            n_matched_pb = [int(x) for x in counts.get("n_matched", []) or []]
            sample_length_pb = [int(x) for x in counts.get("sample_length", []) or []]
            draft_eval_pb = [int(x) for x in counts.get("draft_eval", []) or []]
            target_eval_pb = [int(x) for x in counts.get("target_eval", []) or []]
            total_step_pb = [int(x) for x in counts.get("total_step", []) or []]
            accepted = sum(n_matched_pb) if n_matched_pb else -1
            proposed = sum(draft_eval_pb) if draft_eval_pb else -1

        return BenchmarkRecord(
            question_id="",
            prompt=full_prompt,
            response=text,
            gold=None,
            correct=None,
            accepted_tokens=accepted,
            proposed_tokens=proposed,
            decoding_seconds=dt,
            output_tokens=int(output.shape[-1] - input_len),
            draft_eval_per_block=draft_eval_pb,
            target_eval_per_block=target_eval_pb,
            sample_length_per_block=sample_length_pb,
            total_step_per_block=total_step_pb,
            n_matched_per_block=n_matched_pb,
        )

    # ----- Scoring shim -------------------------------------------------------

    @staticmethod
    def _unpack_score(rv) -> Tuple[Optional[bool], Optional[str]]:
        if rv is None:
            return None, None
        if isinstance(rv, tuple):
            correct = rv[0]
            extracted = rv[1] if len(rv) > 1 else None
            return (
                None if correct is None else bool(correct),
                None if extracted is None else str(extracted),
            )
        return bool(rv), None

    # ----- Top-level driving loop --------------------------------------------

    def run(self, args, method_cfg) -> int:
        target, draft, tokenizer = self.build_model(args)
        questions = list(self.load_questions(args))
        num_samples = getattr(args, "num_samples", None)
        if num_samples is not None and num_samples > 0:
            questions = questions[: int(num_samples)]
        logger.info("Loaded %d questions from %s", len(questions), self.name)

        # One subdirectory per run, mirroring the legacy
        # ``results/{name}/outputs/{accuracy,efficiency,final_result}/`` layout.
        out_dir = os.path.join(args.output_dir, self.name, args.name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "rows.jsonl")
        responses_path = os.path.join(out_dir, "responses.txt")
        efficiency_path = os.path.join(out_dir, "efficiency.json")
        summary_path = os.path.join(out_dir, "summary.json")

        records: List[BenchmarkRecord] = []
        n_correct = 0
        n_scored = 0
        t_total_start = time.perf_counter()

        with open(out_path, "w") as fh_jsonl, open(responses_path, "w") as fh_txt:
            for i, q in enumerate(questions):
                user_prompt = self.format_prompt(q)
                rec = self.generate(
                    target,
                    draft,
                    tokenizer,
                    user_prompt,
                    args,
                    method_cfg,
                    system_prompt=q.get("system_prompt"),
                )
                rec.question_id = str(q.get("question_id", q.get("id", i)))
                rec.gold = q.get("gold")
                rec.level = q.get("level")
                rec.prob_type = q.get("prob_type") or q.get("type")
                correct, extracted = self._unpack_score(self.score(q, rec.response))
                rec.correct = correct
                rec.extracted_answer = extracted

                if rec.correct is not None:
                    n_scored += 1
                    n_correct += int(rec.correct)

                fh_jsonl.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                fh_jsonl.flush()
                _write_response_block(fh_txt, rec, i + 1, len(questions))
                fh_txt.flush()
                records.append(rec)

                running_acc = (n_correct / n_scored) if n_scored else float("nan")
                logger.info(
                    "[%d/%d] qid=%s correct=%s extracted=%r acc=%.4f",
                    i + 1,
                    len(questions),
                    rec.question_id,
                    rec.correct,
                    rec.extracted_answer,
                    running_acc,
                )

        wall_seconds = time.perf_counter() - t_total_start
        summary = _summarize(records, args, method_cfg, n_correct, n_scored, wall_seconds)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(efficiency_path, "w") as f:
            json.dump(_efficiency_dump(records), f)

        _print_summary(self.name, summary, args, out_path, summary_path)
        return 0


# --- helpers ---------------------------------------------------------------


def _write_response_block(fh, rec: BenchmarkRecord, idx: int, total: int) -> None:
    sep = "=" * 60 + "\n"
    fh.write(sep)
    fh.write(f"[{idx}/{total}] question_id={rec.question_id}\n")
    if rec.level is not None:
        fh.write(f"Level: {rec.level}\n")
    if rec.prob_type is not None:
        fh.write(f"Type: {rec.prob_type}\n")
    fh.write(f"Ground Truth: {rec.gold}\n")
    fh.write(f"Extracted Answer: {rec.extracted_answer}\n")
    fh.write(f"Correct: {rec.correct}\n")
    fh.write(f"Decoding seconds: {rec.decoding_seconds:.2f}\n")
    fh.write(f"Output tokens: {rec.output_tokens}\n")
    fh.write(f"Model Response:\n{rec.response}\n")
    fh.write("\n")


def _efficiency_dump(records: List[BenchmarkRecord]) -> Dict[str, List[Any]]:
    """Per-question lists, matching the legacy ``total_counts`` shape."""
    return {
        "draft_eval": [r.draft_eval_per_block for r in records],
        "target_eval": [r.target_eval_per_block for r in records],
        "sample_length": [r.sample_length_per_block for r in records],
        "total_step": [r.total_step_per_block for r in records],
        "n_matched": [r.n_matched_per_block for r in records],
        "time": [r.decoding_seconds for r in records],
    }


def _summarize(
    records: List[BenchmarkRecord],
    args,
    method_cfg,
    n_correct: int,
    n_scored: int,
    wall_seconds: float,
) -> Dict[str, Any]:
    """Compute the legacy-style block efficiency / decoding speed.

    For each SD block where ``draft_eval == gamma`` (full-gamma block, not
    the residual tail), the block efficiency is the per-block accepted-tokens
    count (``sample_length`` in the patched code = n_matched + 1). The
    decoding speed is ``(num_full_blocks * gamma) / total_decoding_seconds``.
    Falls back gracefully when no SD telemetry is available.
    """
    gamma = int(getattr(args, "gamma", 10))

    full_block_accepts: List[int] = []
    total_decode_time = 0.0
    total_output_tokens = 0
    for r in records:
        total_decode_time += r.decoding_seconds
        total_output_tokens += r.output_tokens
        for de, sl in zip(r.draft_eval_per_block, r.sample_length_per_block):
            if de == gamma:
                full_block_accepts.append(sl)

    n_full_blocks = len(full_block_accepts)
    if n_full_blocks > 0 and total_decode_time > 0:
        block_efficiency = sum(full_block_accepts) / n_full_blocks
        decoding_speed = n_full_blocks / total_decode_time * gamma
    else:
        block_efficiency = float("nan")
        decoding_speed = float("nan")

    tokens_per_second = (
        total_output_tokens / total_decode_time if total_decode_time > 0 else float("nan")
    )
    accuracy = (n_correct / n_scored) if n_scored else float("nan")

    return {
        "name": args.name,
        "benchmark": getattr(args, "benchmark", None),
        "method": method_cfg.get("method"),
        "param": method_cfg.get("param"),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "gamma": gamma,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_samples": len(records),
        "num_scored": n_scored,
        "num_correct": n_correct,
        "accuracy": accuracy,
        "block_efficiency": block_efficiency,
        "decoding_speed": decoding_speed,
        "tokens_per_second": tokens_per_second,
        "total_decode_seconds": total_decode_time,
        "total_output_tokens": total_output_tokens,
        "total_wall_seconds": wall_seconds,
        "num_full_gamma_blocks": n_full_blocks,
    }


def _print_summary(bench: str, summary: Dict[str, Any], args, out_path: str, summary_path: str) -> None:
    bar = "=" * 60
    lines = [
        "",
        bar,
        f"FINAL RESULTS SUMMARY — {bench}",
        bar,
        f"Run name        : {summary['name']}",
        f"Method/param    : {summary['method']}/{summary['param']}",
        f"Target model    : {summary['target_model']}",
        f"Draft model     : {summary['draft_model']}",
        f"Gamma           : {summary['gamma']}",
        f"Seed / temp     : {summary['seed']} / {summary['temperature']}",
        f"Samples scored  : {summary['num_correct']}/{summary['num_scored']} (of {summary['num_samples']})",
        "-" * 60,
        f"Accuracy           : {summary['accuracy']:.4f}",
        f"Block efficiency   : {summary['block_efficiency']:.2f} (avg accepted tokens per full-gamma block)",
        f"Decoding speed     : {summary['decoding_speed']:.2f} tok/s (gamma-normalized)",
        f"Tokens/s           : {summary['tokens_per_second']:.2f}",
        f"Total decode time  : {summary['total_decode_seconds']:.2f}s",
        f"Total output tokens: {summary['total_output_tokens']}",
        f"Full-gamma blocks  : {summary['num_full_gamma_blocks']}",
        bar,
        f"Rows JSONL        : {out_path}",
        f"Summary JSON      : {summary_path}",
        bar,
        "",
    ]
    for line in lines:
        logger.info(line)
