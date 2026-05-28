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

    # Optional partial-credit sub-test counts (e.g. MBPP+ assertions passed /
    # total). When set, ``run`` aggregates them into a ``subtest_pass_rate`` in
    # the summary. ``correct`` stays the strict all-pass metric.
    num_subtests_passed: Optional[int] = None
    num_subtests_total: Optional[int] = None

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
        conf = float(getattr(args, "assistant_confidence_threshold", 0.0))
        for m in (target, draft):
            if hasattr(m, "generation_config"):
                m.generation_config.num_assistant_tokens = gamma
                # num_assistant_tokens_schedule="constant" keeps gamma fixed
                # rather than auto-tuning it per block.
                m.generation_config.num_assistant_tokens_schedule = "constant"
                # CRITICAL: HF defaults this to 0.4, which makes the draft
                # *early-stop* its proposal whenever per-token confidence drops
                # below the threshold — so blocks propose fewer than gamma
                # tokens. The legacy eval_*.py scripts (and the paper) set this
                # to 0 so the draft always proposes the full gamma. Leaving the
                # 0.4 default makes block-efficiency / decoding-speed numbers
                # diverge from the legacy runs (only ~65% of blocks reach gamma,
                # and the gamma-filtered DS metric badly undercounts).
                m.generation_config.assistant_confidence_threshold = conf
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

        # EAGLE-3 runs through EaModel.eagenerate (tree drafting), NOT HF's
        # assisted-generation ``generate(assistant_model=...)`` path. Route it
        # to the dedicated method.
        if getattr(args, "use_eagle3", False):
            return self._generate_eagle(target, tokenizer, prompt, args, method_cfg, system_prompt)

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

    # EAGLE method → eagenerate kwargs. Mirrors the plain-SD mapping: the
    # *_spd / verification params drive evaluate_posterior, while min_p / eta
    # "baseline" params drive the target's truncation sampling.
    _EAGLE_METHOD_KWARGS = {
        "baseline": {"lenience": 1.0},
        "lenience": {"lenience": "param"},
        "speccascade": {"min_p": "param"},
        "typical_sampling": {"eta": "param"},
        "min_p_sampling": {"min_p_baseline": "param"},
        "eta_sampling": {"eta_baseline": "param"},
    }

    def _generate_eagle(
        self, model, tokenizer, prompt, args, method_cfg, system_prompt
    ) -> BenchmarkRecord:
        """Generate one sample with EAGLE-3 via ``EaModel.eagenerate``.

        Mirrors the legacy ``gen_ea_answer_*.py`` call: chat-templated prompt,
        ``log=True`` telemetry, lossy-verification kwargs routed through
        ``eagenerate``. Per-step ``step_stats`` (accept_length, block_size)
        are stored so the summary can report EAGLE block efficiency / speed.
        """
        import torch

        method = method_cfg.get("method", "baseline")
        param = method_cfg.get("param")
        if method not in self._EAGLE_METHOD_KWARGS:
            raise ValueError(f"method {method!r} is not supported on the EAGLE-3 path")
        eg_kwargs = {
            k: (float(param) if v == "param" else v)
            for k, v in self._EAGLE_METHOD_KWARGS[method].items()
        }

        full_prompt = self._wrap_with_chat_template(tokenizer, prompt, system_text=system_prompt)
        # add_special_tokens=False — the chat template already injects BOS/headers
        # (matches the legacy gen_ea_answer_*.py tokenization).
        input_ids = tokenizer([full_prompt], add_special_tokens=False).input_ids
        input_ids = torch.as_tensor(input_ids).to(next(model.parameters()).device)
        input_len = input_ids.shape[1]

        is_llama3 = "llama-3" in args.target_model.lower() or "llama3" in args.target_model.lower()

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids, new_token, idx, step_stats = model.eagenerate(
                input_ids,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                log=True,
                is_llama3=is_llama3,
                **eg_kwargs,
            )
        dt = time.perf_counter() - t0

        gen_ids = out_ids[0][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # step_stats: list of (accept_length, block_size) per EAGLE step.
        n_matched_pb = [int(a) for a, _ in step_stats]
        block_size_pb = [int(b) for _, b in step_stats]
        # Per-step tokens committed = accepted drafts + 1 verified token.
        sample_length_pb = [a + 1 for a in n_matched_pb]
        return BenchmarkRecord(
            question_id="",
            prompt=full_prompt,
            response=text,
            gold=None,
            correct=None,
            accepted_tokens=sum(n_matched_pb) if n_matched_pb else -1,
            proposed_tokens=sum(block_size_pb) if block_size_pb else -1,
            decoding_seconds=dt,
            output_tokens=int(new_token),
            draft_eval_per_block=block_size_pb,
            target_eval_per_block=[1] * len(step_stats),
            sample_length_per_block=sample_length_pb,
            total_step_per_block=[1] * len(step_stats),
            n_matched_per_block=n_matched_pb,
        )

    # ----- Scoring shim -------------------------------------------------------

    @staticmethod
    def _unpack_score(rv) -> Tuple[Optional[bool], Optional[str], Optional[Tuple[int, int]]]:
        """Normalize a ``score()`` return value.

        Accepts ``None``, a bare ``bool``, ``(correct, extracted)``, or
        ``(correct, extracted, (n_passed, n_total))``. Returns the triple
        ``(correct, extracted, subtests)``.
        """
        if rv is None:
            return None, None, None
        if isinstance(rv, tuple):
            correct = rv[0]
            extracted = rv[1] if len(rv) > 1 else None
            subtests = rv[2] if len(rv) > 2 else None
            return (
                None if correct is None else bool(correct),
                None if extracted is None else str(extracted),
                tuple(subtests) if subtests is not None else None,
            )
        return bool(rv), None, None

    # ----- Top-level driving loop --------------------------------------------

    def run(self, args, method_cfg) -> int:
        target, draft, tokenizer = self.build_model(args)
        questions = list(self.load_questions(args))
        num_samples = getattr(args, "num_samples", None)
        if num_samples is not None and num_samples > 0:
            questions = questions[: int(num_samples)]
        logger.info("Loaded %d questions from %s", len(questions), self.name)

        # Auto-compose a canonical run id from method/param/bench/seed (and
        # eagle3 flag) so different hyperparameter values land in different
        # directories. ``args.name`` (when given) is appended as a tag so
        # users can label specific runs (e.g. "ablation1") without losing
        # the hyperparameter disambiguation.
        run_id = _canonical_run_id(args, method_cfg)
        # Stash the resolved id on args so summary.json/print_summary use it.
        args.name = run_id
        out_dir = os.path.join(args.output_dir, self.name, run_id)
        rows_existing = os.path.join(out_dir, "rows.jsonl")
        if (
            os.path.exists(rows_existing)
            and os.path.getsize(rows_existing) > 0
            and not getattr(args, "overwrite", False)
        ):
            raise SystemExit(
                f"Refusing to overwrite existing run at {out_dir!r}\n"
                f"(rows.jsonl is non-empty). Pass --overwrite to clobber, or "
                f"--name <tag> to land in a new directory."
            )
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
                correct, extracted, subtests = self._unpack_score(self.score(q, rec.response))
                rec.correct = correct
                rec.extracted_answer = extracted
                if subtests is not None:
                    rec.num_subtests_passed, rec.num_subtests_total = subtests

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


def _canonical_run_id(args, method_cfg) -> str:
    """Build the run id used as the output sub-directory name.

    Format: ``[eagle3_]<method>[_<param>]_<benchmark>_seed<seed>[_<name>]``.
    The ``--name`` tag (if any) is appended last so it acts as a label,
    never displacing the hyperparameter disambiguation.
    """
    parts: List[str] = []
    if getattr(args, "use_eagle3", False):
        parts.append("eagle3")
    method = method_cfg.get("method", "baseline")
    parts.append(str(method))
    param = method_cfg.get("param")
    if param is not None:
        parts.append(str(param))
    parts.append(str(getattr(args, "benchmark", "")))
    parts.append(f"seed{getattr(args, 'seed', 0)}")
    tag = getattr(args, "name", None)
    if tag:
        parts.append(str(tag))
    return "_".join(p for p in parts if p)


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
    is_eagle = bool(getattr(args, "use_eagle3", False))

    total_decode_time = 0.0
    total_output_tokens = 0
    for r in records:
        total_decode_time += r.decoding_seconds
        total_output_tokens += r.output_tokens

    if is_eagle:
        # EAGLE's tree blocks have a variable size, so the fixed-gamma filter
        # doesn't apply. Block efficiency = mean tokens committed per EAGLE
        # step (accepted drafts + 1); n_full_blocks here is the step count.
        all_step_tokens = [sl for r in records for sl in r.sample_length_per_block]
        n_full_blocks = len(all_step_tokens)
        block_efficiency = (sum(all_step_tokens) / n_full_blocks) if n_full_blocks else float("nan")
        # EAGLE has no gamma normalization; report raw throughput.
        decoding_speed = (
            total_output_tokens / total_decode_time if total_decode_time > 0 else float("nan")
        )
    else:
        full_block_accepts: List[int] = []
        for r in records:
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

    # Optional partial-credit (e.g. MBPP+ assertion-level pass rate). Only
    # emitted when at least one record carries sub-test counts.
    subtests_passed = sum(r.num_subtests_passed or 0 for r in records if r.num_subtests_total)
    subtests_total = sum(r.num_subtests_total or 0 for r in records if r.num_subtests_total)
    subtest_pass_rate = (subtests_passed / subtests_total) if subtests_total else None

    summary = {
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
        # Strict, problem-level metric: a problem counts only if it is fully
        # correct (for MBPP+, *all* assertions pass).
        "accuracy": accuracy,
        "block_efficiency": block_efficiency,
        "decoding_speed": decoding_speed,
        "tokens_per_second": tokens_per_second,
        "total_decode_seconds": total_decode_time,
        "total_output_tokens": total_output_tokens,
        "total_wall_seconds": wall_seconds,
        "num_full_gamma_blocks": n_full_blocks,
    }
    if subtest_pass_rate is not None:
        # Assertion-level partial-credit metric (MBPP+ "test pass rate").
        summary["subtest_pass_rate"] = subtest_pass_rate
        summary["subtests_passed"] = subtests_passed
        summary["subtests_total"] = subtests_total
    return summary


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
        f"Accuracy (all-pass): {summary['accuracy']:.4f}",
    ]
    if "subtest_pass_rate" in summary:
        lines.append(
            f"Test pass rate     : {summary['subtest_pass_rate']:.4f} "
            f"({summary['subtests_passed']}/{summary['subtests_total']} sub-tests)"
        )
    lines += [
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
