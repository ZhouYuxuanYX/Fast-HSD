import json
import os
import re
import sys
import time
from typing import Optional

import torch
from datasets import load_dataset


CURRENT_DIR = os.path.dirname(__file__)
MATH_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "MATH"))
if MATH_DIR not in sys.path:
    sys.path.insert(0, MATH_DIR)

from eval_math import MathEvaluator, effective_bandwidth_Bps  # noqa: E402


class AIMEEvaluator(MathEvaluator):
    def load_dataset(self):
        """Load and sample the AIME 2026 dataset."""
        dataset = load_dataset("PATH_TO_AIME_DATASET", split="train")
        self.dataset_name = "aime"

        total_samples = len(dataset)
        requested_samples = self.args.num_samples if self.args.num_samples is not None else total_samples
        start_index = max(0, self.args.sample_start or 0)
        end_index = total_samples if self.args.sample_end is None else min(self.args.sample_end, total_samples)
        end_index = min(end_index, start_index + requested_samples)
        self.num_samples = max(0, end_index - start_index)

        print(
            f"Total AIME samples: {total_samples}, "
            f"Sampling: {self.num_samples} (indices {start_index} to {max(start_index, end_index) - 1})"
        )

        sampled_indices = list(range(start_index, end_index))
        self.problem_ids = [dataset[i]["problem_idx"] for i in sampled_indices]
        self.problems = [dataset[i]["problem"] for i in sampled_indices]
        self.solutions = [str(dataset[i]["answer"]) for i in sampled_indices]
        self.levels = ["AIME"] * self.num_samples
        self.types = ["aime"] * self.num_samples
        self.ground_truths = [self.normalize_integer_answer(answer) for answer in self.solutions]

        print(f"Loaded {self.num_samples} samples from AIME dataset")

    def normalize_integer_answer(self, text: str) -> str:
        if text is None:
            return ""

        cleaned = str(text).strip().replace(",", "")
        cleaned = cleaned.replace("$", "")
        cleaned = re.sub(r"\s+", "", cleaned)

        integer_match = re.search(r"-?\d+", cleaned)
        if not integer_match:
            return cleaned

        try:
            return str(int(integer_match.group(0)))
        except Exception:
            return integer_match.group(0)

    def extract_answer_from_response(self, response: str) -> str:
        boxed_answer = self.extract_boxed_answer(response)
        if boxed_answer:
            return self.normalize_integer_answer(boxed_answer)

        final_answer_match = re.search(
            r"(?:final answer|answer is|therefore,? the answer is)[:\s]*([^\n\.]+)",
            response,
            flags=re.IGNORECASE,
        )
        if final_answer_match:
            return self.normalize_integer_answer(final_answer_match.group(1))

        number_matches = re.findall(r"-?\d+", response)
        if number_matches:
            return self.normalize_integer_answer(number_matches[-1])

        return ""

    def compare_answers(self, pred: str, gt: str) -> bool:
        if not pred or not gt:
            return False
        return self.normalize_integer_answer(pred) == self.normalize_integer_answer(gt)

    def build_prompt(self, problem: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. Solve the AIME problem step by step. "
                    "The final answer must be a single integer from 0 to 999, and you should present it in \\boxed{}."
                ),
            },
            {"role": "user", "content": problem},
        ]

    def generate_response(self, problem: str) -> str:
        messages = self.build_prompt(problem)
        input_text = self.tokenizer2.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
        input_length = input_ids.shape[1]
        embedding_device = self.draft_model.model.embed_tokens.weight.device
        input_ids = input_ids.to(embedding_device)

        start = time.time()
        if self.args.speculative:
            outputs = self.speculative_decoding(input_ids)
        else:
            if self.args.model == "target":
                outputs = self.target_model.generate(
                    input_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=True,
                    tokenizer=self.tokenizer2,
                    eta_cutoff=self.args.eta_cutoff,
                    min_p=self.args.min_p,
                )
            else:
                outputs = self.draft_model.generate(
                    input_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=True,
                    tokenizer=self.tokenizer1,
                    eta_cutoff=self.args.eta_cutoff,
                    min_p=self.args.min_p,
                )
        end = time.time()
        self.total_counts["time"].append(end - start)

        generated_tokens = outputs[0][input_length:]
        return self.tokenizer1.decode(generated_tokens, skip_special_tokens=True)

    def __call__(self):
        if self.args.debug:
            self.debug()
            return

        if self.args.k < 1:
            raise ValueError(f"k must be >= 1, got {self.args.k}")

        self.total_counts = {
            "draft_eval": [],
            "target_eval": [],
            "total_step": [],
            "sample_length": [],
            "step_back_probs": [],
            "p_i": [],
            "q_i": [],
            "time": [],
            "ids": [],
        }

        print("Starting AIME evaluation")
        self.BW = effective_bandwidth_Bps()

        os.makedirs(f"results/{self.args.name}/outputs/accuracy/", exist_ok=True)
        os.makedirs(f"results/{self.args.name}/outputs/efficiency/", exist_ok=True)
        os.makedirs(f"results/{self.args.name}/outputs/final_result/", exist_ok=True)

        acc_file = f"results/{self.args.name}/outputs/accuracy/{self.sd}.txt"
        self.progress = 0
        pass_correct_count = 0
        total_correct_attempts = 0
        per_problem_successes = []

        with open(acc_file, "w") as fd:
            for problem_id, problem, ground_truth, level, prob_type in zip(
                self.problem_ids, self.problems, self.ground_truths, self.levels, self.types
            ):
                self.progress += 1
                print(f"\nprogress: {self.progress}/{self.num_samples}")
                print(f"Problem ID: {problem_id}")
                print(f"Ground truth: {ground_truth}")

                fd.write("=" * 60 + "\n")
                fd.write(f"Problem ID: {problem_id}\n")
                fd.write(f"Problem: {problem}\n")
                fd.write(f"Level: {level}\n")
                fd.write(f"Type: {prob_type}\n")
                fd.write(f"Ground Truth: {ground_truth}\n")

                problem_successes = 0
                round_outputs = []

                for round_idx in range(self.args.k):
                    response = self.generate_response(problem)
                    is_correct, extracted_answer, gt = self.test_answer(response, ground_truth)

                    round_outputs.append(
                        {
                            "round": round_idx + 1,
                            "response": response,
                            "extracted_answer": extracted_answer,
                            "correct": is_correct,
                        }
                    )

                    if is_correct:
                        problem_successes += 1
                        total_correct_attempts += 1

                    print(
                        f"Round {round_idx + 1}/{self.args.k}: "
                        f"{'correct' if is_correct else 'incorrect'} | extracted={extracted_answer}"
                    )

                    fd.write("-" * 40 + "\n")
                    fd.write(f"Round: {round_idx + 1}/{self.args.k}\n")
                    fd.write(f"Model Response:\n{response}\n")
                    fd.write(f"Extracted Answer: {extracted_answer}\n")
                    fd.write(f"Correct: {is_correct}\n")

                per_problem_successes.append(problem_successes)
                problem_pass = problem_successes > 0
                if problem_pass:
                    pass_correct_count += 1

                running_pass = pass_correct_count / self.progress if self.progress > 0 else 0.0
                running_avg = total_correct_attempts / (self.progress * self.args.k)

                print(
                    f"Problem success count: {problem_successes}/{self.args.k} | "
                    f"Running pass@{self.args.k}: {pass_correct_count}/{self.progress} = {running_pass:.4f} | "
                    f"Running avg@{self.args.k}: {running_avg:.4f}"
                )

                fd.write(f"Problem Successes: {problem_successes}/{self.args.k}\n")
                fd.write(f"Pass: {problem_pass}\n\n")
                fd.flush()

                if self.progress % 10 == 0:
                    efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
                    with open(efficiency_file, "w") as f:
                        json.dump(self.total_counts, f)

            efficiency_file = f"results/{self.args.name}/outputs/efficiency/{self.sd}_total_counts_checkpoint.json"
            print(f"Saving to {efficiency_file}")
            with open(efficiency_file, "w") as f:
                json.dump(self.total_counts, f)

        pass_at_k = pass_correct_count / self.num_samples if self.num_samples > 0 else 0.0
        avg_at_k = total_correct_attempts / (self.num_samples * self.args.k) if self.num_samples > 0 else 0.0
        self.efficiency_analysis(efficiency_file)

        self.final_result.update(
            {
                "accuracy": pass_at_k,
                "pass_at_k": pass_at_k,
                "avg_at_k": avg_at_k,
                "k": self.args.k,
                "correct": pass_correct_count,
                "correct_attempts": total_correct_attempts,
                "total": self.num_samples,
                "num_samples": self.num_samples,
                "dataset": self.dataset_name,
                "target_model": self.target_model_name,
                "draft_model": self.draft_model_name,
                "gamma": self.args.gamma,
                "speculative": self.args.speculative,
                "seed": self.args.seed,
                "per_problem_successes": per_problem_successes,
            }
        )

        final_result_file = f"results/{self.args.name}/outputs/final_result/{self.sd}_final_result.json"
        with open(final_result_file, "w") as f:
            json.dump(self.final_result, f, indent=2)

        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of samples: {self.num_samples}")
        print(f"k: {self.args.k}")
        print(f"Random seed: {self.args.seed}")
        print(f"Target model: {self.target_model_name}")
        print(f"Draft model: {self.draft_model_name}")
        print(f"Gamma: {self.args.gamma}")
        print("-" * 60)
        print(f"Pass@{self.args.k}: {pass_at_k:.4f} ({pass_correct_count}/{self.num_samples})")
        print(f"Avg@{self.args.k}: {avg_at_k:.4f} ({total_correct_attempts}/{self.num_samples * self.args.k})")
        print(f"Block Efficiency: {self.final_result.get('Block Efficiency', 'N/A')}")
        print(f"Decoding Speed: {self.final_result.get('Decoding Speed', 'N/A')} tokens/s")
        print("=" * 60)
        print(f"Results saved to: {final_result_file}")


def main():
    evaluator = AIMEEvaluator()
    evaluator()


if __name__ == "__main__":
    main()