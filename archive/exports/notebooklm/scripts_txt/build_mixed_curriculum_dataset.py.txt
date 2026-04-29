from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from lojban_evolution.experiment import Problem, generate_dataset


def crystal_text(problem: Problem) -> str:
    trace_line = " ".join(problem.trace)
    return (
        "[MODE=CRYSTAL]\n"
        "You are a rigid symbolic reasoner.\n"
        "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
        f"QUESTION: {problem.prompt}\n"
        f"TRACE: {trace_line}\n"
        f"ANSWER: {problem.answer}"
    )


def fluid_text(problem: Problem) -> str:
    return (
        "[MODE=FLUID]\n"
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {problem.prompt}\n"
        f"Final answer: {problem.answer}"
    )


def mutate_prompt(prompt: str, rng: random.Random, level: int) -> str:
    out = prompt
    if level >= 1 and rng.random() < 0.25:
        out = f"Quick check: {out}"
    if level >= 2:
        swaps = [
            (" because ", " since "),
            (" while ", " as "),
            (" Who is what?", " Determine each identity."),
            (" Where does ", " Where would "),
        ]
        for src, dst in swaps:
            if src in out and rng.random() < 0.25:
                out = out.replace(src, dst, 1)
    if level >= 3 and rng.random() < 0.2:
        out = out + " Keep it concise."
    return out.strip()


def mutate_problem(problem: Problem, copy_idx: int, noise_level: int) -> Problem:
    rng = random.Random((problem.problem_id + 1) * 1_000_003 + copy_idx * 11939 + 31)
    return Problem(
        problem_id=problem.problem_id,
        prompt=mutate_prompt(problem.prompt, rng, noise_level),
        answer=problem.answer,
        trace=problem.trace,
    )


def dedupe(rows: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for row in rows:
        key = row["text"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def iter_problems(dataset_size: int, seeds: Sequence[int], copies_per_problem: int, noise_level: int) -> Iterable[Problem]:
    for seed in seeds:
        for p in generate_dataset(size=dataset_size, seed=seed):
            yield p
            for i in range(max(0, copies_per_problem - 1)):
                yield mutate_problem(p, copy_idx=i, noise_level=noise_level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed crystal/fluid curriculum dataset for a unified adapter.")
    parser.add_argument("--output", type=Path, default=Path("runs/lora_sft_dataset_mixed.jsonl"))
    parser.add_argument("--dataset-size", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    parser.add_argument("--copies-per-problem", type=int, default=2)
    parser.add_argument("--noise-level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=23)

    parser.add_argument("--crystal-prompt-w", type=float, default=0.15)
    parser.add_argument("--crystal-trace-w", type=float, default=2.0)
    parser.add_argument("--crystal-answer-w", type=float, default=4.0)
    parser.add_argument("--fluid-prompt-w", type=float, default=0.25)
    parser.add_argument("--fluid-trace-w", type=float, default=1.0)
    parser.add_argument("--fluid-answer-w", type=float, default=3.0)
    parser.add_argument("--fluid-ratio", type=float, default=1.0, help="relative number of fluid rows per crystal row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[dict] = []
    rng = random.Random(args.shuffle_seed)

    for p in iter_problems(
        dataset_size=args.dataset_size,
        seeds=args.seeds,
        copies_per_problem=args.copies_per_problem,
        noise_level=args.noise_level,
    ):
        rows.append(
            {
                "text": crystal_text(p),
                "mode": "crystal",
                "problem_id": int(p.problem_id),
                "prompt_loss_multiplier": float(args.crystal_prompt_w),
                "trace_loss_multiplier": float(args.crystal_trace_w),
                "answer_loss_multiplier": float(args.crystal_answer_w),
                "trace_anchor": "\nTRACE:",
                "answer_anchor": "\nANSWER:",
            }
        )

        if rng.random() <= float(args.fluid_ratio):
            rows.append(
                {
                    "text": fluid_text(p),
                    "mode": "fluid",
                    "problem_id": int(p.problem_id),
                    "prompt_loss_multiplier": float(args.fluid_prompt_w),
                    "trace_loss_multiplier": float(args.fluid_trace_w),
                    "answer_loss_multiplier": float(args.fluid_answer_w),
                    "trace_anchor": "",
                    "answer_anchor": "\nFinal answer:",
                }
            )

    rows = dedupe(rows)
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    n_crystal = sum(1 for r in rows if r["mode"] == "crystal")
    n_fluid = sum(1 for r in rows if r["mode"] == "fluid")
    print(f"Wrote: {args.output}")
    print(f"Examples: {len(rows)} (crystal={n_crystal}, fluid={n_fluid})")


if __name__ == "__main__":
    main()

