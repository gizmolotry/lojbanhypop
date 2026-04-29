from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from lojban_evolution.experiment import Problem, generate_dataset


def format_training_text(prompt: str, symbolic_trace: Sequence[str], answer: str) -> str:
    trace_line = " ".join(symbolic_trace)
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
        f"QUESTION: {prompt}\n"
        f"TRACE: {trace_line}\n"
        f"ANSWER: {answer}"
    )


def mutate_prompt(prompt: str, rng: random.Random, noise_level: int) -> str:
    out = prompt

    if noise_level >= 1:
        if rng.random() < 0.35:
            out = f"Quick check: {out}"
        if rng.random() < 0.25:
            out = out + " Answer briefly."

    if noise_level >= 2:
        replacements = [
            (" while ", " as "),
            (" because ", " since "),
            (" Where does ", " Where would "),
            (" Who is what?", " Determine each identity."),
        ]
        for src, dst in replacements:
            if src in out and rng.random() < 0.3:
                out = out.replace(src, dst, 1)
        if rng.random() < 0.2:
            out = out.replace(".", ". ", 1)

    if noise_level >= 3:
        if rng.random() < 0.2:
            out = out.replace(" red ball ", " red object ", 1)
        if rng.random() < 0.2:
            out = out.replace(" blue ball", " blue object", 1)
        if rng.random() < 0.2:
            out = f"Task: {out}"

    return out.strip()


def generate_problems(
    size: int,
    seeds: Iterable[int],
    noise_level: int,
    copies_per_problem: int,
) -> List[Problem]:
    all_problems: List[Problem] = []
    for seed in seeds:
        problems = generate_dataset(size=size, seed=seed)
        all_problems.extend(problems)

    # Add lexical variety while preserving canonical trace/answer mapping.
    if copies_per_problem <= 1 and noise_level <= 0:
        return all_problems

    mutated: List[Problem] = []
    for p in all_problems:
        mutated.append(p)
        for copy_idx in range(max(0, copies_per_problem - 1)):
            rng = random.Random((p.problem_id + 1) * 100003 + copy_idx * 9176 + 23)
            mp = mutate_prompt(p.prompt, rng, noise_level=noise_level)
            mutated.append(
                Problem(
                    problem_id=p.problem_id,
                    prompt=mp,
                    answer=p.answer,
                    trace=p.trace,
                )
            )
    return mutated


def dedupe_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    unique: List[dict] = []
    for row in rows:
        key = row["text"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build large synthetic SFT dataset for LoRA training.")
    parser.add_argument("--output", type=Path, default=Path("runs/lora_sft_dataset_synth.jsonl"))
    parser.add_argument("--dataset-size", type=int, default=2000, help="Problems per seed before mutation.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 11, 13],
        help="One or more seeds to diversify synthetic generation.",
    )
    parser.add_argument("--copies-per-problem", type=int, default=2, help="Lexical variants per problem.")
    parser.add_argument("--noise-level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--shuffle-seed", type=int, default=23)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    problems = generate_problems(
        size=args.dataset_size,
        seeds=args.seeds,
        noise_level=args.noise_level,
        copies_per_problem=args.copies_per_problem,
    )

    rows: List[dict] = []
    for p in problems:
        text = format_training_text(prompt=p.prompt, symbolic_trace=p.trace, answer=p.answer)
        rows.append(
            {
                "text": text,
                "source_eval": "synthetic_generator",
                "problem_id": int(p.problem_id),
                "mode": "synthetic_scale",
            }
        )

    rows = dedupe_rows(rows)

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote: {args.output}")
    print(f"Examples: {len(rows)}")


if __name__ == "__main__":
    main()

