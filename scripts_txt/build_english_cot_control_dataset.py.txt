from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from lojban_evolution.experiment import Problem, generate_dataset


def english_cot_text(problem: Problem) -> str:
    # Deterministic English chain-of-thought target built from the same supervision signal.
    steps = []
    for i, s in enumerate(problem.trace, start=1):
        s = str(s).replace("_", " ").strip()
        steps.append(f"Step {i}: {s}.")
    cot = " ".join(steps) if steps else "Step 1: reason over the stated constraints."
    return (
        "Solve the logic question using explicit reasoning.\n\n"
        f"Question: {problem.prompt}\n"
        f"Reasoning: {cot}\n"
        f"Final answer: {problem.answer}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build English CoT control dataset for apples-to-apples LoRA training.")
    p.add_argument("--output", type=Path, default=Path("runs/lora_sft_dataset_english_cot_control.jsonl"))
    p.add_argument("--dataset-size", type=int, default=1200)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    p.add_argument("--prompt-loss-multiplier", type=float, default=0.15)
    p.add_argument("--trace-loss-multiplier", type=float, default=2.0)
    p.add_argument("--answer-loss-multiplier", type=float, default=4.0)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[dict] = []
    for seed in args.seeds:
        for p in generate_dataset(size=args.dataset_size, seed=seed):
            rows.append(
                {
                    "text": english_cot_text(p),
                    "mode": "english_cot_control",
                    "problem_id": int(p.problem_id),
                    "prompt_loss_multiplier": float(args.prompt_loss_multiplier),
                    "trace_loss_multiplier": float(args.trace_loss_multiplier),
                    "answer_loss_multiplier": float(args.answer_loss_multiplier),
                    "trace_anchor": "\nReasoning:",
                    "answer_anchor": "\nFinal answer:",
                }
            )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(f"Wrote: {args.output}")
    print(f"Examples: {len(rows)}")


if __name__ == "__main__":
    main()
