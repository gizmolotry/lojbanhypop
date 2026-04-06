from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from lojban_evolution.experiment import Problem, generate_dataset


@dataclass
class Example:
    text: str
    source_eval: str
    problem_id: int
    mode: str


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_problem_map(dataset_size: int, seed: int) -> Dict[int, Problem]:
    problems = generate_dataset(size=dataset_size, seed=seed)
    return {p.problem_id: p for p in problems}


def format_training_text(prompt: str, symbolic_trace: Sequence[str], answer: str) -> str:
    trace_line = " ".join(symbolic_trace)
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
        f"QUESTION: {prompt}\n"
        f"TRACE: {trace_line}\n"
        f"ANSWER: {answer}"
    )


def collect_examples(
    eval_payload: dict,
    problem_map: Dict[int, Problem],
    mode: str,
    only_correct: bool,
) -> List[Example]:
    out: List[Example] = []
    src_model = str(eval_payload.get("model", "unknown"))
    for row in eval_payload.get("rows", []):
        if row.get("mode") != mode:
            continue
        if only_correct and not bool(row.get("correct", False)):
            continue
        if row.get("error") is not None:
            continue

        pid = int(row["problem_id"])
        problem = problem_map.get(pid)
        if problem is None:
            continue

        text = format_training_text(
            prompt=problem.prompt,
            symbolic_trace=problem.trace,
            answer=problem.answer,
        )
        out.append(
            Example(
                text=text,
                source_eval=src_model,
                problem_id=pid,
                mode=str(row.get("mode", "")),
            )
        )
    return out


def dedupe(examples: Iterable[Example]) -> List[Example]:
    seen = set()
    unique: List[Example] = []
    for ex in examples:
        key = (ex.problem_id, ex.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LoRA SFT dataset from Phase 3 eval runs.")
    parser.add_argument(
        "--eval-json",
        nargs="+",
        type=Path,
        required=True,
        help="One or more lm_eval_*.json files.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/lora_sft_dataset.jsonl"))
    parser.add_argument("--mode", type=str, default="phase3_fewshot")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-incorrect", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_map = build_problem_map(dataset_size=args.dataset_size, seed=args.seed)

    all_examples: List[Example] = []
    for eval_path in args.eval_json:
        payload = load_eval(eval_path)
        examples = collect_examples(
            eval_payload=payload,
            problem_map=problem_map,
            mode=args.mode,
            only_correct=(not args.allow_incorrect),
        )
        all_examples.extend(examples)

    unique = dedupe(all_examples)
    if args.max_samples > 0:
        unique = unique[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in unique:
            row = {
                "text": ex.text,
                "source_eval": ex.source_eval,
                "problem_id": ex.problem_id,
                "mode": ex.mode,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote: {args.output}")
    print(f"Examples: {len(unique)}")


if __name__ == "__main__":
    main()
