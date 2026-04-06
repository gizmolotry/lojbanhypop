from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b", re.IGNORECASE)


@dataclass
class EvalRow:
    problem_id: int
    mode: str
    prompt: str
    expected: str
    predicted: str
    normalized_expected: str
    normalized_predicted: str
    correct: bool


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("in the ", "").replace("the ", "")
    return NON_ALNUM_RE.sub("", lowered)


def canonicalize_roles(text: str) -> str:
    found = {}
    for person, role in ROLE_RE.findall(text):
        found[person.lower()] = role.lower()
    if {"a", "b", "c"}.issubset(found.keys()):
        return f"a={found['a']},b={found['b']},c={found['c']}"
    return ""


def answers_match(expected: str, predicted: str) -> tuple[bool, str, str]:
    n_expected = normalize_answer(expected)
    n_pred = normalize_answer(predicted)
    if "a=knight,b=knave,c=knight" in n_expected or "a=knight,b=knight,c=knave" in n_expected:
        c_expected = canonicalize_roles(expected)
        c_pred = canonicalize_roles(predicted)
        if c_expected and c_pred:
            return c_expected == c_pred, c_expected, c_pred
    if n_pred.startswith(n_expected):
        return True, n_expected, n_pred
    return n_expected == n_pred, n_expected, n_pred


def build_prompt(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )


def build_symbolic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        tail = text.split("ANSWER:")[-1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        return lines[0].strip() if lines else ""
    if "Final answer:" in text:
        tail = text.split("Final answer:")[-1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        return lines[0].strip() if lines else ""
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    return lines[-1].strip() if lines else ""


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(decoded)


def summarize(rows: List[EvalRow], mode: str) -> dict:
    subset = [r for r in rows if r.mode == mode]
    total = len(subset)
    correct = sum(1 for r in subset if r.correct)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HF base model vs LoRA adapter on the synthetic logic test split.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("runs/hf_adapter_eval.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt-style", choices=["final_answer", "symbolic"], default="final_answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()

    # Load a separate backbone instance so adapter injection cannot alias/mutate the baseline model.
    adapted_backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    adapted_backbone.resize_token_embeddings(len(tokenizer))
    adapted_model = PeftModel.from_pretrained(adapted_backbone, str(args.adapter), local_files_only=args.local_files_only)
    adapted_model.eval()

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = test[: args.sample_size]

    rows: List[EvalRow] = []
    for p in sample:
        prompt = build_prompt(p.prompt)
        if args.prompt_style == "symbolic":
            prompt = build_symbolic_prompt(p.prompt)

        pred_base = generate_text(base_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        ok_b, ne_b, np_b = answers_match(p.answer, pred_base)
        rows.append(
            EvalRow(
                problem_id=p.problem_id,
                mode="base",
                prompt=p.prompt,
                expected=p.answer,
                predicted=pred_base,
                normalized_expected=ne_b,
                normalized_predicted=np_b,
                correct=ok_b,
            )
        )

        pred_adapt = generate_text(adapted_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        ok_a, ne_a, np_a = answers_match(p.answer, pred_adapt)
        rows.append(
            EvalRow(
                problem_id=p.problem_id,
                mode="adapter",
                prompt=p.prompt,
                expected=p.answer,
                predicted=pred_adapt,
                normalized_expected=ne_a,
                normalized_predicted=np_a,
                correct=ok_a,
            )
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "summary": {
            "base": summarize(rows, "base"),
            "adapter": summarize(rows, "adapter"),
        },
        "rows": [asdict(r) for r in rows],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"base: {payload['summary']['base']['correct']}/{payload['summary']['base']['total']} ({payload['summary']['base']['accuracy']:.3f})")
    print(f"adapter: {payload['summary']['adapter']['correct']}/{payload['summary']['adapter']['total']} ({payload['summary']['adapter']['accuracy']:.3f})")


if __name__ == "__main__":
    main()
