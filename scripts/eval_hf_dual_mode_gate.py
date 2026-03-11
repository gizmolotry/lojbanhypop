from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b", re.IGNORECASE)


@dataclass
class EvalSummary:
    total: int
    base_correct: int
    adapter_correct: int

    @property
    def base_acc(self) -> float:
        return self.base_correct / self.total if self.total else 0.0

    @property
    def adapter_acc(self) -> float:
        return self.adapter_correct / self.total if self.total else 0.0

    @property
    def lift(self) -> float:
        return self.adapter_acc - self.base_acc


def gate_pass(
    mean_final_lift: float,
    mean_symbolic_lift: float,
    min_final_lift: float,
    min_symbolic_lift: float,
) -> bool:
    return (mean_final_lift >= min_final_lift) and (mean_symbolic_lift >= min_symbolic_lift)


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


def answers_match(expected: str, predicted: str) -> bool:
    n_expected = normalize_answer(expected)
    n_pred = normalize_answer(predicted)
    if "a=knight,b=knave,c=knight" in n_expected or "a=knight,b=knight,c=knave" in n_expected:
        c_expected = canonicalize_roles(expected)
        c_pred = canonicalize_roles(predicted)
        if c_expected and c_pred:
            return c_expected == c_pred
    if n_pred.startswith(n_expected):
        return True
    return n_expected == n_pred


def build_prompt(question: str, style: str, use_mode_tags: bool) -> str:
    if style == "symbolic":
        mode_line = "[MODE=CRYSTAL]\n" if use_mode_tags else ""
        return (
            f"{mode_line}"
            "You are a rigid symbolic reasoner.\n"
            "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
            f"QUESTION: {question}\n"
            "TRACE:"
        )
    mode_line = "[MODE=FLUID]\n" if use_mode_tags else ""
    return (
        f"{mode_line}"
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
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


def eval_style(base_model, adapted_model, tokenizer, sample, style: str, max_new_tokens: int, use_mode_tags: bool) -> EvalSummary:
    b = 0
    a = 0
    for p in sample:
        prompt = build_prompt(p.prompt, style, use_mode_tags=use_mode_tags)
        pb = generate_text(base_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        pa = generate_text(adapted_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        b += int(answers_match(p.answer, pb))
        a += int(answers_match(p.answer, pa))
    return EvalSummary(total=len(sample), base_correct=b, adapter_correct=a)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-mode gate evaluation for unified adapter.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-symbolic-lift", type=float, default=0.0)
    parser.add_argument("--min-final-lift", type=float, default=0.0)
    parser.add_argument("--use-mode-tags", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/hf_dual_mode_gate.json"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: peft. Install with `pip install peft` to run adapter evaluation."
        ) from exc

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()
    adapted_backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    adapted_backbone.resize_token_embeddings(len(tokenizer))
    adapted_model = PeftModel.from_pretrained(adapted_backbone, str(args.adapter), local_files_only=args.local_files_only)
    adapted_model.eval()

    per_seed: List[dict] = []
    for seed in args.seeds:
        dataset = generate_dataset(size=args.dataset_size, seed=seed)
        _, _, test = split_dataset(dataset)
        sample = test[: args.sample_size]
        final_summary = eval_style(
            base_model,
            adapted_model,
            tokenizer,
            sample,
            style="final_answer",
            max_new_tokens=args.max_new_tokens,
            use_mode_tags=args.use_mode_tags,
        )
        symbolic_summary = eval_style(
            base_model,
            adapted_model,
            tokenizer,
            sample,
            style="symbolic",
            max_new_tokens=args.max_new_tokens,
            use_mode_tags=args.use_mode_tags,
        )
        per_seed.append(
            {
                "seed": seed,
                "final_answer": asdict(final_summary) | {"base_acc": final_summary.base_acc, "adapter_acc": final_summary.adapter_acc, "lift": final_summary.lift},
                "symbolic": asdict(symbolic_summary) | {"base_acc": symbolic_summary.base_acc, "adapter_acc": symbolic_summary.adapter_acc, "lift": symbolic_summary.lift},
            }
        )

    final_lifts = [x["final_answer"]["lift"] for x in per_seed]
    symbolic_lifts = [x["symbolic"]["lift"] for x in per_seed]
    mean_final_lift = sum(final_lifts) / len(final_lifts) if final_lifts else 0.0
    mean_symbolic_lift = sum(symbolic_lifts) / len(symbolic_lifts) if symbolic_lifts else 0.0
    passed = gate_pass(
        mean_final_lift=mean_final_lift,
        mean_symbolic_lift=mean_symbolic_lift,
        min_final_lift=args.min_final_lift,
        min_symbolic_lift=args.min_symbolic_lift,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "thresholds": {"min_final_lift": args.min_final_lift, "min_symbolic_lift": args.min_symbolic_lift},
        "use_mode_tags": args.use_mode_tags,
        "mean_lifts": {"final_answer": mean_final_lift, "symbolic": mean_symbolic_lift},
        "gate_pass": passed,
        "per_seed": per_seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"mean final lift: {mean_final_lift:.3f}")
    print(f"mean symbolic lift: {mean_symbolic_lift:.3f}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
