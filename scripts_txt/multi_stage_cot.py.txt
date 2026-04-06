from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(
    r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b",
    re.IGNORECASE,
)


@contextmanager
def adapter_disabled(model):
    disable_ctx = None
    if hasattr(model, "disable_adapter"):
        disable_ctx = model.disable_adapter()
    elif hasattr(model, "disable_adapters"):
        disable_ctx = model.disable_adapters()
    if disable_ctx is None:
        with nullcontext():
            yield
    else:
        with disable_ctx:
            yield


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower().replace("in the ", "").replace("the ", "")
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
    return n_pred.startswith(n_expected) or (n_expected == n_pred)


def extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        tail = text.split("ANSWER:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    if "Final answer:" in text:
        tail = text.split("Final answer:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    stripped = text.strip()
    lines = stripped.splitlines()
    return lines[-1].strip() if lines else ""


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return extract_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


def build_logic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise TRACE followed by ANSWER.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def build_verifier_prompt(question: str, trace: str) -> str:
    return (
        "Review the TRACE above for identity consistency.\n"
        "Are Person A and B distinct when required? Explain in exactly 1 sentence.\n\n"
        f"QUESTION: {question}\n"
        f"TRACE: {trace}\n"
        "Verification:"
    )


def build_final_prompt(question: str, trace: str, verification: str) -> str:
    return (
        "Use the symbolic trace and verification to answer.\n"
        "Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        f"TRACE: {trace}\n"
        f"Verification: {verification}\n"
        "Final answer:"
    )


def build_baseline_prompt(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )


def _append_ablation_md(path: Path, line: str) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    marker = "## Trinity Expansion"
    if marker not in content:
        content = content.rstrip() + "\n\n## Trinity Expansion\n\n"
    content = content.rstrip() + f"\n- {line}\n"
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run F: multi-stage text self-correction chain.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-trace-new-tokens", type=int, default=64)
    p.add_argument("--max-verify-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("runs/multi_stage_cot_run_f.json"))
    p.add_argument("--ablation-md", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_src = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()

    per_seed: List[Dict[str, float]] = []
    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        dataset = generate_dataset(size=args.dataset_size, seed=seed)
        _, _, test = split_dataset(dataset)
        sample = test[: args.sample_size]
        base_ok = 0
        runf_ok = 0
        for p in sample:
            with adapter_disabled(model):
                base_pred = generate_text(model, tokenizer, build_baseline_prompt(p.prompt), args.max_final_new_tokens)
            trace = generate_text(model, tokenizer, build_logic_prompt(p.prompt), args.max_trace_new_tokens)
            verify = generate_text(model, tokenizer, build_verifier_prompt(p.prompt, trace), args.max_verify_new_tokens)
            final_pred = generate_text(
                model,
                tokenizer,
                build_final_prompt(p.prompt, trace, verify),
                args.max_final_new_tokens,
            )
            b = answers_match(p.answer, base_pred)
            f = answers_match(p.answer, final_pred)
            base_ok += int(b)
            runf_ok += int(f)
            rows.append(
                {
                    "seed": seed,
                    "problem_id": p.problem_id,
                    "question": p.prompt,
                    "expected": p.answer,
                    "base_pred": base_pred,
                    "trace": trace,
                    "verification": verify,
                    "run_f_pred": final_pred,
                    "base_ok": b,
                    "run_f_ok": f,
                }
            )
        per_seed.append(
            {
                "seed": seed,
                "base_acc": base_ok / len(sample) if sample else 0.0,
                "run_f_acc": runf_ok / len(sample) if sample else 0.0,
                "run_f_lift": (runf_ok - base_ok) / len(sample) if sample else 0.0,
            }
        )

    mean_base = sum(x["base_acc"] for x in per_seed) / len(per_seed) if per_seed else 0.0
    mean_runf = sum(x["run_f_acc"] for x in per_seed) / len(per_seed) if per_seed else 0.0
    mean_lift = sum(x["run_f_lift"] for x in per_seed) / len(per_seed) if per_seed else 0.0
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "dataset_size": args.dataset_size,
        "mean_base_acc": mean_base,
        "mean_run_f_acc": mean_runf,
        "mean_run_f_lift": mean_lift,
        "per_seed": per_seed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"mean base acc: {mean_base:.3f}")
    print(f"mean run_f acc: {mean_runf:.3f}")
    print(f"mean run_f lift: {mean_lift:+.3f}")

    if args.ablation_md is not None:
        _append_ablation_md(args.ablation_md, f"`Run F (Self-Correct)` mean_acc={mean_runf:.3f}, lift={mean_lift:+.3f}.")


if __name__ == "__main__":
    main()
