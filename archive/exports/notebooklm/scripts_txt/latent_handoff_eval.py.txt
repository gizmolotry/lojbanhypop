from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b", re.IGNORECASE)


@dataclass
class Row:
    problem_id: int
    question: str
    expected: str
    base_pred: str
    handoff_pred: str
    base_ok: bool
    handoff_ok: bool
    trace_text: str
    error: str | None


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


def extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        tail = text.split("ANSWER:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    if "Final answer:" in text:
        tail = text.split("Final answer:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    t = text.strip()
    return t.splitlines()[-1].strip() if t else ""


def build_logic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise symbolic TRACE for the problem.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def build_final_prompt(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
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


def patch_nope() -> bool:
    # DroPE/NoPE: bypass rotary application so attention relies on causal mask only.
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    except Exception:
        return False

    def _identity_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return q, k

    qwen2_mod.apply_rotary_pos_emb = _identity_rotary
    return True


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    return int(past_key_values[0][0].shape[-2])


def greedy_with_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    past_key_values=None,
) -> Tuple[torch.Tensor, object]:
    # batch size 1 evaluator path.
    assert input_ids.shape[0] == 1, "Only batch_size=1 supported."
    device = input_ids.device
    generated: List[torch.Tensor] = []
    cur_past = past_key_values

    if cur_past is None:
        attn = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
                return_dict=True,
            )
        cur_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len = input_ids.shape[1] + 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            return torch.cat(generated, dim=1), cur_past
    else:
        cur_len = _past_len(cur_past)
        for i in range(int(input_ids.shape[1])):
            tok = input_ids[:, i : i + 1]
            attn = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(
                    input_ids=tok,
                    attention_mask=attn,
                    past_key_values=cur_past,
                    use_cache=True,
                    return_dict=True,
                )
            cur_past = out.past_key_values
            cur_len += 1
        tok = input_ids[:, -1:]

    for _ in range(max_new_tokens - len(generated)):
        attn = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=attn,
                past_key_values=cur_past,
                use_cache=True,
                return_dict=True,
            )
        cur_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break

    if not generated:
        return input_ids.new_empty((1, 0)), cur_past
    return torch.cat(generated, dim=1), cur_past


def base_generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = build_final_prompt(question)
    inp = tokenizer(prompt, return_tensors="pt")
    ids = inp["input_ids"].to(model.device)
    new_ids, _ = greedy_with_cache(model, ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
    return extract_answer(text)


def handoff_generate_answer(model, tokenizer, question: str, max_logic_new_tokens: int, max_final_new_tokens: int) -> Tuple[str, str]:
    logic_prompt = build_logic_prompt(question)
    logic_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)

    # Step 1: adapter-on symbolic logic pass.
    logic_new_ids, past = greedy_with_cache(
        model,
        logic_ids,
        max_new_tokens=max_logic_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        past_key_values=None,
    )
    trace_text = tokenizer.decode(logic_new_ids[0], skip_special_tokens=True)

    # Step 2/3: airlock and injection into adapter-off path.
    suffix_ids = tokenizer("\nTherefore, the final answer is ", return_tensors="pt").input_ids.to(model.device)
    with adapter_disabled(model):
        final_new_ids, _ = greedy_with_cache(
            model,
            suffix_ids,
            max_new_tokens=max_final_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            past_key_values=past,
        )
    final_text = tokenizer.decode(final_new_ids[0], skip_special_tokens=True)
    return trace_text, extract_answer(final_text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DroPE/NoPE latent-handoff smoke evaluator.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("runs/latent_handoff_eval.json"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    nope_patched = patch_nope()

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = test[: args.sample_size]

    rows: List[Row] = []
    for p in sample:
        err = None
        trace_text = ""
        with adapter_disabled(model):
            base_pred = base_generate_answer(model, tokenizer, p.prompt, max_new_tokens=args.max_final_new_tokens)
        try:
            trace_text, handoff_pred = handoff_generate_answer(
                model,
                tokenizer,
                p.prompt,
                max_logic_new_tokens=args.max_logic_new_tokens,
                max_final_new_tokens=args.max_final_new_tokens,
            )
        except Exception as exc:
            handoff_pred = ""
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            Row(
                problem_id=p.problem_id,
                question=p.prompt,
                expected=p.answer,
                base_pred=base_pred,
                handoff_pred=handoff_pred,
                base_ok=answers_match(p.answer, base_pred),
                handoff_ok=answers_match(p.answer, handoff_pred),
                trace_text=trace_text,
                error=err,
            )
        )

    total = len(rows)
    base_correct = sum(int(r.base_ok) for r in rows)
    handoff_correct = sum(int(r.handoff_ok) for r in rows)
    errors = sum(1 for r in rows if r.error is not None)
    payload: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "nope_patched": nope_patched,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "summary": {
            "total": total,
            "base_correct": base_correct,
            "handoff_correct": handoff_correct,
            "base_acc": (base_correct / total) if total else 0.0,
            "handoff_acc": (handoff_correct / total) if total else 0.0,
            "handoff_lift_vs_base": ((handoff_correct - base_correct) / total) if total else 0.0,
            "error_count": errors,
        },
        "rows": [asdict(r) for r in rows],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"NoPE patch active: {nope_patched}")
    print(f"base acc: {payload['summary']['base_acc']:.3f}")
    print(f"handoff acc: {payload['summary']['handoff_acc']:.3f}")
    print(f"handoff lift vs base: {payload['summary']['handoff_lift_vs_base']:.3f}")
    print(f"errors: {errors}")


if __name__ == "__main__":
    main()
