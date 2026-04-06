from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(
    r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b",
    re.IGNORECASE,
)


@dataclass
class EvalRow:
    problem_id: int
    question: str
    expected: str
    base_pred: str
    handoff_pred: str
    base_ok: bool
    handoff_ok: bool
    coconut_trace: str
    error: str | None


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


def patch_nope_qwen2() -> bool:
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    except Exception:
        return False

    def _identity_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return q, k

    qwen2_mod.apply_rotary_pos_emb = _identity_rotary
    return True


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
    return n_pred.startswith(n_expected) or (n_expected == n_pred)


def extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        t = text.split("ANSWER:")[-1].strip()
        return t.splitlines()[0].strip() if t else ""
    if "Final answer:" in text:
        t = text.split("Final answer:")[-1].strip()
        return t.splitlines()[0].strip() if t else ""
    stripped = text.strip()
    lines = stripped.splitlines()
    return lines[-1].strip() if lines else ""


def build_logic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Think in concise symbolic form. End trace with token _E.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def build_final_prompt(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    first = past_key_values[0][0]
    return int(first.shape[-2])


def _validate_cache_shape(cache_obj, model) -> Tuple[bool, str]:
    if cache_obj is None:
        return False, "Missing past_key_values."
    try:
        n_layers = len(cache_obj)
    except Exception:
        return False, "past_key_values is not indexable."
    if n_layers == 0:
        return False, "past_key_values has no layers."
    model_layers = int(getattr(model.config, "num_hidden_layers", n_layers))
    if n_layers != model_layers:
        return False, f"Layer mismatch: cache={n_layers}, model={model_layers}"
    k0 = cache_obj[0][0]
    v0 = cache_obj[0][1]
    if k0.shape != v0.shape:
        return False, f"K/V shape mismatch in layer0: {tuple(k0.shape)} vs {tuple(v0.shape)}"
    cfg = model.config
    expected_kv_heads = int(getattr(cfg, "num_key_value_heads", k0.shape[1]))
    expected_head_dim = int(getattr(cfg, "head_dim", k0.shape[-1]))
    if k0.shape[1] != expected_kv_heads or k0.shape[-1] != expected_head_dim:
        return False, (
            "Head-dim mismatch: "
            f"cache(kv_heads,head_dim)=({k0.shape[1]},{k0.shape[-1]}) "
            f"expected=({expected_kv_heads},{expected_head_dim}). "
            "This often means adapter/modules_to_save modified incompatible internals."
        )
    return True, "ok"


def _generate_with_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    past_key_values=None,
) -> Tuple[torch.Tensor, object]:
    # Deterministic evaluator path supports batch_size=1.
    assert input_ids.shape[0] == 1, "Only batch_size=1 supported."
    device = input_ids.device
    current_past = past_key_values
    generated: List[torch.Tensor] = []

    if current_past is None:
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        current_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len = input_ids.shape[1] + 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            return torch.cat(generated, dim=1), current_past
    else:
        cur_len = _past_len(current_past)
        # Feed suffix tokens one by one into injected cache.
        for i in range(int(input_ids.shape[1])):
            tok = input_ids[:, i : i + 1]
            am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(
                    input_ids=tok,
                    attention_mask=am,
                    past_key_values=current_past,
                    use_cache=True,
                    return_dict=True,
                )
            current_past = out.past_key_values
            cur_len += 1
        tok = input_ids[:, -1:]

    for _ in range(max_new_tokens - len(generated)):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=current_past,
                use_cache=True,
                return_dict=True,
            )
        current_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break

    if not generated:
        return input_ids.new_empty((1, 0)), current_past
    return torch.cat(generated, dim=1), current_past


def _baseline_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = build_final_prompt(question)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    new_ids, _ = _generate_with_cache(model, ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    return extract_answer(tokenizer.decode(new_ids[0], skip_special_tokens=True))


def coconut_handoff_answer(
    model,
    tokenizer,
    question: str,
    max_logic_new_tokens: int,
    max_final_new_tokens: int,
    lojban_exit_token: str,
) -> Tuple[str, str]:
    # Phase 2: System 1 logic run with adapter enabled.
    logic_prompt = build_logic_prompt(question)
    logic_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    logic_new_ids, coconut_seed = _generate_with_cache(
        model,
        logic_ids,
        max_new_tokens=max_logic_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        past_key_values=None,
    )
    trace = tokenizer.decode(logic_new_ids[0], skip_special_tokens=True)
    if lojban_exit_token and lojban_exit_token in trace:
        trace = trace.split(lojban_exit_token)[0] + lojban_exit_token

    ok, msg = _validate_cache_shape(coconut_seed, model)
    if not ok:
        raise RuntimeError(f"KV-cache alignment error before airlock: {msg}")

    # Phase 3/4: disable adapter, inject "coconut seed", continue in base manifold.
    suffix_ids = tokenizer("\nTherefore, the final answer is ", return_tensors="pt").input_ids.to(model.device)
    with adapter_disabled(model):
        final_new_ids, _ = _generate_with_cache(
            model,
            suffix_ids,
            max_new_tokens=max_final_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            past_key_values=coconut_seed,
        )
    final_text = tokenizer.decode(final_new_ids[0], skip_special_tokens=True)
    return trace, extract_answer(final_text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discrete Coconut Fusion latent handoff evaluator.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=64)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--lojban-exit-token", type=str, default="_E")
    p.add_argument("--output", type=Path, default=Path("runs/coconut_handoff_eval.json"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

    nope_patched = patch_nope_qwen2()

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_src = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = test[: args.sample_size]

    rows: List[EvalRow] = []
    for item in sample:
        with adapter_disabled(model):
            base_pred = _baseline_answer(model, tokenizer, item.prompt, max_new_tokens=args.max_final_new_tokens)
        trace = ""
        handoff_pred = ""
        err = None
        try:
            trace, handoff_pred = coconut_handoff_answer(
                model=model,
                tokenizer=tokenizer,
                question=item.prompt,
                max_logic_new_tokens=args.max_logic_new_tokens,
                max_final_new_tokens=args.max_final_new_tokens,
                lojban_exit_token=args.lojban_exit_token,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            EvalRow(
                problem_id=item.problem_id,
                question=item.prompt,
                expected=item.answer,
                base_pred=base_pred,
                handoff_pred=handoff_pred,
                base_ok=answers_match(item.answer, base_pred),
                handoff_ok=answers_match(item.answer, handoff_pred),
                coconut_trace=trace,
                error=err,
            )
        )

    total = len(rows)
    base_correct = sum(int(r.base_ok) for r in rows)
    handoff_correct = sum(int(r.handoff_ok) for r in rows)
    errors = sum(int(r.error is not None) for r in rows)
    payload: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "nope_patched": nope_patched,
        "summary": {
            "total": total,
            "base_correct": base_correct,
            "handoff_correct": handoff_correct,
            "base_acc": (base_correct / total) if total else 0.0,
            "handoff_acc": (handoff_correct / total) if total else 0.0,
            "errors": errors,
        },
        "rows": [asdict(r) for r in rows],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"NoPE patch active: {nope_patched}")
    print(f"base acc: {payload['summary']['base_acc']:.3f}")
    print(f"handoff acc: {payload['summary']['handoff_acc']:.3f}")
    print(f"errors: {errors}")


if __name__ == "__main__":
    main()
