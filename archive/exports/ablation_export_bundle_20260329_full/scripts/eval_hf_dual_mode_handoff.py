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
ROLE_RE = re.compile(r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b", re.IGNORECASE)


@dataclass
class EvalSummary:
    total: int
    base_correct: int
    adapter_correct: int
    handoff_correct: int

    @property
    def base_acc(self) -> float:
        return self.base_correct / self.total if self.total else 0.0

    @property
    def adapter_acc(self) -> float:
        return self.adapter_correct / self.total if self.total else 0.0

    @property
    def handoff_acc(self) -> float:
        return self.handoff_correct / self.total if self.total else 0.0

    @property
    def adapter_lift(self) -> float:
        return self.adapter_acc - self.base_acc

    @property
    def handoff_lift(self) -> float:
        return self.handoff_acc - self.base_acc


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


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(decoded)


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    # layer tuple is usually (k, v, ...) and k shape is [b, h, seq, d]
    first = past_key_values[0][0]
    return int(first.shape[-2])


def _apply_optional_projection(past_key_values, proj: Optional[torch.nn.Linear]):
    if proj is None or past_key_values is None:
        return past_key_values
    out = []
    for layer in past_key_values:
        k, v = layer[0], layer[1]
        b, h, s, d = k.shape
        k2 = proj(k.reshape(b * h * s, d)).reshape(b, h, s, d)
        v2 = proj(v.reshape(b * h * s, d)).reshape(b, h, s, d)
        if len(layer) > 2:
            out.append((k2, v2, *layer[2:]))
        else:
            out.append((k2, v2))
    return tuple(out)


def patch_nope_qwen2() -> bool:
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    except Exception:
        return False

    def _identity_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return q, k

    qwen2_mod.apply_rotary_pos_emb = _identity_rotary
    return True


def _build_projection(path: Optional[Path], device) -> Optional[torch.nn.Linear]:
    if path is None:
        return None
    ckpt = torch.load(str(path), map_location="cpu")
    state = None
    if isinstance(ckpt, torch.nn.Linear):
        proj = ckpt.to(device)
        proj.eval()
        return proj
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "projection" in ckpt and isinstance(ckpt["projection"], dict):
            state = ckpt["projection"]
        elif "weight" in ckpt:
            state = {"weight": ckpt["weight"], "bias": ckpt.get("bias")}
        else:
            state = ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported projection checkpoint format: {path}")
    if "weight" not in state:
        if "projection.weight" in state:
            state = {
                "weight": state["projection.weight"],
                "bias": state.get("projection.bias"),
            }
        else:
            raise ValueError(f"Projection checkpoint missing weight tensor: {path}")
    w = state["weight"]
    if not hasattr(w, "shape") or len(w.shape) != 2:
        raise ValueError(f"Projection weight must be rank-2, got: {type(w)}")
    out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
    proj = torch.nn.Linear(in_dim, out_dim, bias=("bias" in state and state["bias"] is not None))
    proj.load_state_dict({k: v for k, v in state.items() if k in ("weight", "bias")}, strict=False)
    proj.to(device)
    proj.eval()
    return proj


def _cache_decode(
    model,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    past_key_values=None,
) -> Tuple[torch.Tensor, object]:
    # Supports batch_size=1 for deterministic evaluator use.
    assert start_ids.shape[0] == 1, "Only batch_size=1 supported for handoff evaluator."
    device = start_ids.device
    attention_mask = torch.ones_like(start_ids, device=device)
    current_past = past_key_values
    generated = []

    if current_past is None:
        with torch.no_grad():
            out = model(
                input_ids=start_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        current_past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)
        cur_len = start_ids.shape[1] + 1
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            return torch.cat(generated, dim=1), current_past
    else:
        cur_len = _past_len(current_past)
        suffix = start_ids
        for i in range(int(suffix.shape[1])):
            tok = suffix[:, i : i + 1]
            pos = torch.tensor([[cur_len]], dtype=torch.long, device=device)
            am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(
                    input_ids=tok,
                    attention_mask=am,
                    position_ids=pos,
                    past_key_values=current_past,
                    use_cache=True,
                    return_dict=True,
                )
            current_past = out.past_key_values
            cur_len += 1

    # Continue autoregressive generation from existing cache.
    tok = generated[-1] if generated else start_ids[:, -1:]
    for _ in range(max_new_tokens - len(generated)):
        pos = torch.tensor([[cur_len]], dtype=torch.long, device=device)
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                position_ids=pos,
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
        return start_ids.new_empty((1, 0)), current_past
    return torch.cat(generated, dim=1), current_past


def latent_handoff_generate(
    model,
    tokenizer,
    question: str,
    max_logic_new_tokens: int,
    max_final_new_tokens: int,
    handoff_suffix: str,
    projection: Optional[torch.nn.Linear] = None,
) -> Dict[str, str]:
    # Stage 1: logic pass with adapter enabled (system-1).
    logic_prompt = build_prompt(question, style="symbolic", use_mode_tags=False)
    logic_inputs = tokenizer(logic_prompt, return_tensors="pt")
    logic_ids = logic_inputs["input_ids"].to(model.device)
    logic_new, logic_past = _cache_decode(
        model=model,
        start_ids=logic_ids,
        max_new_tokens=max_logic_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        past_key_values=None,
    )
    logic_text = tokenizer.decode(logic_new[0], skip_special_tokens=True)
    logic_past = _apply_optional_projection(logic_past, projection)

    # Stage 2: adapter-off handoff to base fluency (system-2).
    suffix_ids = tokenizer(handoff_suffix, return_tensors="pt").input_ids.to(model.device)
    with adapter_disabled(model):
        final_new, _ = _cache_decode(
            model=model,
            start_ids=suffix_ids,
            max_new_tokens=max_final_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            past_key_values=logic_past,
        )
    final_text = tokenizer.decode(final_new[0], skip_special_tokens=True)
    return {
        "logic_trace_raw": logic_text,
        "final_raw": final_text,
        "final_answer": extract_answer(final_text),
    }


def eval_style(
    base_model,
    adapted_model,
    tokenizer,
    sample,
    style: str,
    max_new_tokens: int,
    use_mode_tags: bool,
    projection: Optional[torch.nn.Linear],
) -> EvalSummary:
    base_correct = 0
    adapter_correct = 0
    handoff_correct = 0
    for p in sample:
        prompt = build_prompt(p.prompt, style, use_mode_tags=use_mode_tags)
        with adapter_disabled(adapted_model):
            pb = generate_text(adapted_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        pa = generate_text(adapted_model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        if style == "final_answer":
            handoff = latent_handoff_generate(
                model=adapted_model,
                tokenizer=tokenizer,
                question=p.prompt,
                max_logic_new_tokens=max_new_tokens,
                max_final_new_tokens=max_new_tokens,
                handoff_suffix="\nTherefore, the final answer is ",
                projection=projection,
            )["final_answer"]
        else:
            # Symbolic score for handoff engine is the system-1 trace quality.
            handoff = latent_handoff_generate(
                model=adapted_model,
                tokenizer=tokenizer,
                question=p.prompt,
                max_logic_new_tokens=max_new_tokens,
                max_final_new_tokens=max_new_tokens,
                handoff_suffix="\nANSWER: ",
                projection=projection,
            )["final_answer"]

        base_correct += int(answers_match(p.answer, pb))
        adapter_correct += int(answers_match(p.answer, pa))
        handoff_correct += int(answers_match(p.answer, handoff))
    return EvalSummary(
        total=len(sample),
        base_correct=base_correct,
        adapter_correct=adapter_correct,
        handoff_correct=handoff_correct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-mode gate evaluation with latent handoff engine.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-symbolic-lift", type=float, default=0.0)
    parser.add_argument("--min-final-lift", type=float, default=0.0)
    parser.add_argument("--use-mode-tags", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/hf_dual_mode_handoff_gate.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--handoff-projection", type=Path, default=None, help="Optional checkpoint for head-dim projection.")
    parser.add_argument("--disable-rope", action="store_true", help="Apply DroPE-style NoPE patch for Qwen2 attention.")
    return parser.parse_args()


def _gate_pass(mean_final_lift: float, mean_symbolic_lift: float, min_final_lift: float, min_symbolic_lift: float) -> bool:
    return (mean_final_lift >= min_final_lift) and (mean_symbolic_lift >= min_symbolic_lift)


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

    rope_patch_active = False
    if args.disable_rope:
        rope_patch_active = patch_nope_qwen2()

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()
    model_device = next(model.parameters()).device
    projection = _build_projection(args.handoff_projection, device=model_device)

    per_seed: List[dict] = []
    for seed in args.seeds:
        dataset = generate_dataset(size=args.dataset_size, seed=seed)
        _, _, test = split_dataset(dataset)
        sample = test[: args.sample_size]
        final_summary = eval_style(
            model,
            model,
            tokenizer,
            sample,
            style="final_answer",
            max_new_tokens=args.max_new_tokens,
            use_mode_tags=args.use_mode_tags,
            projection=projection,
        )
        symbolic_summary = eval_style(
            model,
            model,
            tokenizer,
            sample,
            style="symbolic",
            max_new_tokens=args.max_new_tokens,
            use_mode_tags=args.use_mode_tags,
            projection=projection,
        )
        per_seed.append(
            {
                "seed": seed,
                "final_answer": asdict(final_summary)
                | {
                    "base_acc": final_summary.base_acc,
                    "adapter_acc": final_summary.adapter_acc,
                    "handoff_acc": final_summary.handoff_acc,
                    "adapter_lift": final_summary.adapter_lift,
                    "handoff_lift": final_summary.handoff_lift,
                },
                "symbolic": asdict(symbolic_summary)
                | {
                    "base_acc": symbolic_summary.base_acc,
                    "adapter_acc": symbolic_summary.adapter_acc,
                    "handoff_acc": symbolic_summary.handoff_acc,
                    "adapter_lift": symbolic_summary.adapter_lift,
                    "handoff_lift": symbolic_summary.handoff_lift,
                },
            }
        )

    final_adapter_lifts = [x["final_answer"]["adapter_lift"] for x in per_seed]
    final_handoff_lifts = [x["final_answer"]["handoff_lift"] for x in per_seed]
    symbolic_adapter_lifts = [x["symbolic"]["adapter_lift"] for x in per_seed]
    symbolic_handoff_lifts = [x["symbolic"]["handoff_lift"] for x in per_seed]
    mean_adapter_final_lift = sum(final_adapter_lifts) / len(final_adapter_lifts) if final_adapter_lifts else 0.0
    mean_handoff_final_lift = sum(final_handoff_lifts) / len(final_handoff_lifts) if final_handoff_lifts else 0.0
    mean_adapter_symbolic_lift = sum(symbolic_adapter_lifts) / len(symbolic_adapter_lifts) if symbolic_adapter_lifts else 0.0
    mean_handoff_symbolic_lift = sum(symbolic_handoff_lifts) / len(symbolic_handoff_lifts) if symbolic_handoff_lifts else 0.0
    passed = _gate_pass(
        mean_final_lift=mean_handoff_final_lift,
        mean_symbolic_lift=mean_handoff_symbolic_lift,
        min_final_lift=args.min_final_lift,
        min_symbolic_lift=args.min_symbolic_lift,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "handoff_projection": str(args.handoff_projection) if args.handoff_projection is not None else None,
        "disable_rope": bool(args.disable_rope),
        "rope_patch_active": bool(rope_patch_active),
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "thresholds": {"min_final_lift": args.min_final_lift, "min_symbolic_lift": args.min_symbolic_lift},
        "mean_lifts": {
            "adapter_final_answer": mean_adapter_final_lift,
            "handoff_final_answer": mean_handoff_final_lift,
            "adapter_symbolic": mean_adapter_symbolic_lift,
            "handoff_symbolic": mean_handoff_symbolic_lift,
        },
        "gate_pass_handoff": passed,
        "per_seed": per_seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"mean adapter final lift: {mean_adapter_final_lift:.3f}")
    print(f"mean handoff final lift: {mean_handoff_final_lift:.3f}")
    print(f"mean adapter symbolic lift: {mean_adapter_symbolic_lift:.3f}")
    print(f"mean handoff symbolic lift: {mean_handoff_symbolic_lift:.3f}")
    print(f"HANDOFF GATE: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
