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
import torch.nn.functional as F
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
    coconut_pred: str
    base_ok: bool
    coconut_ok: bool
    logic_trace: str
    virtual_token_count: int
    injection_mode: str
    step_cosine: List[Dict[str, float]] | None
    error: str | None


class SwiGLUBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, expansion_factor: int = 2):
        super().__init__()
        inner = int(hidden_size * expansion_factor)
        self.w1 = torch.nn.Linear(hidden_size, inner, bias=True)
        self.w2 = torch.nn.Linear(inner, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (xW1 ⊙ SiLU(xW1))W2
        h = self.w1(x)
        return self.w2(h * F.silu(h))


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
        tail = text.split("ANSWER:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    if "Final answer:" in text:
        tail = text.split("Final answer:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    stripped = text.strip()
    lines = stripped.splitlines()
    return lines[-1].strip() if lines else ""


def build_logic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise symbolic TRACE and then ANSWER.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def build_final_prompt_prefix(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}"
    )


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    return int(past_key_values[0][0].shape[-2])


def _greedy_logic_with_last_hidden(
    model,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns generated ids and final hidden state of last generated token.
    assert start_ids.shape[0] == 1, "Only batch_size=1 supported."
    device = start_ids.device
    generated: List[torch.Tensor] = []
    last_hidden: Optional[torch.Tensor] = None

    with torch.no_grad():
        out = model(
            input_ids=start_ids,
            attention_mask=torch.ones_like(start_ids, device=device),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(tok)
    last_hidden = out.hidden_states[-1][:, -1:, :]
    cur_len = start_ids.shape[1] + 1
    if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
        return torch.cat(generated, dim=1), last_hidden

    for _ in range(max_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break

    if last_hidden is None:
        last_hidden = start_ids.new_zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    return torch.cat(generated, dim=1), last_hidden


def _greedy_logic_with_hidden_window(
    model,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(window_size) <= 1:
        logic_ids, last_hidden = _greedy_logic_with_last_hidden(model, start_ids, max_new_tokens, eos_token_id)
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(1)
        return logic_ids, last_hidden

    assert start_ids.shape[0] == 1, "Only batch_size=1 supported."
    device = start_ids.device
    generated: List[torch.Tensor] = []
    hidden_steps: List[torch.Tensor] = []

    with torch.no_grad():
        out = model(
            input_ids=start_ids,
            attention_mask=torch.ones_like(start_ids, device=device),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(tok)
    hidden_steps.append(out.hidden_states[-1][:, -1:, :])
    cur_len = start_ids.shape[1] + 1
    if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
        return torch.cat(generated, dim=1), torch.cat(hidden_steps[-window_size:], dim=1)

    for _ in range(max_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        hidden_steps.append(out.hidden_states[-1][:, -1:, :])
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break

    if not hidden_steps:
        hidden_steps = [start_ids.new_zeros((1, 1, model.config.hidden_size), dtype=torch.float32)]
    return torch.cat(generated, dim=1), torch.cat(hidden_steps[-window_size:], dim=1)


def _resolve_decoder_layers(model):
    # Covers common HF + PEFT layouts used in this repo.
    for root in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        inner = getattr(root, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    return None


def _load_swiglu_bridge(path: Optional[Path], hidden_size: int, device: torch.device, dtype: torch.dtype):
    if path is None:
        return None
    ckpt = torch.load(str(path), map_location="cpu")
    state = None
    expansion = 2
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
            expansion = int(ckpt.get("expansion_factor", expansion))
        elif "bridge" in ckpt and isinstance(ckpt["bridge"], dict):
            state = ckpt["bridge"]
            expansion = int(ckpt.get("expansion_factor", expansion))
        elif "w1.weight" in ckpt:
            state = ckpt
            w1 = ckpt.get("w1.weight")
            if hasattr(w1, "shape") and len(w1.shape) == 2 and int(w1.shape[1]) == hidden_size:
                expansion = max(1, int(w1.shape[0] // hidden_size))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported SwiGLU bridge checkpoint format: {path}")
    bridge = SwiGLUBridge(hidden_size=hidden_size, expansion_factor=expansion)
    bridge.load_state_dict(state, strict=True)
    bridge.to(device=device, dtype=dtype)
    bridge.eval()
    return bridge


@contextmanager
def mid_layer_injector(model, layer_index: int, inject_state: torch.Tensor, scale: float, persistent: bool = False):
    layers = _resolve_decoder_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for mid-layer injection.")
    if layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError(f"mid-layer index out of range: {layer_index} (layers={len(layers)})")
    inject_state = inject_state.detach()

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            add = inject_state.to(device=hidden.device, dtype=hidden.dtype)
            if add.dim() == 2:
                add = add.unsqueeze(1)
            hidden = hidden.clone()
            if persistent:
                hidden = hidden + (float(scale) * add)
            else:
                hidden[:, -1:, :] = hidden[:, -1:, :] + (float(scale) * add[:, -1:, :])
            return (hidden, *rest)
        hidden = output
        add = inject_state.to(device=hidden.device, dtype=hidden.dtype)
        if add.dim() == 2:
            add = add.unsqueeze(1)
        hidden = hidden.clone()
        if persistent:
            hidden = hidden + (float(scale) * add)
        else:
            hidden[:, -1:, :] = hidden[:, -1:, :] + (float(scale) * add[:, -1:, :])
        return hidden

    handle = layers[layer_index].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _generate_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
) -> str:
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # For inputs_embeds path, decode all returned ids and extract answer marker/last line.
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_answer(text)


def _generate_from_embeds_with_step_cosine(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
    reference_states: torch.Tensor,
) -> Tuple[str, List[Dict[str, float]]]:
    generated: List[torch.Tensor] = []
    cur_len = int(inputs_embeds.shape[1])
    ref = reference_states
    if ref.dim() == 2:
        ref = ref.unsqueeze(1)
    trace: List[Dict[str, float]] = []

    def _log_step(step_idx: int, hidden: torch.Tensor) -> None:
        # hidden: [1, 1, H], ref: [1, N, H]
        h = hidden.expand(-1, ref.shape[1], -1)
        cos = F.cosine_similarity(h.float(), ref.float(), dim=-1)[0]
        trace.append(
            {
                "step": float(step_idx),
                "cos_mean": float(cos.mean().item()),
                "cos_max": float(cos.max().item()),
                "cos_min": float(cos.min().item()),
            }
        )

    with torch.no_grad():
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(tok)
    _log_step(0, out.hidden_states[-1][:, -1:, :])
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
        return extract_answer(text), trace
    cur_len += 1

    for i in range(max_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=inputs_embeds.device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        _log_step(i + 1, out.hidden_states[-1][:, -1:, :])
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break
    text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    return extract_answer(text), trace


def _contrastive_decode_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
    alpha: float,
    inject_factory,
    reference_states: Optional[torch.Tensor] = None,
) -> Tuple[str, List[Dict[str, float]]]:
    # Pass A: baseline (no injection); Pass B: logic-injected branch.
    trace: List[Dict[str, float]] = []
    cur_len = int(inputs_embeds.shape[1])
    ref = reference_states
    if ref is not None and ref.dim() == 2:
        ref = ref.unsqueeze(1)

    def _step_trace(step_idx: int, hidden: torch.Tensor) -> None:
        if ref is None:
            return
        h = hidden.expand(-1, ref.shape[1], -1)
        cos = F.cosine_similarity(h.float(), ref.float(), dim=-1)[0]
        trace.append(
            {
                "step": float(step_idx),
                "cos_mean": float(cos.mean().item()),
                "cos_max": float(cos.max().item()),
                "cos_min": float(cos.min().item()),
            }
        )

    with torch.no_grad():
        out_a = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        with inject_factory():
            out_b = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=(ref is not None),
            )

    probs_a = torch.softmax(out_a.logits[:, -1, :], dim=-1)
    probs_b = torch.softmax(out_b.logits[:, -1, :], dim=-1)
    probs = torch.clamp(probs_b + (float(alpha) * (probs_b - probs_a)), min=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    tok = torch.argmax(probs, dim=-1, keepdim=True)
    generated: List[torch.Tensor] = [tok]
    past_a = out_a.past_key_values
    past_b = out_b.past_key_values
    if ref is not None and getattr(out_b, "hidden_states", None) is not None:
        _step_trace(0, out_b.hidden_states[-1][:, -1:, :])
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
        return extract_answer(text), trace
    cur_len += 1

    for i in range(max_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=inputs_embeds.device)
        with torch.no_grad():
            out_a = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=past_a,
                use_cache=True,
                return_dict=True,
            )
            with inject_factory():
                out_b = model(
                    input_ids=tok,
                    attention_mask=am,
                    past_key_values=past_b,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=(ref is not None),
                )
        past_a = out_a.past_key_values
        past_b = out_b.past_key_values
        probs_a = torch.softmax(out_a.logits[:, -1, :], dim=-1)
        probs_b = torch.softmax(out_b.logits[:, -1, :], dim=-1)
        probs = torch.clamp(probs_b + (float(alpha) * (probs_b - probs_a)), min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tok = torch.argmax(probs, dim=-1, keepdim=True)
        generated.append(tok)
        if ref is not None and getattr(out_b, "hidden_states", None) is not None:
            _step_trace(i + 1, out_b.hidden_states[-1][:, -1:, :])
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break

    text = tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
    return extract_answer(text), trace


def baseline_generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = build_final_prompt_prefix(question) + "\nFinal answer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    return extract_answer(text)


def true_coconut_generate(
    model,
    tokenizer,
    question: str,
    max_logic_new_tokens: int,
    max_final_new_tokens: int,
    handoff_suffix: str,
    virtual_token_window: int = 1,
    injection_mode: str = "input",
    mid_layer_index: int = 12,
    mid_layer_scale: float = 1.0,
    log_step_cosine: bool = False,
    swiglu_bridge: Optional[torch.nn.Module] = None,
    contrastive_alpha: float = 0.0,
) -> Dict[str, str]:
    # Step 1: system-1 logic pass and extract final hidden state window.
    logic_prompt = build_logic_prompt(question)
    logic_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    logic_new_ids, h_window = _greedy_logic_with_hidden_window(
        model=model,
        start_ids=logic_ids,
        max_new_tokens=max_logic_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        window_size=virtual_token_window,
    )
    logic_trace = tokenizer.decode(logic_new_ids[0], skip_special_tokens=True)

    # Step 2/3: adapter-off, assemble soft prompt with virtual token h_T.
    prefix = build_final_prompt_prefix(question)
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    suffix_ids = tokenizer(handoff_suffix, return_tensors="pt").input_ids.to(model.device)

    emb_layer = model.get_input_embeddings()
    step_cosine: List[Dict[str, float]] = []
    with adapter_disabled(model):
        prefix_emb = emb_layer(prefix_ids)  # [1, Lp, H]
        suffix_emb = emb_layer(suffix_ids)  # [1, Ls, H]
        virtual = h_window.to(prefix_emb.dtype)  # [1, N, H]
        if swiglu_bridge is not None:
            virtual = swiglu_bridge(virtual)

        if injection_mode == "input":
            concat_emb = torch.cat([prefix_emb, virtual, suffix_emb], dim=1)
            inject_factory = lambda: nullcontext()
        elif injection_mode == "midlayer":
            # Mid-brain bypass: let EN prompt flow, inject final logic vector at target layer.
            concat_emb = torch.cat([prefix_emb, suffix_emb], dim=1)
            inject_factory = lambda: mid_layer_injector(
                model=model,
                layer_index=mid_layer_index,
                inject_state=virtual[:, -1:, :],
                scale=mid_layer_scale,
                persistent=False,
            )
        elif injection_mode == "midlayer_persistent":
            # H4: continuously anchor residual stream across prefill+decode.
            concat_emb = torch.cat([prefix_emb, suffix_emb], dim=1)
            inject_factory = lambda: mid_layer_injector(
                model=model,
                layer_index=mid_layer_index,
                inject_state=virtual[:, -1:, :],
                scale=mid_layer_scale,
                persistent=True,
            )
        else:
            raise ValueError(f"Unsupported injection_mode: {injection_mode}")

        am = torch.ones((1, concat_emb.shape[1]), dtype=torch.long, device=concat_emb.device)
        if contrastive_alpha > 0.0:
            final_answer, step_cosine = _contrastive_decode_from_embeds(
                model=model,
                inputs_embeds=concat_emb,
                attention_mask=am,
                max_new_tokens=max_final_new_tokens,
                tokenizer=tokenizer,
                alpha=contrastive_alpha,
                inject_factory=inject_factory,
                reference_states=virtual if log_step_cosine else None,
            )
        else:
            with inject_factory():
                if log_step_cosine:
                    final_answer, step_cosine = _generate_from_embeds_with_step_cosine(
                        model=model,
                        inputs_embeds=concat_emb,
                        attention_mask=am,
                        max_new_tokens=max_final_new_tokens,
                        tokenizer=tokenizer,
                        reference_states=virtual,
                    )
                else:
                    final_answer = _generate_from_embeds(
                        model=model,
                        inputs_embeds=concat_emb,
                        attention_mask=am,
                        max_new_tokens=max_final_new_tokens,
                        tokenizer=tokenizer,
                    )
    return {
        "logic_trace": logic_trace,
        "final_answer": final_answer,
        "virtual_token_dim": str(int(h_window.shape[-1])),
        "virtual_token_count": str(int(h_window.shape[1])),
        "injection_mode": injection_mode,
        "step_cosine": step_cosine,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True Coconut: virtual-token handoff (no KV injection).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--handoff-suffix", type=str, default="\nTherefore, the final answer is ")
    p.add_argument("--virtual-token-window", type=int, default=1)
    p.add_argument("--injection-mode", type=str, choices=["input", "midlayer", "midlayer_persistent"], default="input")
    p.add_argument("--mid-layer-index", type=int, default=12)
    p.add_argument("--mid-layer-scale", type=float, default=1.0)
    p.add_argument("--swiglu-bridge", type=Path, default=None)
    p.add_argument("--contrastive-alpha", type=float, default=0.0)
    p.add_argument("--log-step-cosine", action="store_true")
    p.add_argument("--output", type=Path, default=Path("runs/true_coconut_eval.json"))
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
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()
    bridge = _load_swiglu_bridge(
        path=args.swiglu_bridge,
        hidden_size=int(model.config.hidden_size),
        device=model.device,
        dtype=model.dtype,
    )

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = test[: args.sample_size]

    rows: List[EvalRow] = []
    for item in sample:
        err = None
        logic_trace = ""
        with adapter_disabled(model):
            base_pred = baseline_generate(model, tokenizer, item.prompt, args.max_final_new_tokens)
        try:
            coco = true_coconut_generate(
                model=model,
                tokenizer=tokenizer,
                question=item.prompt,
                max_logic_new_tokens=args.max_logic_new_tokens,
                max_final_new_tokens=args.max_final_new_tokens,
                handoff_suffix=args.handoff_suffix,
                virtual_token_window=args.virtual_token_window,
                injection_mode=args.injection_mode,
                mid_layer_index=args.mid_layer_index,
                mid_layer_scale=args.mid_layer_scale,
                log_step_cosine=args.log_step_cosine,
                swiglu_bridge=bridge,
                contrastive_alpha=args.contrastive_alpha,
            )
            coconut_pred = coco["final_answer"]
            logic_trace = coco["logic_trace"]
            step_cosine = coco.get("step_cosine")
            virtual_token_count = int(coco.get("virtual_token_count", "1"))
        except Exception as exc:
            coconut_pred = ""
            err = f"{type(exc).__name__}: {exc}"
            step_cosine = None
            virtual_token_count = int(args.virtual_token_window)
        rows.append(
            EvalRow(
                problem_id=item.problem_id,
                question=item.prompt,
                expected=item.answer,
                base_pred=base_pred,
                coconut_pred=coconut_pred,
                base_ok=answers_match(item.answer, base_pred),
                coconut_ok=answers_match(item.answer, coconut_pred),
                logic_trace=logic_trace,
                virtual_token_count=virtual_token_count,
                injection_mode=args.injection_mode,
                step_cosine=step_cosine,
                error=err,
            )
        )

    total = len(rows)
    base_correct = sum(int(r.base_ok) for r in rows)
    coconut_correct = sum(int(r.coconut_ok) for r in rows)
    errors = sum(int(r.error is not None) for r in rows)
    payload: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "virtual_token_window": args.virtual_token_window,
        "injection_mode": args.injection_mode,
        "mid_layer_index": args.mid_layer_index if args.injection_mode == "midlayer" else None,
        "mid_layer_scale": args.mid_layer_scale if args.injection_mode in {"midlayer", "midlayer_persistent"} else None,
        "swiglu_bridge": str(args.swiglu_bridge) if args.swiglu_bridge is not None else None,
        "contrastive_alpha": args.contrastive_alpha,
        "log_step_cosine": bool(args.log_step_cosine),
        "summary": {
            "total": total,
            "base_correct": base_correct,
            "coconut_correct": coconut_correct,
            "base_acc": (base_correct / total) if total else 0.0,
            "coconut_acc": (coconut_correct / total) if total else 0.0,
            "coconut_lift": ((coconut_correct - base_correct) / total) if total else 0.0,
            "errors": errors,
        },
        "rows": [asdict(r) for r in rows],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"base acc: {payload['summary']['base_acc']:.3f}")
    print(f"true coconut acc: {payload['summary']['coconut_acc']:.3f}")
    print(f"true coconut lift: {payload['summary']['coconut_lift']:+.3f}")
    print(f"errors: {errors}")


if __name__ == "__main__":
    main()
