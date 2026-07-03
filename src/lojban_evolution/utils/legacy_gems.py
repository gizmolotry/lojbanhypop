import re
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")

def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("in the ", "").replace("the ", "")
    return NON_ALNUM_RE.sub("", lowered)

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

def _resolve_decoder_layers(model):
    for root in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        inner = getattr(root, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    return None

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
