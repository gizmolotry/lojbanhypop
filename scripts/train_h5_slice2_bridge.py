from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import sys

# Ensure src is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.experiment import generate_dataset, split_dataset


@dataclass
class TrainItem:
    prompt: str
    answer: str


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


def _resolve_layers(model):
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
def persistent_advisor_hook(model, layer_index: int, adapter_module, advisor_states: torch.Tensor, advisor_ids: torch.Tensor, scale: float):
    layers = _resolve_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for persistent advisor hook.")
    if layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError(f"layer_index out of range: {layer_index} (layers={len(layers)})")

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        delta = adapter_module(
            hidden_states=hidden,
            advisor_states=advisor_states,
            advisor_ids=advisor_ids,
        ).to(dtype=hidden.dtype, device=hidden.device)
        hidden = hidden + (float(scale) * delta)
        if rest is None:
            return hidden
        return (hidden, *rest)

    handle = layers[layer_index].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


class BooleanAnchorTable(torch.nn.Module):
    def __init__(self, codebook_size: int, hidden_size: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.hidden_size = int(hidden_size)
        self.emb = torch.nn.Parameter(torch.empty(self.codebook_size, self.hidden_size))
        torch.nn.init.normal_(self.emb, mean=0.0, std=0.02)
        anchors = torch.zeros(5, self.hidden_size)
        eye = torch.eye(min(5, self.hidden_size))
        anchors[:, : eye.shape[1]] = eye
        self.register_buffer("anchor_values", anchors)
        grad_mask = torch.ones(self.codebook_size, self.hidden_size)
        grad_mask[:5, :] = 0.0
        self.register_buffer("grad_mask", grad_mask)
        self.enforce_anchor_values()

    @torch.no_grad()
    def enforce_anchor_values(self) -> None:
        self.emb.data[:5, :] = self.anchor_values.to(self.emb.device, dtype=self.emb.dtype)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h = z.shape
        flat = z.reshape(b * t, h)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = torch.argmin(dist, dim=1)
        z_q = self.emb[idx].view(b, t, h)
        z_st = z + (z_q - z).detach()
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        return z_st, idx.view(b, t), codebook_loss, commit_loss


class AdvisorCrossAttentionAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, advisor_states: torch.Tensor, advisor_ids: torch.Tensor) -> torch.Tensor:
        b, l, h = advisor_states.shape
        if l % 3 != 0:
            q = self.q_proj(hidden_states)
            k = self.k_proj(advisor_states)
            v = self.v_proj(advisor_states)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(h))
            attn = torch.softmax(scores, dim=-1)
            return self.out_proj(torch.matmul(attn, v))

        triplets_state = advisor_states.view(b, l // 3, 3, h)
        triplets_ids = advisor_ids.view(b, l // 3, 3)
        v_projected = self.v_proj(triplets_state)
        
        effective_v = []
        for i in range(l // 3):
            rel_id = triplets_ids[:, i, 0]
            v_var1 = v_projected[:, i, 1]
            v_var2 = v_projected[:, i, 2]
            v_rel = v_projected[:, i, 0]

            is_and = (rel_id == 0).float().view(-1, 1)
            is_or = (rel_id == 1).float().view(-1, 1)
            is_not = (rel_id == 2).float().view(-1, 1)
            is_implies = (rel_id == 3).float().view(-1, 1)
            is_xor = (rel_id == 4).float().view(-1, 1)
            is_learned = (rel_id >= 5).float().view(-1, 1)

            v_and = torch.min(v_var1, v_var2)
            v_or = torch.max(v_var1, v_var2)
            v_not = -v_var1
            v_implies = torch.max(-v_var1, v_var2)
            v_xor = torch.abs(v_var1 - v_var2)

            res = (is_and * v_and + is_or * v_or + is_not * v_not + is_implies * v_implies + is_xor * v_xor + is_learned * v_rel)
            effective_v.append(res.unsqueeze(1))

        v_final = torch.cat(effective_v, dim=1)
        k_final = self.k_proj(triplets_state[:, :, 0, :])
        q = self.q_proj(hidden_states)
        scores = torch.matmul(q, k_final.transpose(-1, -2)) / math.sqrt(float(h))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_final)
        return self.out_proj(context)


class AdvisorArityHead(torch.nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int, pointer_vocab: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.pointer_vocab = int(pointer_vocab)
        self.head_rel = torch.nn.Linear(hidden_size, self.codebook_size, bias=True)
        self.head_var1 = torch.nn.Linear(hidden_size, self.pointer_vocab, bias=True)
        self.head_var2 = torch.nn.Linear(hidden_size, self.pointer_vocab, bias=True)
        self.pointer_emb = torch.nn.Embedding(pointer_vocab, hidden_size)

    def decode_with_arity(self, latent: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        b, l, h = latent.shape
        all_tokens = []
        all_logits = []
        for i in range(l):
            z = latent[:, i, :]
            l_rel = self.head_rel(z)
            t_rel = torch.argmax(l_rel, dim=-1)
            l_v1 = self.head_var1(z)
            t_v1 = torch.argmax(l_v1, dim=-1) + self.codebook_size
            l_v2 = self.head_var2(z)
            t_v2 = torch.argmax(l_v2, dim=-1) + self.codebook_size
            all_tokens.extend([t_rel, t_v1, t_v2])
            all_logits.extend([l_rel, l_v1, l_v2])
        return all_tokens, all_logits, 0

    def token_to_embedding(self, token: torch.Tensor, codebook: BooleanAnchorTable) -> torch.Tensor:
        b = token.shape[0]
        emb = []
        for i in range(b):
            tid = int(token[i].item())
            if tid < self.codebook_size:
                emb.append(codebook.emb[tid : tid + 1, :])
            else:
                pid = tid - self.codebook_size
                emb.append(self.pointer_emb.weight[pid : pid + 1, :])
        return torch.cat(emb, dim=0).unsqueeze(1)


def build_logic_prompt(q: str) -> str:
    return f"You are a rigid symbolic reasoner.\nOutput a concise symbolic TRACE and then ANSWER.\n\nQUESTION: {q}\nTRACE:"


def build_final_prefix(q: str) -> str:
    return f"Solve the logic question. Return only the final answer with no explanation.\n\nQuestion: {q}\nFinal answer:"


def extract_trace_hidden_states(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    logic_prompt = build_logic_prompt(question)
    ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True, output_hidden_states=True)
    hiddens = [out.hidden_states[-1][:, -1:, :]]
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_len = ids.shape[1] + 1
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        return hiddens[0]
    for _ in range(max_logic_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model(input_ids=tok, attention_mask=am, past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        hiddens.append(out.hidden_states[-1][:, -1:, :])
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break
    return torch.cat(hiddens, dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H5 slice-2: train Cross-Attention bridge with frozen dictionary.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--slice1-checkpoint", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-root", type=Path, default=Path("runs/i_series"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"slice2_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = "auto" if torch.cuda.is_available() else None
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map=device_map)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map=device_map)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    train_rows: List[TrainItem] = []
    for sd in args.seeds:
        ds = generate_dataset(size=args.dataset_size, seed=int(sd))
        _, _, test = split_dataset(ds)
        for r in test[: args.sample_size]:
            train_rows.append(TrainItem(prompt=r.prompt, answer=r.answer))
    hidden_size = int(model.config.hidden_size)
    checkpoint = torch.load(args.slice1_checkpoint, map_location=model.device)
    codebook = BooleanAnchorTable(codebook_size=checkpoint["codebook_size"], hidden_size=hidden_size).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(checkpoint["codebook_state"])
    codebook.enforce_anchor_values()
    for p in codebook.parameters():
        p.requires_grad = False
    arity_head = AdvisorArityHead(hidden_size=hidden_size, codebook_size=checkpoint["codebook_size"], pointer_vocab=checkpoint["pointer_vocab_size"]).to(model.device, dtype=model.dtype)
    arity_head.load_state_dict(checkpoint["arity_head_state"])
    for p in arity_head.parameters():
        p.requires_grad = False
    advisor_adapter = AdvisorCrossAttentionAdapter(hidden_size=hidden_size).to(model.device, dtype=model.dtype)
    advisor_adapter.load_state_dict(checkpoint["advisor_adapter_state"])
    opt = torch.optim.AdamW(advisor_adapter.parameters(), lr=args.lr)
    emb = model.get_input_embeddings()
    ce_hist = []
    for step in range(args.train_steps):
        item = train_rows[step % len(train_rows)]
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=48).to(model.dtype)
            _, idx, _, _ = codebook.quantize(h_t)
            # Input to decode_with_arity needs to be [B, L, H]
            z_q = codebook.emb[idx].view(1, idx.shape[1], -1)
            tokens, _, _ = arity_head.decode_with_arity(z_q)
            advisor_states_list = [arity_head.token_to_embedding(tok, codebook) for tok in tokens]
            advisor_state = torch.cat(advisor_states_list, dim=1)
            advisor_ids = torch.stack(tokens, dim=1)
        prefix = build_final_prefix(item.prompt)
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :6]
        if int(target_ids.numel()) == 0:
            continue
        opt.zero_grad()
        with adapter_disabled(model):
            prefix_emb = emb(prefix_ids)
            ce = torch.tensor(0.0, device=model.device, dtype=model.dtype)
            cur_emb = prefix_emb
            for t in range(target_ids.shape[1]):
                am = torch.ones((1, cur_emb.shape[1]), dtype=torch.long, device=model.device)
                with persistent_advisor_hook(model=model, layer_index=args.layer_index, adapter_module=advisor_adapter, advisor_states=advisor_state, advisor_ids=advisor_ids, scale=args.inject_scale):
                    out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
                logits = out.logits[:, -1, :]
                ce = ce + F.cross_entropy(logits, target_ids[:, t])
                cur_emb = torch.cat([cur_emb, emb(target_ids[:, t : t + 1])], dim=1)
        ce.backward()
        opt.step()
        ce_hist.append(float(ce.item()))
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{args.train_steps} - CE: {sum(ce_hist[-10:])/10:.4f}")
    ckpt_path = run_dir / "h5_slice2_bridge.pt"
    torch.save({
        "advisor_adapter_state": advisor_adapter.state_dict(),
        "ce_history": ce_hist,
        "slice1_checkpoint": str(args.slice1_checkpoint)
    }, ckpt_path)
    print(f"Wrote: {ckpt_path}")

if __name__ == "__main__":
    main()
