from __future__ import annotations

import argparse
import json
import math
import random
import re
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
def persistent_advisor_hook(model, layer_index: int, adapter_module, advisor_states: torch.Tensor, advisor_ids: torch.Tensor, pointer_ids: torch.Tensor, scale: float):
    layers = _resolve_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for persistent advisor hook.")
    
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        t_curr = hidden.shape[1]
        t_total = pointer_ids.shape[1]
        curr_p_ids = pointer_ids[:, t_total - t_curr:] if t_total > t_curr else pointer_ids
        
        delta = adapter_module(
            hidden_states=hidden,
            advisor_states=advisor_states,
            advisor_ids=advisor_ids,
            pointer_ids=curr_p_ids,
        ).to(dtype=hidden.dtype, device=hidden.device)
        hidden = hidden + (float(scale) * delta)
        return (hidden, *rest) if rest is not None else hidden

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
        self.emb.register_hook(lambda g: g * self.grad_mask.to(g.device, dtype=g.dtype))
        self.enforce_anchor_values()

    @torch.no_grad()
    def enforce_anchor_values(self) -> None:
        self.emb.data[:5, :] = self.anchor_values.to(self.emb.device, dtype=self.emb.dtype)

    def quantize(self, z: torch.Tensor, relation_bias: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h = z.shape
        flat = z.reshape(b * t, h)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        if relation_bias > 0:
            dist[:, :5] -= float(relation_bias)
        idx = torch.argmin(dist, dim=1)
        z_q = self.emb[idx].view(b, t, h)
        z_st = z + (z_q - z).detach()
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        return z_st, idx.view(b, t), codebook_loss, commit_loss


class CouncilCrossAttentionAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int, use_boolean_surgery: bool = True):
        super().__init__()
        # I-2: Council Heads
        self.q_judge = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_judge = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_judge = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.q_intuitor = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_intuitor = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_intuitor = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.use_boolean_surgery = use_boolean_surgery
        self.gain = torch.nn.Parameter(torch.tensor(0.1))
        self.council_weights = torch.nn.Parameter(torch.ones(2)) # [Judge, Intuitor]

    def forward(self, hidden_states: torch.Tensor, advisor_states: torch.Tensor, advisor_ids: torch.Tensor, pointer_ids: torch.Tensor) -> torch.Tensor:
        b, t, h = hidden_states.shape
        l = advisor_states.shape[1]
        
        p = pointer_ids.unsqueeze(-1)
        indices = torch.cat([p, p+1, p+2], dim=-1).clamp(0, l - 1)
        batch_idx = torch.arange(b, device=hidden_states.device).view(b, 1, 1).expand(b, t, 3)
        win_states = advisor_states[batch_idx, indices, :]
        win_ids = advisor_ids[batch_idx, indices]
        
        # Head 1: The Judge (Log-Space Surgery)
        qj = self.q_judge(hidden_states).unsqueeze(2)
        kj = self.k_judge(win_states)
        vj = self.v_judge(win_states)
        sj = torch.matmul(qj, kj.transpose(-1, -2)).squeeze(2) / math.sqrt(float(h))
        
        if self.use_boolean_surgery:
            l_rel, l_v1, l_v2 = sj[:, :, 0], sj[:, :, 1], sj[:, :, 2]
            rel_ids = win_ids[:, :, 0]
            is_and, is_or, is_not, is_implies, is_xor = [(rel_ids == i).float() for i in range(5)]
            
            # Wired Surgery for ALL 5 tokens
            l_new = is_and * (l_v1 + l_v2)
            l_new = l_new + is_or * torch.max(l_v1, l_v2)
            l_new = l_new + is_not * (-l_v1)
            l_new = l_new + is_implies * torch.max(-l_v1, l_v2)
            l_new = l_new + is_xor * torch.abs(l_v1 - l_v2)
            
            mask_any = (is_and + is_or + is_not + is_implies + is_xor).clamp(0, 1)
            l_final = mask_any * l_new + (1.0 - mask_any) * l_rel
            sj = torch.stack([l_final, l_v1, l_v2], dim=-1)
            
        aj = torch.softmax(sj, dim=-1)
        cj = torch.matmul(aj.unsqueeze(2), vj).squeeze(2)
        
        # Head 2: The Intuitor (Fuzzy Context)
        qi = self.q_intuitor(hidden_states).unsqueeze(2)
        ki = self.k_intuitor(win_states)
        vi = self.v_intuitor(win_states)
        si = torch.matmul(qi, ki.transpose(-1, -2)).squeeze(2) / math.sqrt(float(h))
        ai = torch.softmax(si, dim=-1)
        ci = torch.matmul(ai.unsqueeze(2), vi).squeeze(2)
        
        # Council Consensus
        w = torch.softmax(self.council_weights, dim=0)
        context = w[0] * cj + w[1] * ci
        return self.out_proj(context) * self.gain


class AdvisorArityHead(torch.nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.head_rel = torch.nn.Linear(hidden_size, self.codebook_size, bias=True)
        self.head_var1 = torch.nn.Linear(hidden_size, self.codebook_size, bias=True)
        self.head_var2 = torch.nn.Linear(hidden_size, self.codebook_size, bias=True)

    def decode_with_arity(self, latent: torch.Tensor, use_iron_collar: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        b, l, h = latent.shape
        all_tokens, all_logits = [], []
        for i in range(l):
            z = latent[:, i, :]
            l_rel = self.head_rel(z)
            if use_iron_collar:
                m = torch.full_like(l_rel, -1e9); m[:, :5] = 0; l_rel = l_rel + m
            t_rel = torch.argmax(l_rel, dim=-1)
            
            l_v1 = self.head_var1(z)
            if use_iron_collar:
                m = torch.full_like(l_v1, -1e9); m[:, 5:] = 0; l_v1 = l_v1 + m
            t_v1 = torch.argmax(l_v1, dim=-1)
            
            l_v2 = self.head_var2(z)
            if use_iron_collar:
                l_v2 = l_v2 + m
            t_v2 = torch.argmax(l_v2, dim=-1)
            
            all_tokens.extend([t_rel, t_v1, t_v2])
            all_logits.extend([l_rel, l_v1, l_v2])
        return all_tokens, all_logits, 0

    def token_to_embedding(self, token: torch.Tensor, codebook: BooleanAnchorTable) -> torch.Tensor:
        return codebook.emb[token].unsqueeze(1)


def build_logic_prompt(q: str) -> str:
    return f"You are a rigid symbolic reasoner.\nOutput must contain a symbolic TRACE line and an ANSWER line.\n\nQUESTION: {q}\nTRACE:"


def build_final_prefix(q: str) -> str:
    return f"Solve the logic question. Return only the final answer with no explanation.\n\nQuestion: {q}\nFinal answer:"


def extract_trace_hidden_states(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    p = build_logic_prompt(question)
    ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True, output_hidden_states=True)
    hiddens = [out.hidden_states[-1][:, -1:, :]]
    past, tok = out.past_key_values, torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_len = ids.shape[1] + 1
    for _ in range(max_logic_new_tokens - 1):
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id): break
        with torch.no_grad():
            out = model(input_ids=tok, attention_mask=torch.ones((1, cur_len + 1), device=model.device), past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
        past, tok = out.past_key_values, torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        hiddens.append(out.hidden_states[-1][:, -1:, :])
        cur_len += 1
    return torch.cat(hiddens, dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H5 'I' Step: Integration and Insistence.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--teacher-checkpoint", type=Path, help="I-1: H5.3 Dark Reasoner for distillation.")
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use-iron-collar", action="store_true")
    p.add_argument("--variable-warmup", action="store_true")
    p.add_argument("--distill-weight", type=float, default=0.1)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/i_series/{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    
    # Load adapter with 151701 vocab
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")
    
    # Now expand for [ADVANCE] (151702)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ADVANCE]"]})
    advance_id = tokenizer.convert_tokens_to_ids("[ADVANCE]")
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    advisor_adapter = CouncilCrossAttentionAdapter(hidden_size).to(model.device, dtype=model.dtype)
    arity_head = AdvisorArityHead(hidden_size, 2000).to(model.device, dtype=model.dtype)

    teacher_codebook = None
    if args.teacher_checkpoint:
        print(f"I-1 DISTILLATION: Loading teacher from {args.teacher_checkpoint}")
        tc = torch.load(args.teacher_checkpoint, map_location=model.device)
        teacher_codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
        teacher_codebook.load_state_dict(tc["codebook_state"])
        for p in teacher_codebook.parameters(): p.requires_grad = False

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=model.device)
        codebook.load_state_dict(ckpt["codebook_state"])
        advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
        # ArityHead resize support
        cs, ps = arity_head.state_dict(), ckpt["arity_head_state"]
        for n, p in ps.items():
            if n in cs and cs[n].shape == p.shape: cs[n].copy_(p)
        arity_head.load_state_dict(cs)

    if args.variable_warmup:
        print("I-3: Variable-Anchor Pre-Warming ACTIVE.")
        params = [{'params': [codebook.emb], 'lr': args.lr}, {'params': list(advisor_adapter.parameters()) + list(arity_head.parameters()), 'lr': args.lr}]
    else:
        params = list(codebook.parameters()) + list(advisor_adapter.parameters()) + list(arity_head.parameters())
    
    opt = torch.optim.AdamW(params, lr=args.lr)
    ds = generate_dataset(size=1000, seed=7)
    _, _, test = split_dataset(ds)

    for step in range(args.train_steps):
        item = test[step % len(test)]
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, 48).to(model.dtype)
        
        z_st, idx, cb_loss, commit_loss = codebook.quantize(h_t)
        tokens, logits, _ = arity_head.decode_with_arity(z_st, use_iron_collar=args.use_iron_collar)
        adv_state = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
        adv_ids = torch.stack(tokens, dim=1)

        # I-1 Distillation Loss
        distill_loss = torch.tensor(0.0, device=model.device)
        if teacher_codebook:
            with torch.no_grad():
                _, t_idx, _, _ = teacher_codebook.quantize(h_t)
            distill_loss = F.cross_entropy(logits[0], t_idx[0, 0].view(-1).long())

        prefix = build_final_prefix(item.prompt)
        p_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        t_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :12]
        
        opt.zero_grad()
        with adapter_disabled(model):
            ce, cur_emb, ptr = torch.tensor(0.0, device=model.device), model.get_input_embeddings()(p_ids), 0
            for t in range(t_ids.shape[1]):
                ptr_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
                with persistent_advisor_hook(model, 12, advisor_adapter, adv_state, adv_ids, ptr_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, use_cache=False)
                ce += F.cross_entropy(out.logits[:, -1, :], t_ids[:, t])
                cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(t_ids[:, t:t+1])], dim=1)
                # I-4 Handshake Constraint: Advance pointer on [ADVANCE] in target or heuristically
                if int(t_ids[0, t].item()) == advance_id: ptr = min(ptr + 3, len(tokens) - 3)

        loss = ce + cb_loss + (0.25 * commit_loss) + (args.distill_weight * distill_loss)
        loss.backward(); opt.step(); codebook.enforce_anchor_values()
        
        if (step + 1) % 10 == 0:
            w = torch.softmax(advisor_adapter.council_weights, dim=0)
            print(f"Step {step+1}/{args.train_steps} - CE: {ce.item():.4f}, Gain: {advisor_adapter.gain.item():.4f}, Council: [J:{w[0]:.2f}, I:{w[1]:.2f}]")

    torch.save({"codebook_state": codebook.state_dict(), "advisor_adapter_state": advisor_adapter.state_dict(), "arity_head_state": arity_head.state_dict()}, run_dir / "h5_checkpoint.pt")
    print(f"Wrote: {run_dir / 'h5_checkpoint.pt'}")

if __name__ == "__main__":
    main()
