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
    # ids 0..4 are fixed boolean anchors (AND, OR, NOT, IMPLIES, XOR)
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

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, 1, H]
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
        # hidden_states: [B,T,H], advisor_states: [B,L,H], advisor_ids: [B,L]
        b, l, h = advisor_states.shape
        
        # Ensure we can process as triplets
        if l % 3 != 0:
            q = self.q_proj(hidden_states)
            k = self.k_proj(advisor_states)
            v = self.v_proj(advisor_states)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(h))
            attn = torch.softmax(scores, dim=-1)
            return self.out_proj(torch.matmul(attn, v))

        # Reshape to triplets: [B, L//3, 3, H]
        triplets_state = advisor_states.view(b, l // 3, 3, h)
        triplets_ids = advisor_ids.view(b, l // 3, 3)

        v_projected = self.v_proj(triplets_state) # [B, L//3, 3, H]
        
        effective_v = []
        for i in range(l // 3):
            rel_id = triplets_ids[:, i, 0]
            v_rel = v_projected[:, i, 0]
            v_var1 = v_projected[:, i, 1]
            v_var2 = v_projected[:, i, 2]

            # Boolean anchors (0..4)
            is_and = (rel_id == 0).float().view(-1, 1)
            is_or = (rel_id == 1).float().view(-1, 1)
            is_not = (rel_id == 2).float().view(-1, 1)
            is_implies = (rel_id == 3).float().view(-1, 1)
            is_xor = (rel_id == 4).float().view(-1, 1)
            is_learned = (rel_id >= 5).float().view(-1, 1)

            # Physically execute logical intersection/union on manifolds
            v_and = torch.min(v_var1, v_var2)
            v_or = torch.max(v_var1, v_var2)
            v_not = -v_var1
            v_implies = torch.max(-v_var1, v_var2)
            v_xor = torch.abs(v_var1 - v_var2)

            res = (is_and * v_and +
                   is_or * v_or +
                   is_not * v_not +
                   is_implies * v_implies +
                   is_xor * v_xor +
                   is_learned * v_rel)
            effective_v.append(res.unsqueeze(1))

        v_final = torch.cat(effective_v, dim=1) # [B, L//3, H]
        # Use REL token geometry for Keys
        k_final = self.k_proj(triplets_state[:, :, 0, :]) # [B, L//3, H]

        q = self.q_proj(hidden_states)
        scores = torch.matmul(q, k_final.transpose(-1, -2)) / math.sqrt(float(h))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_final)
        return self.out_proj(context)


class AdvisorArityHead(torch.nn.Module):
    # Output space: relation ids [0..codebook_size-1], pointer ids [codebook_size..codebook_size+pointer_vocab-1]
    def __init__(self, hidden_size: int, codebook_size: int, pointer_vocab: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.pointer_vocab = int(pointer_vocab)
        self.total_vocab = int(codebook_size + pointer_vocab)
        # Separate heads for structural enforcement
        self.head_rel = torch.nn.Linear(hidden_size, self.codebook_size, bias=True)
        self.head_var1 = torch.nn.Linear(hidden_size, self.pointer_vocab, bias=True)
        self.head_var2 = torch.nn.Linear(hidden_size, self.pointer_vocab, bias=True)
        self.pointer_emb = torch.nn.Embedding(pointer_vocab, hidden_size)

    def decode_with_arity(self, latent: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        # latent: [B, L, H]
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
        # token: [B]
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
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise symbolic TRACE and then ANSWER.\n\n"
        f"QUESTION: {q}\n"
        "TRACE:"
    )


def build_final_prefix(q: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {q}\n"
        "Final answer:"
    )


def extract_trace_hidden_states(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    logic_prompt = build_logic_prompt(question)
    ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    hiddens = [out.hidden_states[-1][:, -1:, :]]
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_len = ids.shape[1] + 1
    
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        return hiddens[0]
        
    for _ in range(max_logic_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=model.device)
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
        hiddens.append(out.hidden_states[-1][:, -1:, :])
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break
    return torch.cat(hiddens, dim=1)


def _kmeans_centroids(x: torch.Tensor, k: int, iters: int = 6) -> torch.Tensor:
    # x: [N,H]
    n = int(x.shape[0])
    if n <= 0:
        return x
    k = max(1, min(k, n))
    centroids = x[torch.randperm(n, device=x.device)[:k]].clone()
    for _ in range(iters):
        dist = torch.cdist(x.float(), centroids.float())
        assign = torch.argmin(dist, dim=1)
        next_c = []
        for i in range(k):
            pts = x[assign == i]
            if int(pts.shape[0]) == 0:
                next_c.append(centroids[i : i + 1])
            else:
                next_c.append(pts.mean(dim=0, keepdim=True))
        centroids = torch.cat(next_c, dim=0)
    return centroids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H5 slice-1: persistent VQ advisor with arity and anti-collapse.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--target-answer-tokens", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--codebook-size", type=int, default=2000)
    p.add_argument("--pointer-vocab-size", type=int, default=32)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--dead-code-steps", type=int, default=100)
    p.add_argument("--mdl-warmup-steps", type=int, default=100)
    p.add_argument("--mdl-max-lambda", type=float, default=0.02)
    p.add_argument("--commitment-weight", type=float, default=0.25)
    p.add_argument("--identity-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-root", type=Path, default=Path("runs/true_coconut_h5"))
    p.add_argument("--output-extension-json", type=Path, default=Path("runs/coconut_ablation_matrix/20260226_090029/ablation_matrix_extensions.json"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / ts
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
    if not train_rows:
        raise RuntimeError("No training rows built from dataset.")

    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(codebook_size=args.codebook_size, hidden_size=hidden_size).to(model.device, dtype=model.dtype)
    advisor_adapter = AdvisorCrossAttentionAdapter(hidden_size=hidden_size).to(model.device, dtype=model.dtype)
    arity_head = AdvisorArityHead(hidden_size=hidden_size, codebook_size=args.codebook_size, pointer_vocab=args.pointer_vocab_size).to(
        model.device, dtype=model.dtype
    )
    params = list(codebook.parameters()) + list(advisor_adapter.parameters()) + list(arity_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    usage = torch.zeros(args.codebook_size, dtype=torch.long, device=model.device)
    stale = torch.zeros(args.codebook_size, dtype=torch.long, device=model.device)
    stale[:5] = -10**6  # never revive fixed anchors
    pointer_bank: Dict[int, torch.Tensor] = {}
    emb = model.get_input_embeddings()

    loss_hist: List[float] = []
    ce_hist: List[float] = []
    arity_violations = 0
    advisor_lens: List[int] = []
    revivals = 0

    for step in range(args.train_steps):
        item = train_rows[step % len(train_rows)]
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=args.max_logic_new_tokens).to(model.dtype)

        z_st, idx, cb_loss, commit_loss = codebook.quantize(h_t)
        
        # update usage for all codes in trace
        for i in range(idx.shape[1]):
            relation_id = int(idx[0, i].item())
            usage[relation_id] += 1
            stale[relation_id] = 0

        tokens, advisor_logits, violation = arity_head.decode_with_arity(z_st)
        advisor_len = len(tokens)
        advisor_lens.append(advisor_len)

        # arity CE to keep constrained head well-defined and structurally rigid
        arity_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
        for i in range(idx.shape[1]):
            target_rel = idx[0, i]
            # Relation head must match VQ selection
            lg_rel = advisor_logits[i * 3]
            arity_loss = arity_loss + F.cross_entropy(lg_rel, target_rel.view(-1).long())
            # Variable heads use entropy minimization (self-reinforcement)
            lg_v1 = advisor_logits[i * 3 + 1]
            lg_v2 = advisor_logits[i * 3 + 2]
            arity_loss = arity_loss + F.cross_entropy(lg_v1, torch.argmax(lg_v1, dim=-1).view(-1).long())
            arity_loss = arity_loss + F.cross_entropy(lg_v2, torch.argmax(lg_v2, dim=-1).view(-1).long())

        advisor_states = [arity_head.token_to_embedding(tok, codebook) for tok in tokens]
        advisor_state = torch.cat(advisor_states, dim=1)
        advisor_ids = torch.stack(tokens, dim=1)

        # identity rigidity loss (Contrastive Pointer Identity)
        # Enforce orthogonality between pointer embeddings to prevent semantic smearing
        all_p = arity_head.pointer_emb.weight
        cos_sim = F.cosine_similarity(all_p.unsqueeze(1), all_p.unsqueeze(0), dim=-1)
        # mask diagonal
        cos_sim = cos_sim - torch.eye(cos_sim.shape[0], device=cos_sim.device)
        identity_loss = (cos_sim.pow(2).sum()) / (cos_sim.shape[0] * (cos_sim.shape[0] - 1))

        prefix = build_final_prefix(item.prompt)
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        if int(target_ids.numel()) == 0:
            continue
        target_ids = target_ids[:, : args.target_answer_tokens]

        opt.zero_grad(set_to_none=True)
        with adapter_disabled(model):
            prefix_emb = emb(prefix_ids)
            ce = torch.tensor(0.0, device=model.device, dtype=model.dtype)
            cur_emb = prefix_emb
            for t in range(target_ids.shape[1]):
                am = torch.ones((1, cur_emb.shape[1]), dtype=torch.long, device=model.device)
                with persistent_advisor_hook(
                    model=model,
                    layer_index=args.layer_index,
                    adapter_module=advisor_adapter,
                    advisor_states=advisor_state,
                    advisor_ids=advisor_ids,
                    scale=args.inject_scale,
                ):
                    out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
                logits = out.logits[:, -1, :]
                ce = ce + F.cross_entropy(logits, target_ids[:, t])
                next_emb = emb(target_ids[:, t : t + 1])
                cur_emb = torch.cat([cur_emb, next_emb], dim=1)

        mdl_lambda = 0.0
        if step >= args.mdl_warmup_steps:
            prog = float(step - args.mdl_warmup_steps + 1) / float(max(1, args.train_steps - args.mdl_warmup_steps))
            mdl_lambda = float(args.mdl_max_lambda) * min(1.0, max(0.0, prog))

        # Rate-distortion adjustment
        with torch.no_grad():
            pred0 = int(torch.argmax(logits, dim=-1).item())
            tgt0 = int(target_ids[:, -1].item())
            if pred0 == tgt0:
                mdl_lambda *= 1.1
            else:
                mdl_lambda *= 0.9

        mdl_len = torch.tensor(float(advisor_len), device=model.device, dtype=model.dtype)
        loss = ce + arity_loss + cb_loss + (float(args.commitment_weight) * commit_loss) + (float(args.identity_weight) * identity_loss) + (mdl_lambda * mdl_len)
        loss.backward()
        opt.step()
        codebook.enforce_anchor_values()

        # dead-code revival (for trainable slice 5..1999)
        stale[5:] += 1
        # dead_ids check
        dead_ids = torch.nonzero(stale[5:] >= int(args.dead_code_steps), as_tuple=False).flatten() + 5
        if int(dead_ids.numel()) > 0:
            with torch.no_grad():
                batch_vecs = h_t.reshape(-1, h_t.shape[-1]).detach()
                cents = _kmeans_centroids(batch_vecs, k=int(dead_ids.numel()))
                for j, did in enumerate(dead_ids.tolist()):
                    codebook.emb.data[did, :] = cents[j % cents.shape[0], :].to(codebook.emb.dtype)
                    stale[did] = 0
                    revivals += 1
                codebook.enforce_anchor_values()

        loss_hist.append(float(loss.item()))
        ce_hist.append(float(ce.item()))
        
        if (step + 1) % 10 == 0:
            avg_loss = sum(loss_hist[-10:]) / 10
            avg_ce = sum(ce_hist[-10:]) / 10
            print(f"Step {step+1}/{args.train_steps} - Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Advisor Len: {advisor_len}, Revivals: {revivals}")

    used = int((usage[5:] > 0).sum().item())
    usage_ratio = float(used / max(1, args.codebook_size - 5))
    arity_violation_rate = 0.0 # structurally enforced now
    advisor_len_mean = float(sum(advisor_lens) / max(1, len(advisor_lens)))

    ckpt_path = run_dir / "h5_codebook_advisor.pt"
    report_path = run_dir / "h5_slice1_report.json"
    torch.save(
        {
            "codebook_state": codebook.state_dict(),
            "advisor_adapter_state": advisor_adapter.state_dict(),
            "arity_head_state": arity_head.state_dict(),
            "codebook_size": args.codebook_size,
            "pointer_vocab_size": args.pointer_vocab_size,
        },
        ckpt_path,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "train_steps": args.train_steps,
        "codebook_size": args.codebook_size,
        "boolean_frozen_count": 5,
        "dead_code_steps": args.dead_code_steps,
        "mdl_warmup_steps": args.mdl_warmup_steps,
        "mdl_max_lambda": args.mdl_max_lambda,
        "metrics": {
            "codes_used": used,
            "usage_ratio": usage_ratio,
            "arity_violation_rate": arity_violation_rate,
            "advisor_len_mean": advisor_len_mean,
            "dead_code_revivals": int(revivals),
            "loss_start": float(loss_hist[0]) if loss_hist else 0.0,
            "loss_end": float(loss_hist[-1]) if loss_hist else 0.0,
            "ce_start": float(ce_hist[0]) if ce_hist else 0.0,
            "ce_end": float(ce_hist[-1]) if ce_hist else 0.0,
        },
        "checkpoint": str(ckpt_path),
        "loss_history_tail": loss_hist[-20:],
        "ce_history_tail": ce_hist[-20:],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # additive-only extension artifact for ablation dashboard
    ext_path = args.output_extension_json
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    ext_payload = {}
    if ext_path.exists():
        try:
            ext_payload = json.loads(ext_path.read_text(encoding="utf-8"))
        except Exception:
            ext_payload = {}
    ext_payload["h5_slice1"] = {
        "timestamp": report["timestamp"],
        "run_dir": str(run_dir),
        "report": str(report_path),
        "checkpoint": str(ckpt_path),
        "metrics": {
            "codes_used": used,
            "usage_ratio": usage_ratio,
            "arity_violation_rate": arity_violation_rate,
            "advisor_len_mean": advisor_len_mean,
        },
    }
    ext_path.write_text(json.dumps(ext_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {ckpt_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {ext_path}")
    print(f"codes_used: {used}")
    print(f"usage_ratio: {usage_ratio:.4f}")
    print(f"arity_violation_rate: {arity_violation_rate:.4f}")
    print(f"advisor_len_mean: {advisor_len_mean:.4f}")


if __name__ == "__main__":
    main()

