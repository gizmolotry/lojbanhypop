from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import Problem, generate_dataset
from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_11_winograd_failure_anatomy import (
    _answer_match,
    _decode_logic_tokens,
    _gen_controlled_variants,
    _infer_family,
    _taxonomize_failure,
    _triples,
)
from train_h5_persistent_vq_advisor import (
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    adapter_disabled,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)


def _masked_logits(logits: torch.Tensor, legal_start: int, legal_end: int) -> torch.Tensor:
    out = logits.clone()
    if legal_start > 0:
        out[:, :legal_start] = -1e9
    if legal_end < out.shape[-1]:
        out[:, legal_end:] = -1e9
    return out

# --------------------------------------------------------------------------------
# RELATIONAL PROJECTORS
# --------------------------------------------------------------------------------

class RelationalProjector(torch.nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int, mode: str, num_heads: int = 8):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.cue_scale = torch.nn.Parameter(torch.tensor(0.1))
        
        if mode == "LRAP":
            # Learned Relational Attention Projection
            # Mixes heads, then compresses
            self.head_mixer = torch.nn.Linear(num_heads, 1, bias=False)
            self.down = torch.nn.Linear(hidden_size, bottleneck_dim, bias=False)
            self.up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
            
        elif mode == "SVD":
            # Pure math projection bottleneck
            self.up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
            
        elif mode == "ELEMENTS":
            # Decomposes into Type, Polarity, Direction
            # Represented as 3 compressed slots
            self.element_proj = torch.nn.Linear(hidden_size, bottleneck_dim * 3, bias=False)
            self.up = torch.nn.Linear(bottleneck_dim * 3, hidden_size, bias=False)

    def forward(self, attn_weights: torch.Tensor, eng_hidden: torch.Tensor, z_st: torch.Tensor) -> torch.Tensor:
        # attn_weights: [1, num_heads, L, L]
        # eng_hidden: [1, L, H]
        b, l_eng, h = eng_hidden.shape
        
        if self.mode == "SVD":
            # Extract dominant geometric skeleton
            # We take the mean across heads, then SVD the attention map
            a_mean = torch.mean(attn_weights, dim=1)[0] # [L, L]
            u, s, vh = torch.linalg.svd(a_mean.float())
            # Capture top-k singular vectors as the 'shape'
            k = min(self.bottleneck_dim // 2, l_eng)
            shape_vec = (u[:, :k] @ torch.diag(s[:k])).mean(dim=0) # [k]
            # Pad or project to match H
            cue = self.up(F.pad(shape_vec, (0, self.bottleneck_dim - k))).view(1, 1, h)
            
        elif self.mode == "LRAP":
            # Denoise heads -> pooled interaction -> low rank
            # [1, L, L, heads]
            a_perm = attn_weights.permute(0, 2, 3, 1)
            a_mixed = self.head_mixer(a_perm).squeeze(-1) # [1, L, L]
            # Relation-weighted hidden state
            rel_hidden = torch.matmul(a_mixed, eng_hidden) # [1, L, H]
            eng_pooled = torch.mean(rel_hidden, dim=1, keepdim=True)
            cue = self.up(F.relu(self.down(eng_pooled)))
            
        elif self.mode == "ELEMENTS":
            # Attention-weighted entity extraction
            # We assume the most attended tokens are the key entities
            importance = torch.mean(attn_weights, dim=(1, 2)) # [1, L]
            top_hidden = torch.matmul(importance.unsqueeze(1), eng_hidden) # [1, 1, H]
            cue = self.up(F.relu(self.element_proj(top_hidden)))
            
        else:
            return z_st

        cue = torch.tanh(cue) * torch.sigmoid(self.cue_scale)
        return z_st + torch.sigmoid(self.gate) * cue

# --------------------------------------------------------------------------------
# EXPERIMENT ENGINE
# --------------------------------------------------------------------------------

def _extract_attention(model, tokenizer, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with adapter_disabled(model):
        out = model(input_ids=ids, output_attentions=True, output_hidden_states=True)
    # Target middle-layer attention (Layer 12)
    # out.attentions is a tuple of [batch, heads, seq, seq]
    layer_idx = 12
    attn = out.attentions[layer_idx].detach()
    hidden = out.hidden_states[layer_idx].detach()
    return attn, hidden

def _train_cell(
    cell: str,
    projector: RelationalProjector,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    train_ds: List[Problem],
    args: argparse.Namespace
) -> None:
    if cell == "A": return
    
    print(f"\n--- Training Cell {cell} ({projector.mode}) ---")
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    projector.train()
    
    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        with torch.no_grad():
            attn, eng_hidden = _extract_attention(model, tokenizer, build_final_prefix(item.prompt))
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, 48)
            z_st, _, _, _ = codebook.quantize(h_t)
            
        opt.zero_grad()
        z_refined = projector(attn, eng_hidden, z_st)
        
        # Differentiable Decode (Gumbel), with typed masking to match eval decode contract.
        refined_tokens = []
        refined_ids: list[torch.Tensor] = []
        for i in range(z_refined.shape[1]):
            z = z_refined[:, i, :]
            l_rel_raw = arity_head.head_rel(z)
            l_v1_raw = arity_head.head_var1(z)
            l_v2_raw = arity_head.head_var2(z)
            l_rel = _masked_logits(l_rel_raw, 0, int(args.relation_vocab))
            l_v1 = _masked_logits(l_v1_raw, int(args.var_min_id), l_v1_raw.shape[-1])
            l_v2 = _masked_logits(l_v2_raw, int(args.var_min_id), l_v2_raw.shape[-1])

            p_rel = F.gumbel_softmax(l_rel, tau=1.0, hard=True)
            p_v1 = F.gumbel_softmax(l_v1, tau=1.0, hard=True)
            p_v2 = F.gumbel_softmax(l_v2, tau=1.0, hard=True)
            refined_tokens.extend([p_rel @ codebook.emb, p_v1 @ codebook.emb, p_v2 @ codebook.emb])
            refined_ids.extend(
                [
                    torch.argmax(p_rel, dim=-1),
                    torch.argmax(p_v1, dim=-1),
                    torch.argmax(p_v2, dim=-1),
                ]
            )
            
        advisor_states = torch.cat([v.unsqueeze(1) for v in refined_tokens], dim=1)
        advisor_ids = torch.stack(refined_ids, dim=1).to(dtype=torch.long)

        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :6]
        if int(target_ids.numel()) == 0: continue
            
        with adapter_disabled(model):
            emb = model.get_input_embeddings()
            cur_emb = emb(tokenizer(build_final_prefix(item.prompt), return_tensors="pt").input_ids.to(model.device))
            ce = torch.tensor(0.0, device=model.device, dtype=model.dtype)
            ptr = 0
            for t in range(target_ids.shape[1]):
                ptr_ids = torch.full((1, cur_emb.shape[1]), int(ptr), dtype=torch.long, device=model.device)
                with persistent_advisor_hook(model, int(args.layer_index), advisor_adapter, advisor_states, advisor_ids, ptr_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, return_dict=True)
                ce = ce + F.cross_entropy(out.logits[:, -1, :], target_ids[:, t])
                cur_emb = torch.cat([cur_emb, emb(target_ids[:, t:t+1])], dim=1)
                ptr = min(ptr + 1, max(0, advisor_ids.shape[1] - 1))
        
        ce.backward()
        opt.step()
        if (step+1) % 50 == 0:
            print(f"  Step {step+1} | CE: {ce.item():.4f} | Gate: {torch.sigmoid(projector.gate).item():.4f}")

def _evaluate_cell(
    cell: str,
    projector: RelationalProjector,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    eval_ds: List[Problem],
    args: argparse.Namespace
) -> Dict[str, Any]:
    projector.eval()
    rows = []
    
    for item in eval_ds:
        with torch.no_grad():
            attn, eng_hidden = _extract_attention(model, tokenizer, build_final_prefix(item.prompt))
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, int(args.max_logic_new_tokens))
            z_st, _, _, _ = codebook.quantize(h_t)
            
            if cell != "A":
                z_refined = projector(attn, eng_hidden, z_st)
            else:
                z_refined = z_st
                
            tokens = _decode_logic_tokens(arity_head, z_refined, int(args.relation_vocab), int(args.var_min_id))
            advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
            advisor_ids = torch.stack(tokens, dim=1)
            
            cur_ids = tokenizer(build_final_prefix(item.prompt), return_tensors="pt").input_ids.to(model.device)
            cur_emb = model.get_input_embeddings()(cur_ids)
            generated = []
            ptr = 0
            for _ in range(int(args.max_final_new_tokens)):
                p_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
                with persistent_advisor_hook(model, int(args.layer_index), advisor_adapter, advisor_states, advisor_ids, p_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, return_dict=True)
                next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                if next_id == tokenizer.eos_token_id: break
                cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(torch.tensor([[next_id]], device=model.device))], dim=1)
                ptr = min(ptr + 1, advisor_ids.shape[1] - 1)

        ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
        correct = _answer_match(item.answer, ans)
        rows.append({"correct": bool(correct), "family": _infer_family(item.prompt)})

    acc = sum(1 for r in rows if r["correct"]) / len(rows)
    adj_acc = [r["correct"] for r in rows if "adjective" in r["family"]]
    return {
        "overall_accuracy": acc,
        "adjective_accuracy": sum(adj_acc)/len(adj_acc) if adj_acc else 0.0,
        "gate": torch.sigmoid(projector.gate).item() if cell != "A" else 0.0
    }

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=16)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_13_relational_grid"))
    return p.parse_args()

def main():
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    ts = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    assert_output_path_allowed("M", args.output_root)
    run_dir = args.output_root / ts
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter)
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    tokenizer_source = str(adapter_path) if adapter_has_tokenizer else args.base_model
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        local_files_only=args.local_files_only,
        attn_implementation="eager"
    )
    backbone.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(backbone, str(adapter_path), local_files_only=args.local_files_only)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=model.device)
    h_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    
    codebook = BooleanAnchorTable(2000, h_size).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    arity_head = AdvisorArityHead(h_size, 2000).to(model.device, dtype=model.dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    advisor_adapter = CouncilCrossAttentionAdapter(h_size, use_boolean_surgery=True).to(model.device, dtype=model.dtype)
    advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)

    ds_train = generate_dataset(size=args.eval_size, seed=42, profile="winograd_bench_v1", difficulty_tier="all")
    ds_eval = generate_dataset(size=args.eval_size, seed=43, profile="winograd_bench_v1", difficulty_tier="all")

    grid = {
        "A": ("NONE", "Control"),
        "B": ("SVD", "Proposal 1 (Pure Math)"),
        "C": ("LRAP", "Refined Proposal (Learned Relational)"),
        "D": ("ELEMENTS", "Proposal 3 (Relational Atoms)")
    }
    
    report = {
        "timestamp": ts,
        "series": series_metadata("M", "M3.13", "scripts/run_m3_13_geometric_ablation_grid.py"),
        "track": "M3.13",
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=str(args.checkpoint).replace("\\", "/"),
            checkpoint_out=None,
            dataset_profile="winograd_bench_v1",
            difficulty_tier="all",
        ),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "cells": {},
    }
    
    for cell, (mode, desc) in grid.items():
        print(f"\n=== M3.13 CELL {cell}: {desc} ===")
        proj = RelationalProjector(h_size, 32, mode, num_heads=n_heads).to(model.device, dtype=model.dtype)
        _train_cell(cell, proj, model, tokenizer, codebook, arity_head, advisor_adapter, ds_train, args)
        res = _evaluate_cell(cell, proj, model, tokenizer, codebook, arity_head, advisor_adapter, ds_eval, args)
        report["cells"][cell] = res
        print(f"Results: {res}")

    (run_dir / "m3_13_report.json").write_text(json.dumps(report, indent=2))
    
    md = ["# M3.13 Relational Projection Ablation Grid", "", "| Cell | Mode | Accuracy | Adj Acc | Gate |", "|---|---|---|---|---|"]
    for c, (m, d) in grid.items():
        r = report["cells"][c]
        md.append(f"| {c} | {m} | {r['overall_accuracy']:.3f} | {r['adjective_accuracy']:.3f} | {r['gate']:.3f} |")
    
    (run_dir / "m3_13_report.md").write_text("\n".join(md))
    print(f"\nGrid Complete. Results at {run_dir}")

if __name__ == "__main__":
    main()
