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


class GeometricReturnAdapter(torch.nn.Module):
    """
    M3.12 v2: Relational Return Stream.
    Uses a small set of learned queries to extract structural interactions
    (asymmetry, polarity, causal direction) from English semantics rather
    than flattening them with mean pooling.
    """
    def __init__(self, hidden_size: int, bottleneck_dim: int = 32, num_queries: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = torch.nn.Parameter(torch.empty(1, num_queries, hidden_size))
        torch.nn.init.normal_(self.queries, mean=0.0, std=0.02)
        
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Multi-query interaction bottleneck
        self.down = torch.nn.Linear(hidden_size * num_queries, bottleneck_dim, bias=False)
        self.up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0)) # start mostly closed
        self.cue_scale = torch.nn.Parameter(torch.tensor(0.1))
        
    def forward(self, eng_hidden: torch.Tensor, z_st: torch.Tensor, periodic: bool = False) -> torch.Tensor:
        # eng_hidden: [1, L_eng, H]
        # z_st: [1, L_adv, H]
        b, l_eng, h = eng_hidden.shape
        
        # 1. Relational Attention: Extract structural "slots" from English hidden states.
        # This preserves asymmetry because the attention weights vary per learned query.
        k = self.k_proj(eng_hidden)
        v = self.v_proj(eng_hidden)
        q = self.queries.expand(b, -1, -1)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(h))
        attn = torch.softmax(scores, dim=-1)
        relational_summary = torch.matmul(attn, v) # [b, num_queries, h]
        
        # 2. Geometric Compression: Flatten the slots into a dense interaction cue.
        flat_summary = relational_summary.view(b, 1, -1) # [b, 1, queries * h]
        cue = self.up(F.relu(self.down(flat_summary))) # [b, 1, h]
        cue = torch.tanh(cue) * torch.sigmoid(self.cue_scale)
        
        if not periodic:
            # Cell B: Static relational cue applied globally to the advisor latent sequence
            return z_st + torch.sigmoid(self.gate) * cue
        else:
            # Cell C: Periodic refinement. Modulate cue strength along the trace length
            # to simulate the advisor "looking back" at specific relational slots.
            b_adv, l_adv, _ = z_st.shape
            pos_weights = torch.linspace(0.8, 1.2, l_adv, device=z_st.device).view(1, l_adv, 1)
            return z_st + torch.sigmoid(self.gate) * cue * pos_weights


def _train_return_adapter(
    cell: str,
    adapter_net: GeometricReturnAdapter,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    train_dataset: List[Problem],
    args: argparse.Namespace,
) -> None:
    if cell == "A":
        return # Control needs no training
        
    opt = torch.optim.AdamW(adapter_net.parameters(), lr=args.lr, weight_decay=0.01)
    periodic = (cell == "C")
    
    print(f"\nTraining Relational Return Adapter for Cell {cell}...")
    adapter_net.train()
    
    for step in range(args.train_steps):
        item = train_dataset[step % len(train_dataset)]
        
        with torch.no_grad():
            # 1. Forward English Pass (to get return signal base)
            eng_prefix = build_final_prefix(item.prompt)
            eng_ids = tokenizer(eng_prefix, return_tensors="pt").input_ids.to(model.device)
            with adapter_disabled(model):
                eng_out = model(input_ids=eng_ids, output_hidden_states=True)
            eng_hidden = eng_out.hidden_states[-1].detach()
            
            # 2. Extract Base Advisor Trace
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _, _, _ = codebook.quantize(h_t)
            
        opt.zero_grad()
        
        # 3. Apply Relational Return Cue
        z_refined = adapter_net(eng_hidden, z_st, periodic=periodic)
        
        # 4. Differentiable Arity Decode (Gumbel-Softmax for gradient flow)
        b, l, _ = z_refined.shape
        refined_tokens = []
        refined_ids: list[torch.Tensor] = []
        for i in range(l):
            z = z_refined[:, i, :]
            l_rel_raw = arity_head.head_rel(z)
            l_v1_raw = arity_head.head_var1(z)
            l_v2_raw = arity_head.head_var2(z)

            # Keep train decode contract aligned with eval decode contract.
            l_rel = _masked_logits(l_rel_raw, 0, int(args.relation_vocab))
            l_v1 = _masked_logits(l_v1_raw, int(args.var_min_id), l_v1_raw.shape[-1])
            l_v2 = _masked_logits(l_v2_raw, int(args.var_min_id), l_v2_raw.shape[-1])
            
            # Straight-through Gumbel-Softmax
            p_rel = F.gumbel_softmax(l_rel, tau=1.0, hard=True)
            p_v1 = F.gumbel_softmax(l_v1, tau=1.0, hard=True)
            p_v2 = F.gumbel_softmax(l_v2, tau=1.0, hard=True)
            
            v_rel = p_rel @ codebook.emb
            v_v1 = p_v1 @ codebook.emb
            v_v2 = p_v2 @ codebook.emb
            refined_tokens.extend([v_rel.unsqueeze(1), v_v1.unsqueeze(1), v_v2.unsqueeze(1)])
            refined_ids.extend(
                [
                    torch.argmax(p_rel, dim=-1),
                    torch.argmax(p_v1, dim=-1),
                    torch.argmax(p_v2, dim=-1),
                ]
            )
            
        advisor_states = torch.cat(refined_tokens, dim=1) # [1, L*3, H]
        advisor_ids = torch.stack(refined_ids, dim=1).to(dtype=torch.long)  # Real ids for pointer routing

        # 5. Compute CE Loss
        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :6]
        if int(target_ids.numel()) == 0:
            continue
            
        with adapter_disabled(model):
            emb_layer = model.get_input_embeddings()
            cur_emb = emb_layer(eng_ids)
            ce = torch.tensor(0.0, device=model.device, dtype=model.dtype)
            ptr = 0
            for t in range(target_ids.shape[1]):
                am = torch.ones((1, cur_emb.shape[1]), dtype=torch.long, device=model.device)
                ptr_ids = torch.full((1, cur_emb.shape[1]), int(ptr), dtype=torch.long, device=model.device)
                with persistent_advisor_hook(model, int(args.layer_index), advisor_adapter, advisor_states, advisor_ids, ptr_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
                logits = out.logits[:, -1, :]
                ce = ce + F.cross_entropy(logits, target_ids[:, t])
                cur_emb = torch.cat([cur_emb, emb_layer(target_ids[:, t : t + 1])], dim=1)
                ptr = min(ptr + 1, max(0, advisor_ids.shape[1] - 1))
                
        # Anti-Leakage Norm Penalty (keep cue geometric, not lexical)
        cue_norm = torch.norm(adapter_net.gate * adapter_net.up.weight)
        loss = ce + (0.05 * cue_norm)
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1} | Loss: {loss.item():.4f} | Gate: {torch.sigmoid(adapter_net.gate).item():.4f}")


def _evaluate_cell(
    cell: str,
    adapter_net: GeometricReturnAdapter,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    eval_dataset: List[Problem],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    
    adapter_net.eval()
    periodic = (cell == "C")
    rows = []
    
    cue_norms = []
    gate_strengths = []
    
    for item in eval_dataset:
        with torch.no_grad():
            # 1. Base generation 
            eng_prefix = build_final_prefix(item.prompt)
            eng_ids = tokenizer(eng_prefix, return_tensors="pt").input_ids.to(model.device)
            if cell in ("B", "C"):
                with adapter_disabled(model):
                    eng_out = model(input_ids=eng_ids, output_hidden_states=True)
                eng_hidden = eng_out.hidden_states[-1]
            
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _, _, _ = codebook.quantize(h_t)
            
            # 2. Refine with Relational Return Stream
            if cell in ("B", "C"):
                z_refined = adapter_net(eng_hidden, z_st, periodic=periodic)
                gate_strengths.append(torch.sigmoid(adapter_net.gate).item())
                # Re-calculate cue for norm stats
                b, l_eng, h = eng_hidden.shape
                k = adapter_net.k_proj(eng_hidden)
                v = adapter_net.v_proj(eng_hidden)
                q = adapter_net.queries.expand(b, -1, -1)
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(h))
                attn = torch.softmax(scores, dim=-1)
                rel_sum = torch.matmul(attn, v)
                cue = adapter_net.up(F.relu(adapter_net.down(rel_sum.view(b, 1, -1))))
                cue_norms.append(torch.norm(cue).item())
            else:
                z_refined = z_st
                
            # 3. Decode Tokens
            tokens = _decode_logic_tokens(arity_head, z_refined, relation_vocab=5, var_min_id=5)
            advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
            advisor_ids = torch.stack(tokens, dim=1)
            
            # 4. Final Answer Generation
            cur_emb = model.get_input_embeddings()(eng_ids)
            generated: list[int] = []
            pointer = 0
            for _ in range(int(args.max_final_new_tokens)):
                p_ids = torch.full((1, cur_emb.shape[1]), pointer, device=model.device, dtype=torch.long)
                with persistent_advisor_hook(model, int(args.layer_index), advisor_adapter, advisor_states, advisor_ids, p_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, return_dict=True)
                logits = out.logits[:, -1, :]
                next_id = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_id)
                if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                    break
                next_emb = model.get_input_embeddings()(torch.tensor([[next_id]], device=model.device))
                cur_emb = torch.cat([cur_emb, next_emb], dim=1)
                pointer = min(pointer + 1, max(0, advisor_ids.shape[1] - 1))

        model_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        token_ids = [int(t[0].detach().item()) for t in tokens]
        rel_ids = token_ids[0::3]
        rel_counts = Counter(rel_ids)
        rel_total = max(1, len(rel_ids))
        probs = [float(c) / float(rel_total) for c in rel_counts.values()]
        op_entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
        op_top1 = float(max(probs)) if probs else 1.0
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        correct = _answer_match(item.answer, model_answer)
        
        row = {
            "problem_id": int(item.problem_id),
            "prompt": item.prompt,
            "family": _infer_family(item.prompt),
            "gold_answer": item.answer,
            "model_answer": model_answer,
            "correct": bool(correct),
            "active_token_count": int(len(set(token_ids))),
            "operator_entropy": float(op_entropy),
            "operator_top1_share": float(op_top1),
            "scope": float(scope.get("scope_total", 1.0)),
            "advisor_trace_state": "articulated" if (len(set(token_ids)) >= 20 and op_entropy >= 0.35) else "collapsed",
        }
        row["failure_taxonomy"] = _taxonomize_failure(row)
        rows.append(row)
        
    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    
    # Specific metric subsets
    adjective_flip_rows = [r for r in rows if "adjective_property" in r["family"]]
    causal_rows = [r for r in rows if "causal_direction" in r["family"]]
    
    metrics = {
        "overall_accuracy": acc,
        "adjective_flip_accuracy": sum(1 for r in adjective_flip_rows if r["correct"]) / max(1, len(adjective_flip_rows)) if adjective_flip_rows else 0.0,
        "causal_connective_accuracy": sum(1 for r in causal_rows if r["correct"]) / max(1, len(causal_rows)) if causal_rows else 0.0,
        "mean_active_tokens": sum(r["active_token_count"] for r in rows) / max(1, len(rows)),
        "mean_operator_entropy": sum(r["operator_entropy"] for r in rows) / max(1, len(rows)),
        "mean_scope_violation": sum(r["scope"] for r in rows) / max(1, len(rows)),
        "return_signal_stats": {
            "mean_gate_magnitude": sum(gate_strengths) / max(1, len(gate_strengths)) if gate_strengths else 0.0,
            "mean_cue_norm": sum(cue_norms) / max(1, len(cue_norms)) if cue_norms else 0.0,
        }
    }
    
    return {"metrics": metrics, "rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.12 v2: Geometric Relational Return Stream Refinement")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=16)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("runs/m3_12_geometric_return_v2"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapter_has_tokenizer = (Path(args.adapter) / "tokenizer.json").exists() or (Path(args.adapter) / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.checkpoint, map_location=model.device)
    hidden_size = int(model.config.hidden_size)
    
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    codebook.eval()
    
    adapter_mod = CouncilCrossAttentionAdapter(hidden_size, use_boolean_surgery=True).to(model.device, dtype=model.dtype)
    adapter_mod.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    adapter_mod.eval()
    
    arity_head = AdvisorArityHead(hidden_size=hidden_size, codebook_size=2000).to(model.device, dtype=model.dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    arity_head.eval()

    # Data
    ds_train = generate_dataset(size=int(args.eval_size), seed=args.seed, profile="winograd_bench_v1", difficulty_tier="all")
    ds_eval = generate_dataset(size=int(args.eval_size), seed=args.seed + 1, profile="winograd_bench_v1", difficulty_tier="all")

    cells = ["A", "B", "C"]
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) for k, v in vars(args).items()},
        "cells": {}
    }
    
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for cell in cells:
        print(f"\n=== Executing M3.12 Cell {cell} ===")
        return_adapter = GeometricReturnAdapter(hidden_size, bottleneck_dim=args.bottleneck_dim, num_queries=args.num_queries).to(model.device, dtype=model.dtype)
        
        _train_return_adapter(cell, return_adapter, model, tokenizer, codebook, arity_head, adapter_mod, ds_train, args)
        
        cell_results = _evaluate_cell(cell, return_adapter, model, tokenizer, codebook, arity_head, adapter_mod, ds_eval, args)
        report_data["cells"][cell] = cell_results["metrics"]
        
        # Write specific artifact for cell
        (run_dir / f"m3_12_{cell}_winograd_eval.json").write_text(json.dumps(cell_results["rows"], indent=2), encoding="utf-8")
        
        print(f"Cell {cell} Evaluation:")
        for k, v in cell_results["metrics"].items():
            print(f"  {k}: {v}")

    # Final Report
    report_json = run_dir / "m3_12_return_stream_report.json"
    report_json.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    
    md = [
        "# M3.12 v2: Relational Geometric Return Stream Refinement",
        "",
        "| Cell | Accuracy | Adjective Flip Acc | Causal Acc | Mean Gate | Mean Scope |",
        "|---|---|---|---|---|---|",
    ]
    for cell in cells:
        m = report_data["cells"][cell]
        gate = m.get("return_signal_stats", {}).get("mean_gate_magnitude", 0.0)
        md.append(f"| {cell} | {m['overall_accuracy']:.3f} | {m['adjective_flip_accuracy']:.3f} | {m['causal_connective_accuracy']:.3f} | {gate:.3f} | {m['mean_scope_violation']:.3f} |")
        
    report_md = run_dir / "m3_12_return_stream_report.md"
    report_md.write_text("\n".join(md), encoding="utf-8")
    
    print(f"\nExperiment complete. Reports written to {run_dir}")


if __name__ == "__main__":
    main()
