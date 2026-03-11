from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import Problem, generate_dataset
from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    adapter_disabled,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", str(text).strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and (p == e or p.startswith(e))


def _infer_family(prompt: str) -> str:
    p = prompt.lower()
    if "too big" in p or "too small" in p or "too heavy" in p or "too weak" in p:
        return "adjective_property"
    if "because" in p or "since" in p:
        return "causal_direction"
    return "other"


def _triples(token_ids: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for i in range(0, len(token_ids) - 2, 3):
        out.append((int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2])))
    return out


def _masked_logits(logits: torch.Tensor, legal_start: int, legal_end: int) -> torch.Tensor:
    out = logits.clone()
    if legal_start > 0:
        out[:, :legal_start] = -1e9
    if legal_end < out.shape[-1]:
        out[:, legal_end:] = -1e9
    return out


def _decode_logic_tokens(
    arity_head: AdvisorArityHead,
    z_st: torch.Tensor,
    relation_vocab: int,
    var_min_id: int,
) -> list[torch.Tensor]:
    tokens: list[torch.Tensor] = []
    for i in range(z_st.shape[1]):
        z = z_st[:, i, :]
        l_rel = _masked_logits(arity_head.head_rel(z), 0, int(relation_vocab))
        l_v1_raw = arity_head.head_var1(z)
        l_v1 = _masked_logits(l_v1_raw, int(var_min_id), l_v1_raw.shape[-1])
        l_v2_raw = arity_head.head_var2(z)
        l_v2 = _masked_logits(l_v2_raw, int(var_min_id), l_v2_raw.shape[-1])
        t_rel = torch.argmax(l_rel, dim=-1)
        t_v1 = torch.argmax(l_v1, dim=-1)
        t_v2 = torch.argmax(l_v2, dim=-1)
        tokens.extend([t_rel, t_v1, t_v2])
    return tokens


def _pairwise_dist_vec(nodes: torch.Tensor) -> torch.Tensor:
    # nodes: [1, N, H]
    n = int(nodes.shape[1])
    if n < 2:
        return torch.zeros((1,), device=nodes.device, dtype=nodes.dtype)
    x = F.normalize(nodes[0], dim=-1)
    d = torch.cdist(x, x, p=2.0)
    iu = torch.triu_indices(n, n, offset=1, device=d.device)
    v = d[iu[0], iu[1]]
    return torch.sort(v)[0]


class StructuralAlignmentBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.cue_down = torch.nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.cue_up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.cue_scale = torch.nn.Parameter(torch.tensor(0.1))

    def project_english_nodes(self, eng_hidden: torch.Tensor, max_nodes: int) -> torch.Tensor:
        # Select structurally salient tokens by activation norm.
        scores = torch.norm(eng_hidden, dim=-1)  # [1, L]
        k = max(2, min(int(max_nodes), int(eng_hidden.shape[1])))
        top_idx = torch.topk(scores[0], k=k, largest=True).indices
        top_idx_sorted, _ = torch.sort(top_idx)
        nodes = eng_hidden[:, top_idx_sorted, :]
        return self.projector(nodes)

    def alignment_objective(
        self,
        eng_hidden: torch.Tensor,
        advisor_relation_states: torch.Tensor,
        max_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_nodes = self.project_english_nodes(eng_hidden, max_nodes=max_nodes)
        k = min(int(e_nodes.shape[1]), int(advisor_relation_states.shape[1]))
        if k < 2:
            z = torch.zeros((), device=eng_hidden.device, dtype=eng_hidden.dtype)
            return z, z, e_nodes
        e_use = e_nodes[:, :k, :]
        a_use = advisor_relation_states[:, :k, :].detach()
        ve = _pairwise_dist_vec(e_use)
        va = _pairwise_dist_vec(a_use)
        m = min(int(ve.shape[0]), int(va.shape[0]))
        if m <= 0:
            z = torch.zeros((), device=eng_hidden.device, dtype=eng_hidden.dtype)
            return z, z, e_use
        ve_m = ve[:m]
        va_m = va[:m]
        loss = F.mse_loss(ve_m, va_m)
        sim = F.cosine_similarity(ve_m.unsqueeze(0), va_m.unsqueeze(0), dim=-1)[0]
        return loss, sim, e_use

    def runtime_cue(self, eng_hidden: torch.Tensor, max_nodes: int) -> torch.Tensor:
        e_nodes = self.project_english_nodes(eng_hidden, max_nodes=max_nodes)
        pooled = torch.mean(e_nodes, dim=1, keepdim=True)
        cue = self.cue_up(F.relu(self.cue_down(pooled)))
        cue = torch.tanh(cue) * torch.sigmoid(self.cue_scale)
        return cue


def _extract_english_hidden(model, tokenizer, prompt: str, layer_index: int) -> torch.Tensor:
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(model.device)
    with adapter_disabled(model):
        out = model(input_ids=p_ids, output_hidden_states=True, use_cache=False)
    idx = max(0, min(int(layer_index), len(out.hidden_states) - 1))
    return out.hidden_states[idx].detach()


def _train_alignment_bridge(
    cell: str,
    bridge: StructuralAlignmentBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    train_ds: list[Problem],
    args: argparse.Namespace,
) -> dict[str, float]:
    if cell == "A":
        return {"mean_alignment_loss": 0.0, "mean_alignment_similarity": 0.0}

    bridge.train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=float(args.lr), weight_decay=0.01)
    losses: list[float] = []
    sims: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_ds[step % len(train_ds)]
        with torch.no_grad():
            eng_hidden = _extract_english_hidden(model, tokenizer, item.prompt, layer_index=int(args.layer_index))
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)
            tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
            relation_states = advisor_states[:, 0::3, :]

        opt.zero_grad()
        align_loss, align_sim, e_nodes = bridge.alignment_objective(
            eng_hidden=eng_hidden,
            advisor_relation_states=relation_states,
            max_nodes=int(args.max_nodes),
        )
        reg = 0.01 * torch.mean(e_nodes * e_nodes)
        loss = align_loss + reg
        loss.backward()
        opt.step()
        losses.append(float(align_loss.detach().item()))
        sims.append(float(align_sim.detach().item()))

    return {
        "mean_alignment_loss": float(sum(losses) / max(1, len(losses))),
        "mean_alignment_similarity": float(sum(sims) / max(1, len(sims))),
    }


def _evaluate_cell(
    cell: str,
    bridge: StructuralAlignmentBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    eval_ds: list[Problem],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge.eval()
    rows: list[dict[str, Any]] = []
    align_losses: list[float] = []
    align_sims: list[float] = []
    cue_norms: list[float] = []

    for item in eval_ds:
        with torch.no_grad():
            eng_hidden = _extract_english_hidden(model, tokenizer, item.prompt, layer_index=int(args.layer_index))
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)

            # Build baseline decoded tokens for relation-state alignment.
            baseline_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            baseline_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in baseline_tokens], dim=1)
            baseline_rel_states = baseline_states[:, 0::3, :]
            al, sim, _ = bridge.alignment_objective(eng_hidden, baseline_rel_states, max_nodes=int(args.max_nodes))
            align_losses.append(float(al.detach().item()))
            align_sims.append(float(sim.detach().item()))

            if cell == "C":
                cue = bridge.runtime_cue(eng_hidden, max_nodes=int(args.max_nodes))
                cue_norms.append(float(torch.norm(cue).item()))
                z_use = z_st + torch.sigmoid(bridge.gate) * cue
            else:
                z_use = z_st

            tokens = _decode_logic_tokens(arity_head, z_use, int(args.relation_vocab), int(args.var_min_id))
            advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
            advisor_ids = torch.stack(tokens, dim=1)

            p_ids = tokenizer(build_final_prefix(item.prompt), return_tensors="pt").input_ids.to(model.device)
            cur_emb = model.get_input_embeddings()(p_ids)
            generated: list[int] = []
            ptr = 0
            for _ in range(int(args.max_final_new_tokens)):
                ptr_ids = torch.full((1, cur_emb.shape[1]), int(ptr), device=model.device, dtype=torch.long)
                with persistent_advisor_hook(model, int(args.layer_index), advisor_adapter, advisor_states, advisor_ids, ptr_ids, 1.0):
                    out = model(inputs_embeds=cur_emb, return_dict=True)
                next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                    break
                next_emb = model.get_input_embeddings()(torch.tensor([[next_id]], device=model.device))
                cur_emb = torch.cat([cur_emb, next_emb], dim=1)
                ptr = min(ptr + 1, max(0, advisor_ids.shape[1] - 1))

        pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
        token_ids = [int(t[0].detach().item()) for t in tokens]
        rel_ids = token_ids[0::3]
        counts = Counter(rel_ids)
        total = max(1, len(rel_ids))
        probs = [float(c) / float(total) for c in counts.values()]
        entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        correct = _answer_match(item.answer, pred)
        rows.append(
            {
                "problem_id": int(item.problem_id),
                "prompt": item.prompt,
                "family": _infer_family(item.prompt),
                "gold_answer": item.answer,
                "model_answer": pred,
                "correct": bool(correct),
                "active_token_count": int(len(set(token_ids))),
                "operator_entropy": float(entropy),
                "scope": float(scope.get("scope_total", 1.0)),
                "alignment_loss": float(align_losses[-1]),
                "alignment_similarity": float(align_sims[-1]),
            }
        )

    n = max(1, len(rows))
    adj = [r for r in rows if r["family"] == "adjective_property"]
    cau = [r for r in rows if r["family"] == "causal_direction"]
    metrics = {
        "overall_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
        "adjective_accuracy": float(sum(1 for r in adj if r["correct"]) / max(1, len(adj))),
        "causal_accuracy": float(sum(1 for r in cau if r["correct"]) / max(1, len(cau))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_scope": float(sum(float(r["scope"]) for r in rows) / n),
        "mean_alignment_loss": float(sum(align_losses) / max(1, len(align_losses))),
        "mean_alignment_similarity": float(sum(align_sims) / max(1, len(align_sims))),
        "mean_gate": float(torch.sigmoid(bridge.gate).item()) if cell != "A" else 0.0,
        "mean_cue_norm": float(sum(cue_norms) / max(1, len(cue_norms))) if cue_norms else 0.0,
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.14 Structural Alignment Bridge (A/B/C).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=16)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(args.seed))

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter)
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    tokenizer_source = str(adapter_path) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(adapter_path), local_files_only=args.local_files_only)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=model.device)
    h = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, h).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    arity_head = AdvisorArityHead(h, 2000).to(model.device, dtype=model.dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    advisor_adapter = CouncilCrossAttentionAdapter(h, use_boolean_surgery=True).to(model.device, dtype=model.dtype)
    advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)

    ds_train = generate_dataset(size=int(args.eval_size), seed=int(args.seed), profile="winograd_bench_v1", difficulty_tier="all")
    ds_eval = generate_dataset(size=int(args.eval_size), seed=int(args.seed) + 1, profile="winograd_bench_v1", difficulty_tier="all")

    cells = {
        "A": "Control",
        "B": "Structural alignment objective only",
        "C": "Alignment-trained runtime cue",
    }
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": "M3.14",
        "config": {k: str(v) for k, v in vars(args).items()},
        "cells": {},
    }

    for cell, desc in cells.items():
        print(f"\n=== M3.14 CELL {cell}: {desc} ===")
        bridge = StructuralAlignmentBridge(hidden_size=h, bottleneck_dim=64).to(model.device, dtype=model.dtype)
        train_stats = _train_alignment_bridge(
            cell=cell,
            bridge=bridge,
            model=model,
            tokenizer=tokenizer,
            codebook=codebook,
            arity_head=arity_head,
            train_ds=ds_train,
            args=args,
        )
        metrics, rows = _evaluate_cell(
            cell=cell,
            bridge=bridge,
            model=model,
            tokenizer=tokenizer,
            codebook=codebook,
            arity_head=arity_head,
            advisor_adapter=advisor_adapter,
            eval_ds=ds_eval,
            args=args,
        )
        metrics["train_alignment_loss"] = float(train_stats["mean_alignment_loss"])
        metrics["train_alignment_similarity"] = float(train_stats["mean_alignment_similarity"])
        report["cells"][cell] = metrics
        (run_dir / f"m3_14_{cell}_winograd_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Results: {metrics}")

    (run_dir / "m3_14_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# M3.14 Structural Alignment Bridge",
        "",
        "| Cell | Accuracy | Adj Acc | Causal Acc | Align Loss | Align Sim | Gate | Cue Norm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ("A", "B", "C"):
        m = report["cells"][cell]
        md.append(
            f"| {cell} | {m['overall_accuracy']:.3f} | {m['adjective_accuracy']:.3f} | {m['causal_accuracy']:.3f} | "
            f"{m['mean_alignment_loss']:.4f} | {m['mean_alignment_similarity']:.4f} | {m['mean_gate']:.3f} | {m['mean_cue_norm']:.3f} |"
        )
    (run_dir / "m3_14_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nM3.14 complete. Results at {run_dir}")


if __name__ == "__main__":
    main()

