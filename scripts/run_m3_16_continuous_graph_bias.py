from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_15d_answer_path_forcing import (  # type: ignore
    CUE_TERMS,
    _answer_match,
    _build_winograd_pack,
    _build_winograd_split_packs,
    _decode_logic_tokens,
    _find_char_span,
    _load_pack,
    _model_device,
    _score_candidate_first_token_with_adapter_tensor,
    _triples,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    adapter_disabled,
    build_final_prefix,
    extract_trace_hidden_states,
)


def _extract_prompt_hidden_with_offsets(model, tokenizer, prompt: str, layer_index: int) -> tuple[str, torch.Tensor, list[tuple[int, int]], torch.Tensor]:
    dev = _model_device(model)
    prefix = build_final_prefix(prompt)
    try:
        enc = tokenizer(prefix, return_tensors="pt", return_offsets_mapping=True)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    except NotImplementedError:
        enc = tokenizer(prefix, return_tensors="pt")
        offsets = [(0, 0) for _ in range(int(enc["input_ids"].shape[1]))]
    p_ids = enc["input_ids"].to(dev)
    with adapter_disabled(model):
        out = model(input_ids=p_ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[int(layer_index)].detach()
    return prefix, p_ids, offsets, hidden


def _mask_from_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int] | None], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, len(offsets)), dtype=torch.bool, device=device)
    for span in spans:
        if span is None:
            continue
        start, end = int(span[0]), int(span[1])
        for i, (s, e) in enumerate(offsets):
            if e > s and not (e <= start or s >= end):
                mask[0, i] = True
    return mask


def _build_graph_context(prefix: str, offsets: list[tuple[int, int]], hidden: torch.Tensor, candidates: list[str]) -> dict[str, Any]:
    dev = hidden.device
    valid_mask = torch.tensor([[bool(e > s) for s, e in offsets]], dtype=torch.bool, device=dev)
    if int(valid_mask.sum().item()) <= 0:
        valid_mask = torch.ones((1, len(offsets)), dtype=torch.bool, device=dev)

    candidate_spans: list[tuple[int, int] | None] = []
    for candidate in list(candidates)[:2]:
        candidate_spans.append(_find_char_span(prefix, str(candidate)))
    candidate_mask = _mask_from_spans(offsets, candidate_spans, dev) & valid_mask

    cue_span = None
    cue_term = ""
    for term in CUE_TERMS:
        span = _find_char_span(prefix, term)
        if span is not None:
            cue_span = span
            cue_term = term
            break
    cue_mask = _mask_from_spans(offsets, [cue_span], dev) & valid_mask

    return {
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "cue_mask": cue_mask,
        "candidate_spans": [list(s) if s is not None else None for s in candidate_spans],
        "cue_span": list(cue_span) if cue_span is not None else None,
        "cue_term": cue_term,
    }


def _build_base_advisor_state(
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    prompt: str,
    relation_vocab: int,
    var_min_id: int,
    max_logic_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(max_logic_new_tokens))
    z_st, _idx, _cb, _commit = codebook.quantize(h_t)
    base_tokens = _decode_logic_tokens(arity_head, z_st, int(relation_vocab), int(var_min_id))
    base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
    base_ids = torch.stack(base_tokens, dim=1)
    rel_nodes = base_states[:, 0::3, :]
    token_ids = [int(t[0].detach().item()) for t in base_tokens]
    return base_states, base_ids, rel_nodes, token_ids


class SoftGraphBiasCouncilAdapter(CouncilCrossAttentionAdapter):
    def __init__(self, hidden_size: int, use_boolean_surgery: bool = True) -> None:
        super().__init__(hidden_size, use_boolean_surgery=use_boolean_surgery)
        self.token_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.rel_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.ctx_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.rel_slot_head = torch.nn.Linear(hidden_size, 3, bias=False)
        self.bias_gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.bias_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.current_graph_bias: torch.Tensor | None = None
        self.current_stats: dict[str, float] = {}

    def freeze_base(self) -> None:
        base_names = {
            "q_judge.weight",
            "k_judge.weight",
            "v_judge.weight",
            "q_intuitor.weight",
            "k_intuitor.weight",
            "v_intuitor.weight",
            "out_proj.weight",
            "gain",
            "council_weights",
        }
        for name, param in self.named_parameters():
            param.requires_grad = name not in base_names

    def configure_graph_bias(self, prompt_hidden: torch.Tensor, context: dict[str, Any], rel_nodes: torch.Tensor, *, mode: str) -> None:
        valid_mask = context["valid_mask"]
        candidate_mask = context["candidate_mask"]
        cue_mask = context["cue_mask"]
        token_keys = self.token_key(prompt_hidden)
        rel_summary = rel_nodes.mean(dim=1)
        query = self.query_proj(rel_summary).unsqueeze(-1)
        scores = torch.matmul(token_keys, query).squeeze(-1) / math.sqrt(float(token_keys.shape[-1]))
        if mode == "candidate":
            allow = candidate_mask.clone()
        elif mode == "candidate_cue":
            allow = candidate_mask | cue_mask
        elif mode == "global":
            allow = valid_mask.clone()
        else:
            raise ValueError(f"Unsupported graph-bias mode {mode!r}")
        if int(allow.sum().item()) <= 0:
            allow = valid_mask.clone()
        masked_scores = scores.masked_fill(~allow, -1e9)
        token_weights = torch.softmax(masked_scores, dim=-1)
        ctx = torch.einsum("bt,bth->bh", token_weights, prompt_hidden)
        fused = torch.tanh(self.rel_proj(rel_summary) + self.ctx_proj(ctx))
        slot_bias = self.rel_slot_head(fused)
        gate = torch.sigmoid(self.bias_gate)
        bias = gate * self.bias_scale * token_weights.unsqueeze(-1) * slot_bias.unsqueeze(1)
        self.current_graph_bias = bias.to(dtype=prompt_hidden.dtype)
        self.current_stats = {
            "bias_norm": float(torch.mean(self.current_graph_bias * self.current_graph_bias).detach().item()),
            "bias_gate": float(gate.detach().item()),
            "candidate_mass": float((token_weights * candidate_mask.float()).sum().detach().item()),
            "cue_mass": float((token_weights * cue_mask.float()).sum().detach().item()),
        }

    def clear_graph_bias(self) -> None:
        self.current_graph_bias = None
        self.current_stats = {"bias_norm": 0.0, "bias_gate": 0.0, "candidate_mass": 0.0, "cue_mass": 0.0}

    def forward(self, hidden_states: torch.Tensor, advisor_states: torch.Tensor, advisor_ids: torch.Tensor, pointer_ids: torch.Tensor) -> torch.Tensor:
        b, t, h = hidden_states.shape
        l = advisor_states.shape[1]
        p = pointer_ids.unsqueeze(-1)
        indices = torch.cat([p, p + 1, p + 2], dim=-1).clamp(0, l - 1)
        batch_idx = torch.arange(b, device=hidden_states.device).view(b, 1, 1).expand(b, t, 3)
        win_states = advisor_states[batch_idx, indices, :]
        win_ids = advisor_ids[batch_idx, indices]
        qj = self.q_judge(hidden_states).unsqueeze(2)
        kj = self.k_judge(win_states)
        vj = self.v_judge(win_states)
        sj = torch.matmul(qj, kj.transpose(-1, -2)).squeeze(2) / math.sqrt(float(h))
        if self.use_boolean_surgery:
            l_rel, l_v1, l_v2 = sj[:, :, 0], sj[:, :, 1], sj[:, :, 2]
            rel_ids = win_ids[:, :, 0]
            is_and, is_or, is_not, is_implies, is_xor = [(rel_ids == i).float() for i in range(5)]
            l_new = is_and * (l_v1 + l_v2)
            l_new = l_new + is_or * torch.max(l_v1, l_v2)
            l_new = l_new + is_not * (-l_v1)
            l_new = l_new + is_implies * torch.max(-l_v1, l_v2)
            l_new = l_new + is_xor * torch.abs(l_v1 - l_v2)
            mask_any = (is_and + is_or + is_not + is_implies + is_xor).clamp(0, 1)
            l_final = mask_any * l_new + (1.0 - mask_any) * l_rel
            sj = torch.stack([l_final, l_v1, l_v2], dim=-1)
        if self.current_graph_bias is not None:
            gb = self.current_graph_bias.to(device=hidden_states.device, dtype=sj.dtype)
            if int(gb.shape[1]) < t:
                gb = F.pad(gb, (0, 0, 0, t - int(gb.shape[1])))
            elif int(gb.shape[1]) > t:
                gb = gb[:, :t, :]
            sj = sj + gb
        aj = torch.softmax(sj, dim=-1)
        cj = torch.matmul(aj.unsqueeze(2), vj).squeeze(2)
        qi = self.q_intuitor(hidden_states).unsqueeze(2)
        ki = self.k_intuitor(win_states)
        vi = self.v_intuitor(win_states)
        si = torch.matmul(qi, ki.transpose(-1, -2)).squeeze(2) / math.sqrt(float(h))
        ai = torch.softmax(si, dim=-1)
        ci = torch.matmul(ai.unsqueeze(2), vi).squeeze(2)
        w = torch.softmax(self.council_weights, dim=0)
        context = w[0] * cj + w[1] * ci
        return self.out_proj(context) * self.gain


def _train_cell(cell: str, bias_adapter: SoftGraphBiasCouncilAdapter, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, train_pack: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float]:
    if cell == "A":
        return {"answer_path_loss": 0.0, "answer_delta": 0.0, "bias_norm": 0.0, "bias_gate": 0.0, "candidate_mass": 0.0, "cue_mass": 0.0}
    mode_map = {"B": "candidate", "C": "candidate_cue", "D": "global"}
    trainable = [p for p in bias_adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.01)
    loss_hist, delta_hist, bias_norm_hist, gate_hist, candidate_hist, cue_hist = [], [], [], [], [], []
    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        with torch.no_grad():
            prefix, _p_ids, offsets, prompt_hidden = _extract_prompt_hidden_with_offsets(model, tokenizer, prompt, int(args.layer_index))
            context = _build_graph_context(prefix, offsets, prompt_hidden, candidates)
            base_states, base_ids, rel_nodes, _ = _build_base_advisor_state(model, tokenizer, codebook, arity_head, prompt, int(args.relation_vocab), int(args.var_min_id), int(args.max_logic_new_tokens))
        opt.zero_grad()
        bias_adapter.configure_graph_bias(prompt_hidden, context, rel_nodes, mode=mode_map[cell])
        logp_gold = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, gold, bias_adapter, base_states, base_ids, int(args.layer_index))
        logp_foil = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, foil, bias_adapter, base_states, base_ids, int(args.layer_index))
        answer_delta = logp_gold - logp_foil
        answer_loss = torch.relu(torch.tensor(float(args.margin), device=answer_delta.device, dtype=answer_delta.dtype) - answer_delta)
        bias_norm = torch.mean(bias_adapter.current_graph_bias * bias_adapter.current_graph_bias)
        loss = float(args.answer_weight) * answer_loss + float(args.bias_norm_weight) * bias_norm
        loss.backward()
        opt.step()
        stats = dict(bias_adapter.current_stats)
        loss_hist.append(float(answer_loss.detach().item()))
        delta_hist.append(float(answer_delta.detach().item()))
        bias_norm_hist.append(float(stats.get("bias_norm", 0.0)))
        gate_hist.append(float(stats.get("bias_gate", 0.0)))
        candidate_hist.append(float(stats.get("candidate_mass", 0.0)))
        cue_hist.append(float(stats.get("cue_mass", 0.0)))
        bias_adapter.clear_graph_bias()
    return {
        "answer_path_loss": float(sum(loss_hist) / max(1, len(loss_hist))),
        "answer_delta": float(sum(delta_hist) / max(1, len(delta_hist))),
        "bias_norm": float(sum(bias_norm_hist) / max(1, len(bias_norm_hist))),
        "bias_gate": float(sum(gate_hist) / max(1, len(gate_hist))),
        "candidate_mass": float(sum(candidate_hist) / max(1, len(candidate_hist))),
        "cue_mass": float(sum(cue_hist) / max(1, len(cue_hist))),
    }


def _evaluate_cell(cell: str, bias_adapter: SoftGraphBiasCouncilAdapter, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, base_adapter: CouncilCrossAttentionAdapter, eval_pack: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mode_map = {"B": "candidate", "C": "candidate_cue", "D": "global"}
    rows: list[dict[str, Any]] = []
    deltas, answer_delta_hist, bias_norm_hist, bias_gate_hist, candidate_hist, cue_hist = [], [], [], [], [], []
    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        with torch.no_grad():
            prefix, _p_ids, offsets, prompt_hidden = _extract_prompt_hidden_with_offsets(model, tokenizer, prompt, int(args.layer_index))
            context = _build_graph_context(prefix, offsets, prompt_hidden, candidates)
            base_states, base_ids, rel_nodes, token_ids = _build_base_advisor_state(model, tokenizer, codebook, arity_head, prompt, int(args.relation_vocab), int(args.var_min_id), int(args.max_logic_new_tokens))
            logp_off_gold = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, gold, base_adapter, base_states, base_ids, int(args.layer_index))
            logp_off_foil = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, foil, base_adapter, base_states, base_ids, int(args.layer_index))
            if cell == "A":
                logp_on_gold = logp_off_gold
                logp_on_foil = logp_off_foil
                stats = {"bias_norm": 0.0, "bias_gate": 0.0, "candidate_mass": 0.0, "cue_mass": 0.0}
            else:
                bias_adapter.configure_graph_bias(prompt_hidden, context, rel_nodes, mode=mode_map[cell])
                logp_on_gold = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, gold, bias_adapter, base_states, base_ids, int(args.layer_index))
                logp_on_foil = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, foil, bias_adapter, base_states, base_ids, int(args.layer_index))
                stats = dict(bias_adapter.current_stats)
                bias_adapter.clear_graph_bias()
        pred_idx = gi if float(logp_on_gold.item()) >= float(logp_on_foil.item()) else fi
        pred = str(candidates[pred_idx])
        correct = _answer_match(gold, pred)
        rel_ids = token_ids[0::3]
        counts = Counter(rel_ids)
        total = max(1, len(rel_ids))
        probs = [float(c) / float(total) for c in counts.values()]
        entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
        active_op_count = int(sum(1 for _, c in counts.items() if int(c) >= 1))
        top1_op_share = float(max(counts.values())) / float(total) if counts else 1.0
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        answer_delta = float((logp_on_gold - logp_on_foil).item())
        intervention = float((logp_on_gold - logp_off_gold).item())
        answer_delta_hist.append(answer_delta)
        deltas.append(intervention)
        bias_norm_hist.append(float(stats.get("bias_norm", 0.0)))
        bias_gate_hist.append(float(stats.get("bias_gate", 0.0)))
        candidate_hist.append(float(stats.get("candidate_mass", 0.0)))
        cue_hist.append(float(stats.get("cue_mass", 0.0)))
        rows.append({
            "item_id": item.get("item_id", ""),
            "pair_id": item.get("pair_id", ""),
            "variant_id": int(item.get("variant_id", 0)),
            "family": item.get("family", "other"),
            "prompt": prompt,
            "candidates": candidates,
            "gold_answer": gold,
            "model_answer": pred,
            "correct": bool(correct),
            "active_token_count": int(len(set(token_ids))),
            "relation_ids": [int(x) for x in rel_ids],
            "active_op_count": int(active_op_count),
            "operator_entropy": float(entropy),
            "operator_top1_share": float(top1_op_share),
            "scope": float(scope.get("scope_total", 1.0)),
            "score_on_gold": float(logp_on_gold.item()),
            "score_on_foil": float(logp_on_foil.item()),
            "score_off_gold": float(logp_off_gold.item()),
            "answer_delta": float(answer_delta),
            "gold_delta_on_off": float(intervention),
            "cell_mode": {"A": "control", "B": "candidate_only", "C": "candidate_plus_cue", "D": "global_prompt_bias"}[cell],
            "candidate_spans": context["candidate_spans"],
            "cue_span": context["cue_span"],
            "cue_term": context["cue_term"],
            "bias_norm": float(stats.get("bias_norm", 0.0)),
            "bias_gate": float(stats.get("bias_gate", 0.0)),
            "candidate_mass": float(stats.get("candidate_mass", 0.0)),
            "cue_mass": float(stats.get("cue_mass", 0.0)),
        })
    n = max(1, len(rows))
    fam_adj = [r for r in rows if str(r["family"]) == "adjective_property"]
    fam_cau = [r for r in rows if str(r["family"]) == "causal_direction"]
    metrics = {
        "overall_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
        "adjective_accuracy": float(sum(1 for r in fam_adj if r["correct"]) / max(1, len(fam_adj))),
        "causal_accuracy": float(sum(1 for r in fam_cau if r["correct"]) / max(1, len(fam_cau))),
        "mean_answer_delta": float(sum(answer_delta_hist) / max(1, len(answer_delta_hist))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_active_op_count": float(sum(float(r["active_op_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_top1_op_share": float(sum(float(r["operator_top1_share"]) for r in rows) / n),
        "mean_scope": float(sum(float(r["scope"]) for r in rows) / n),
        "mean_intervention_delta_gold": float(sum(deltas) / max(1, len(deltas))),
        "mean_bias_norm": float(sum(bias_norm_hist) / max(1, len(bias_norm_hist))),
        "mean_bias_gate": float(sum(bias_gate_hist) / max(1, len(bias_gate_hist))),
        "mean_candidate_mass": float(sum(candidate_hist) / max(1, len(candidate_hist))),
        "mean_cue_mass": float(sum(cue_hist) / max(1, len(cue_hist))),
    }
    return metrics, rows


def _promotion_gates(report: dict[str, Any], min_acc_gain: float, scope_tol: float, intervention_min: float) -> dict[str, Any]:
    a = report["cells"]["A"]["metrics"]
    gates: dict[str, Any] = {}
    for cell in ("B", "C", "D"):
        m = report["cells"][cell]["metrics"]
        seed2 = report["cells"].get(f"{cell}_seed2_eval", {}).get("metrics", {})
        gates[cell] = {
            "accuracy_up": float(m["overall_accuracy"]) >= float(a["overall_accuracy"]) + float(min_acc_gain),
            "no_scope_regression": float(m["mean_scope"]) <= float(a["mean_scope"]) + float(scope_tol),
            "positive_intervention_delta": float(m["mean_intervention_delta_gold"]) >= float(intervention_min),
            "seed_stability": abs(float(seed2.get("overall_accuracy", m["overall_accuracy"])) - float(m["overall_accuracy"])) <= 0.05,
        }
        gates[cell]["promote_to_next"] = all(bool(v) for v in gates[cell].values())
    return gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.16 Continuous Graph Bias: soft structural attention bias from frozen System 1 graph into System 2 answer resolution.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--bias-norm-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias"))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    assert_output_path_allowed("M", args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter)
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    tok_src = str(adapter_path) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(adapter_path), local_files_only=args.local_files_only)
    target_device = torch.device("cuda" if torch.cuda.is_available() else _model_device(model))
    model = model.to(target_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ckpt = torch.load(args.checkpoint, map_location=target_device)
    hidden = int(model.config.hidden_size)
    module_dtype = next(model.parameters()).dtype
    codebook = BooleanAnchorTable(2000, hidden).to(target_device, dtype=module_dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    codebook.eval()
    for param in codebook.parameters():
        param.requires_grad = False
    arity_head = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    arity_head.eval()
    for param in arity_head.parameters():
        param.requires_grad = False
    base_adapter = CouncilCrossAttentionAdapter(hidden, use_boolean_surgery=True).to(target_device, dtype=module_dtype)
    base_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    base_adapter.eval()
    for param in base_adapter.parameters():
        param.requires_grad = False

    if args.pack_jsonl is None:
        pack_train, pack_val, pack_eval, split_meta = _build_winograd_split_packs(size=int(args.eval_size), seed=int(args.seed), strict_balance=bool(args.strict_balance))
        _t2, _v2, pack_eval_seed2, split_meta_seed2 = _build_winograd_split_packs(size=int(args.eval_size), seed=int(args.seed) + 1, strict_balance=bool(args.strict_balance))
    else:
        base_pack = _load_pack(args.pack_jsonl, size=int(args.eval_size) * 3, seed=int(args.seed), strict_balance=bool(args.strict_balance))
        pair_ids = sorted({str(r.get("pair_id", "")) for r in base_pack if str(r.get("pair_id", "")).strip()})
        rng = random.Random(int(args.seed))
        rng.shuffle(pair_ids)
        eval_count = max(1, round(len(pair_ids) * 0.30))
        val_count = max(1, round(len(pair_ids) * 0.15))
        train_ids = set(pair_ids[: max(1, len(pair_ids) - eval_count - val_count)])
        val_ids = set(pair_ids[max(1, len(pair_ids) - eval_count - val_count) : max(1, len(pair_ids) - eval_count)])
        eval_ids = set(pair_ids[max(1, len(pair_ids) - eval_count) :])
        if not eval_ids:
            eval_ids = set(pair_ids[-1:])
        if not val_ids:
            val_ids = set(pair_ids[-2:-1] or pair_ids[-1:])
        pack_train = [r for r in base_pack if str(r.get("pair_id", "")) in train_ids][: int(args.eval_size)]
        pack_val = [r for r in base_pack if str(r.get("pair_id", "")) in val_ids][: int(args.eval_size)]
        pack_eval = [r for r in base_pack if str(r.get("pair_id", "")) in eval_ids][: int(args.eval_size)]
        pack_eval_seed2 = _build_winograd_pack(size=int(args.eval_size), seed=int(args.seed) + 2, strict_balance=bool(args.strict_balance))
        split_meta = {"train_pair_ids": sorted(train_ids), "val_pair_ids": sorted(val_ids), "eval_pair_ids": sorted(eval_ids)}
        split_meta_seed2 = {"seed": int(args.seed) + 1}

    (run_dir / "m3_16_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M3.16", "scripts/run_m3_16_continuous_graph_bias.py"),
        "track": "M3.16",
        "lineage": lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed"),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {k: str(v) for k, v in vars(args).items()},
        "data_split": {**split_meta, "seed2_eval_meta": split_meta_seed2, "runtime_policy_source": "validation_metrics", "final_metrics_source": "eval_pack"},
        "cells": {},
    }

    def make_bias_adapter() -> SoftGraphBiasCouncilAdapter:
        mod = SoftGraphBiasCouncilAdapter(hidden, use_boolean_surgery=True).to(target_device, dtype=module_dtype)
        mod.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
        mod.freeze_base()
        mod.eval()
        mod.clear_graph_bias()
        return mod

    adapter_a = make_bias_adapter()
    train_a = _train_cell("A", adapter_a, model, tokenizer, codebook, arity_head, pack_train, args)
    met_a_val, _ = _evaluate_cell("A", adapter_a, model, tokenizer, codebook, arity_head, base_adapter, pack_val, args)
    met_a, rows_a = _evaluate_cell("A", adapter_a, model, tokenizer, codebook, arity_head, base_adapter, pack_eval, args)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val}
    (run_dir / "m3_16_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")

    for cell in ("B", "C", "D"):
        adapter = make_bias_adapter()
        train_cell = _train_cell(cell, adapter, model, tokenizer, codebook, arity_head, pack_train, args)
        met_val, _ = _evaluate_cell(cell, adapter, model, tokenizer, codebook, arity_head, base_adapter, pack_val, args)
        met_eval, rows_eval = _evaluate_cell(cell, adapter, model, tokenizer, codebook, arity_head, base_adapter, pack_eval, args)
        met_seed2, rows_seed2 = _evaluate_cell(cell, adapter, model, tokenizer, codebook, arity_head, base_adapter, pack_eval_seed2, args)
        report["cells"][cell] = {"train": train_cell, "metrics": met_eval, "validation_metrics": met_val}
        report["cells"][f"{cell}_seed2_eval"] = {"metrics": met_seed2}
        (run_dir / f"m3_16_{cell}_eval.json").write_text(json.dumps(rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m3_16_{cell}_seed2_eval.json").write_text(json.dumps(rows_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(report, min_acc_gain=0.02, scope_tol=0.02, intervention_min=0.01)
    (run_dir / "m3_16_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# M3.16 Continuous Graph Bias",
        "",
        "| Cell | Regime | Acc | Adj Acc | Causal Acc | Answer Delta | Gold On-Off | Active Ops | Entropy | Scope | Cand Mass | Cue Mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"A": "control frozen bridge base", "B": "candidate-only soft graph bias", "C": "candidate+cue soft graph bias", "D": "global prompt soft graph bias"}
    for c in ("A", "B", "C", "D"):
        m = report["cells"][c]["metrics"]
        md.append(f"| {c} | {labels[c]} | {m['overall_accuracy']:.3f} | {m['adjective_accuracy']:.3f} | {m['causal_accuracy']:.3f} | {m['mean_answer_delta']:.4f} | {m['mean_intervention_delta_gold']:.4f} | {m['mean_active_op_count']:.2f} | {m['mean_operator_entropy']:.4f} | {m['mean_scope']:.4f} | {m['mean_candidate_mass']:.4f} | {m['mean_cue_mass']:.4f} |")
    md.extend(["", "## Regimes", "- B: graph-derived bias only on candidate spans.", "- C: graph-derived bias on candidate spans plus cue span.", "- D: graph-derived bias over the whole prompt token field.", "", "## Promotion Gates"])
    for cell in ("B", "C", "D"):
        md.append(f"- {cell}:")
        for key, value in report["promotion_gates"][cell].items():
            md.append(f"  - {key}: `{value}`")
    (run_dir / "m3_16_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M3.16 complete: {run_dir}")


if __name__ == "__main__":
    main()
