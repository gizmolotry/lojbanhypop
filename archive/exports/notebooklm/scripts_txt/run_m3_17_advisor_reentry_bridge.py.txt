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
from lojban_evolution.m_bridge_ablation_family import build_bridge_report, finalize_bridge_report, track_cell_labels
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_15d_answer_path_forcing import (  # type: ignore
    _answer_match,
    _build_attention_mask_for_final_prefix,
    _build_winograd_pack,
    _build_winograd_split_packs,
    _candidate_first_token_id,
    _decode_logic_tokens,
    _load_pack,
    _model_device,
    _triples,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    extract_trace_hidden_states,
)


class AdvisorReentryBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = 64, max_return_tokens: int = 3) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_return_tokens = int(max_return_tokens)

        self.prefix_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.advisor_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.advisor_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.learned_queries = torch.nn.Parameter(torch.empty(1, self.max_return_tokens, hidden_size))
        torch.nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)

        self.token_down = torch.nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.token_up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.residual_down = torch.nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.residual_up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)

        self.token_gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.residual_gate = torch.nn.Parameter(torch.tensor(-2.0))

    def _attend(self, prefix_hidden: torch.Tensor, advisor_states: torch.Tensor, n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        b = prefix_hidden.shape[0]
        q0 = self.prefix_q(prefix_hidden[:, -1:, :])
        q = q0 + self.learned_queries[:, : int(n_tokens), :].expand(b, -1, -1)
        k = self.advisor_k(advisor_states)
        v = self.advisor_v(advisor_states)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn

    def return_tokens(self, prefix_hidden: torch.Tensor, advisor_states: torch.Tensor, n_tokens: int) -> tuple[torch.Tensor, dict[str, float]]:
        context, attn = self._attend(prefix_hidden, advisor_states, n_tokens)
        tokens = self.token_up(F.relu(self.token_down(context)))
        gate = torch.sigmoid(self.token_gate)
        tokens = torch.tanh(tokens) * gate
        stats = {
            "gate": float(gate.detach().item()),
            "token_norm": float(torch.norm(tokens, dim=-1).mean().detach().item()),
            "attn_entropy": float((-(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()).detach().item()),
        }
        return tokens, stats

    def residual_delta(self, prefix_hidden: torch.Tensor, advisor_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        context, attn = self._attend(prefix_hidden, advisor_states, 1)
        delta = self.residual_up(F.relu(self.residual_down(context)))
        gate = torch.sigmoid(self.residual_gate)
        delta = torch.tanh(delta) * gate
        stats = {
            "gate": float(gate.detach().item()),
            "token_norm": float(torch.norm(delta, dim=-1).mean().detach().item()),
            "attn_entropy": float((-(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()).detach().item()),
        }
        return delta, stats


def _build_base_advisor_state(
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    prompt: str,
    relation_vocab: int,
    var_min_id: int,
    max_logic_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(max_logic_new_tokens))
    z_st, _idx, _cb, _commit = codebook.quantize(h_t)
    base_tokens = _decode_logic_tokens(arity_head, z_st, int(relation_vocab), int(var_min_id))
    base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
    base_ids = torch.stack(base_tokens, dim=1)
    token_ids = [int(t[0].detach().item()) for t in base_tokens]
    return base_states, base_ids, token_ids


def _extract_reentry_context(model, tokenizer, prompt: str, layer_index: int) -> dict[str, torch.Tensor]:
    dev = _model_device(model)
    p_ids, am = _build_attention_mask_for_final_prefix(tokenizer, prompt, dev, blindfold_question=False)
    out = model(input_ids=p_ids, attention_mask=am, output_hidden_states=True, use_cache=False, return_dict=True)
    prefix_hidden = out.hidden_states[int(layer_index)].detach()
    h_base = out.hidden_states[-1][:, -1:, :].detach()
    prefix_embs = model.get_input_embeddings()(p_ids).detach()
    return {
        "prefix_ids": p_ids,
        "attention_mask": am,
        "prefix_hidden": prefix_hidden,
        "h_base": h_base,
        "prefix_embs": prefix_embs,
    }


def _score_candidate_from_hidden(model, tokenizer, candidate: str, hidden: torch.Tensor) -> torch.Tensor:
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    logits = model.lm_head(hidden)
    return torch.log_softmax(logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _score_candidate_from_return_tokens(model, tokenizer, candidate: str, prefix_embs: torch.Tensor, return_tokens: torch.Tensor) -> torch.Tensor:
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    full_embs = torch.cat([prefix_embs, return_tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)], dim=1)
    full_am = torch.ones((1, full_embs.shape[1]), dtype=torch.long, device=full_embs.device)
    out = model(inputs_embeds=full_embs, attention_mask=full_am, use_cache=False, return_dict=True)
    return torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _advisor_stats(token_ids: list[int]) -> dict[str, float]:
    rel_ids = token_ids[0::3]
    counts = Counter(rel_ids)
    total = max(1, len(rel_ids))
    probs = [float(c) / float(total) for c in counts.values()]
    entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
    top1 = float(max(counts.values())) / float(total) if counts else 1.0
    return {
        "active_token_count": float(len(set(token_ids))),
        "active_op_count": float(sum(1 for _k, v in counts.items() if int(v) >= 1)),
        "operator_entropy": float(entropy),
        "operator_top1_share": float(top1),
    }


def _train_cell(cell: str, bridge: AdvisorReentryBridge, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, train_pack: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float]:
    if cell == "A":
        return {"answer_path_loss": 0.0, "answer_delta": 0.0, "return_norm": 0.0, "gate": 0.0, "attn_entropy": 0.0}

    opt = torch.optim.AdamW(bridge.parameters(), lr=float(args.lr), weight_decay=0.01)
    loss_hist: list[float] = []
    delta_hist: list[float] = []
    norm_hist: list[float] = []
    gate_hist: list[float] = []
    entropy_hist: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            ctx = _extract_reentry_context(model, tokenizer, prompt, int(args.layer_index))
            advisor_states, _advisor_ids, _token_ids = _build_base_advisor_state(
                model,
                tokenizer,
                codebook,
                arity_head,
                prompt,
                int(args.relation_vocab),
                int(args.var_min_id),
                int(args.max_logic_new_tokens),
            )

        opt.zero_grad()
        if cell == "B":
            return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, 1)
            logp_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
            logp_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
        elif cell == "C":
            return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, int(args.num_return_tokens))
            logp_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
            logp_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
        elif cell == "D":
            delta, stats = bridge.residual_delta(ctx["prefix_hidden"], advisor_states)
            logp_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"] + delta)
            logp_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"] + delta)
        else:
            raise ValueError(f"Unsupported cell {cell!r}")

        answer_delta = logp_gold - logp_foil
        answer_loss = torch.relu(torch.tensor(float(args.margin), device=answer_delta.device, dtype=answer_delta.dtype) - answer_delta)
        if cell in {"B", "C"}:
            reg = torch.norm(return_tokens, dim=-1).mean()
        else:
            reg = torch.norm(delta, dim=-1).mean()
        loss = float(args.answer_weight) * answer_loss + float(args.return_norm_weight) * reg
        loss.backward()
        opt.step()

        loss_hist.append(float(answer_loss.detach().item()))
        delta_hist.append(float(answer_delta.detach().item()))
        norm_hist.append(float(stats["token_norm"]))
        gate_hist.append(float(stats["gate"]))
        entropy_hist.append(float(stats["attn_entropy"]))

    return {
        "answer_path_loss": float(sum(loss_hist) / max(1, len(loss_hist))),
        "answer_delta": float(sum(delta_hist) / max(1, len(delta_hist))),
        "return_norm": float(sum(norm_hist) / max(1, len(norm_hist))),
        "gate": float(sum(gate_hist) / max(1, len(gate_hist))),
        "attn_entropy": float(sum(entropy_hist) / max(1, len(entropy_hist))),
    }


def _evaluate_cell(cell: str, bridge: AdvisorReentryBridge, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, eval_pack: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    answer_delta_hist: list[float] = []
    norm_hist: list[float] = []
    gate_hist: list[float] = []
    entropy_hist: list[float] = []

    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            ctx = _extract_reentry_context(model, tokenizer, prompt, int(args.layer_index))
            advisor_states, _advisor_ids, token_ids = _build_base_advisor_state(
                model,
                tokenizer,
                codebook,
                arity_head,
                prompt,
                int(args.relation_vocab),
                int(args.var_min_id),
                int(args.max_logic_new_tokens),
            )
            base_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"])
            base_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"])

            if cell == "A":
                on_gold = base_gold
                on_foil = base_foil
                stats = {"gate": 0.0, "token_norm": 0.0, "attn_entropy": 0.0}
            elif cell == "B":
                return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, 1)
                on_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
                on_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
            elif cell == "C":
                return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, int(args.num_return_tokens))
                on_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
                on_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
            elif cell == "D":
                delta, stats = bridge.residual_delta(ctx["prefix_hidden"], advisor_states)
                on_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"] + delta)
                on_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"] + delta)
            else:
                raise ValueError(f"Unsupported cell {cell!r}")

        pred_idx = gi if float(on_gold.item()) >= float(on_foil.item()) else fi
        pred = str(candidates[pred_idx])
        correct = _answer_match(gold, pred)
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        advisor = _advisor_stats(token_ids)
        answer_delta = float((on_gold - on_foil).item())
        intervention = float((on_gold - base_gold).item())

        answer_delta_hist.append(answer_delta)
        deltas.append(intervention)
        norm_hist.append(float(stats["token_norm"]))
        gate_hist.append(float(stats["gate"]))
        entropy_hist.append(float(stats["attn_entropy"]))

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
            "active_token_count": int(advisor["active_token_count"]),
            "active_op_count": int(advisor["active_op_count"]),
            "operator_entropy": float(advisor["operator_entropy"]),
            "operator_top1_share": float(advisor["operator_top1_share"]),
            "scope": float(scope.get("scope_total", 1.0)),
            "score_on_gold": float(on_gold.item()),
            "score_on_foil": float(on_foil.item()),
            "score_off_gold": float(base_gold.item()),
            "answer_delta": float(answer_delta),
            "gold_delta_on_off": float(intervention),
            "return_norm": float(stats["token_norm"]),
            "return_gate": float(stats["gate"]),
            "return_attn_entropy": float(stats["attn_entropy"]),
            "cell_mode": {
                "A": "control_no_reentry",
                "B": "single_return_state",
                "C": "three_return_states",
                "D": "direct_residual_reencoder",
            }[cell],
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
        "mean_return_norm": float(sum(norm_hist) / max(1, len(norm_hist))),
        "mean_return_gate": float(sum(gate_hist) / max(1, len(gate_hist))),
        "mean_return_attn_entropy": float(sum(entropy_hist) / max(1, len(entropy_hist))),
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
    p = argparse.ArgumentParser(description="M3.17 Advisor Re-entry Bridge: compress advisor cognition into decoder-native return states before answer continuation.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--num-return-tokens", type=int, default=3)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--return-norm-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge"))
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

    (run_dir / "m3_17_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")
    report: dict[str, Any] = build_bridge_report(
        track="M3.17",
        script_path="scripts/run_m3_17_advisor_reentry_bridge.py",
        args=args,
        baseline_manifest_path=args.baseline_manifest,
        baseline_id=str(baseline_manifest.get("baseline_id", "")),
        checkpoint_in=str(args.checkpoint),
        split_meta=split_meta,
        seed2_meta=split_meta_seed2,
        runtime_policy_source="validation_metrics",
        final_metrics_source="eval_pack",
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["series"] = series_metadata("M", "M3.17", "scripts/run_m3_17_advisor_reentry_bridge.py")
    report["lineage"] = lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed")

    def make_bridge() -> AdvisorReentryBridge:
        mod = AdvisorReentryBridge(hidden, bottleneck_dim=int(args.bottleneck_dim), max_return_tokens=int(args.num_return_tokens)).to(target_device, dtype=module_dtype)
        mod.eval()
        return mod

    bridge_a = make_bridge()
    train_a = _train_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_train, args)
    met_a_val, _ = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_val, args)
    met_a, rows_a = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_eval, args)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val}
    (run_dir / "m3_17_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")

    for cell in ("B", "C", "D"):
        bridge = make_bridge()
        train_cell = _train_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_train, args)
        met_val, _ = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_val, args)
        met_eval, rows_eval = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_eval, args)
        met_seed2, rows_seed2 = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_eval_seed2, args)
        report["cells"][cell] = {"train": train_cell, "metrics": met_eval, "validation_metrics": met_val}
        report["cells"][f"{cell}_seed2_eval"] = {"metrics": met_seed2}
        (run_dir / f"m3_17_{cell}_eval.json").write_text(json.dumps(rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m3_17_{cell}_seed2_eval.json").write_text(json.dumps(rows_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(report, min_acc_gain=0.02, scope_tol=0.02, intervention_min=0.01)
    report = finalize_bridge_report(report, "M3.17")
    (run_dir / "m3_17_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M3.17 Advisor Re-entry Bridge",
        "",
        "| Cell | Regime | Acc | Adj Acc | Causal Acc | Answer Delta | Gold On-Off | Return Norm | Gate | Attn Entropy | Scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = track_cell_labels("M3.17")
    for c in ("A", "B", "C", "D"):
        m = report["cells"][c]["metrics"]
        md.append(f"| {c} | {labels[c]} | {m['overall_accuracy']:.3f} | {m['adjective_accuracy']:.3f} | {m['causal_accuracy']:.3f} | {m['mean_answer_delta']:.4f} | {m['mean_intervention_delta_gold']:.4f} | {m['mean_return_norm']:.4f} | {m['mean_return_gate']:.4f} | {m['mean_return_attn_entropy']:.4f} | {m['mean_scope']:.4f} |")
    md.extend([
        "",
        "## Regimes",
        "- B: compress advisor state into one return state appended once before answer continuation.",
        "- C: compress advisor state into a short return-state bundle appended before answer continuation.",
        "- D: translate advisor state into one decoder-compatible residual continuation vector.",
        "",
        "## Promotion Gates",
    ])
    for cell in ("B", "C", "D"):
        md.append(f"- {cell}:")
        for key, value in report["promotion_gates"][cell].items():
            md.append(f"  - {key}: `{value}`")
    (run_dir / "m3_17_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M3.17 complete: {run_dir}")


if __name__ == "__main__":
    main()



