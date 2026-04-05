from __future__ import annotations

import argparse
import json
import math
import random
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from build_m3_19_resumption_pack import BUCKETS, _build_row, _specs, _validate_rows  # type: ignore
from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from lojban_evolution.m_bridge_ablation_family import BRIDGE_ABLATION_REGISTRY
from lojban_evolution.m_symbiote_scratchpad_family import (
    SYMBIOTE_SCRATCHPAD_REGISTRY,
    build_scratchpad_protocol_manifest,
    scratchpad_cell_labels,
)
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_15d_answer_path_forcing import (  # type: ignore
    _answer_match,
    _candidate_first_token_id,
    _load_pack,
    _model_device,
    _triples,
)
from run_m3_17_advisor_reentry_bridge import (  # type: ignore
    _advisor_stats,
    _build_base_advisor_state,
)
from run_m3_18_decoder_reentry_resume import (  # type: ignore
    _continuation_overlap_f1,
    _english_fluency_score,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    _resolve_layers,
)


LOJBAN_BLE_RE = re.compile(r"(<\s*loj_[^>\s]+>|loj_[a-z0-9_]+)", re.IGNORECASE)


def _latest_named_file(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(file_name), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]

def _build_m11_advisor_state(
    model,
    s1,
    prompt: str,
    tokenizer,
    device
) -> torch.Tensor:
    """
    Track 2: The M11 Hyper-Modulated Advisor (Rg).
    Generates logic tensors [1, 10, H] using the Multi-Slot Role Geometry.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_s2 = model(**inputs, output_hidden_states=True)
        h_eng = out_s2.hidden_states[-1][:, -1, :]
        
        # Extract prompt embeddings for dynamic PointerBind
        prompt_embs = model.get_input_embeddings()(inputs.input_ids)
        
        # M11 Forge Call (Rg API)
        op_vec, x_probs, _ = s1.build_graph(h_eng, prompt_embs)
        
        # ptr_vecs is [1, 10, H] - we project role probabilities onto embs
        ptr_vecs = torch.matmul(x_probs, prompt_embs)
        
        # logic_state: [1, 10, H]
        logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)
    return logic_state


def _round_robin_rich_pack(size: int, seed: int, strict_balance: bool) -> list[dict[str, Any]]:
    if strict_balance and size % (len(BUCKETS) * 2) != 0:
        raise ValueError("strict balance requires pack size divisible by 8")
    rng = random.Random(int(seed))
    specs = _specs()
    rows: list[dict[str, Any]] = []
    pair_id = 0
    while len(rows) < size:
        spec = specs[pair_id % len(specs)]
        bucket = BUCKETS[(pair_id // len(specs)) % len(BUCKETS)]
        rows.append(_build_row(spec, pair_id=pair_id, variant_index=0, bucket=bucket))
        if len(rows) < size:
            rows.append(_build_row(spec, pair_id=pair_id, variant_index=1, bucket=bucket))
        pair_id += 1
    rows = rows[:size]
    rng.shuffle(rows)
    _validate_rows(rows, strict_balance=bool(strict_balance))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _pack_summary(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "size": len(rows),
        "bucket_counts": {bucket: sum(1 for r in rows if str(r.get("bucket")) == bucket) for bucket in BUCKETS},
        "bucket_label_counts": {
            bucket: {
                "candidate_0": sum(1 for r in rows if str(r.get("bucket")) == bucket and int(r.get("gold_index", -1)) == 0),
                "candidate_1": sum(1 for r in rows if str(r.get("bucket")) == bucket and int(r.get("gold_index", -1)) == 1),
            }
            for bucket in BUCKETS
        },
        "families": sorted({str(r.get("family", "")) for r in rows}),
        "output": str(out_path).replace("\\", "/"),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resume_target_text(item: dict[str, Any]) -> str:
    explicit = str(item.get("target_text", "") or item.get("resumption_target", "")).strip()
    if explicit:
        return explicit
    candidates = list(item.get("candidates", []))
    gold_index = int(item.get("gold_index", 0))
    if 0 <= gold_index < len(candidates):
        return str(candidates[gold_index]).strip()
    return ""


def _resume_target_ids(tokenizer, item: dict[str, Any], max_tokens: int) -> list[int]:
    text = _resume_target_text(item)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return [int(x) for x in ids[: max(1, int(max_tokens))]]


def _contains_lojban_bleed(text: str) -> bool:
    return bool(LOJBAN_BLE_RE.search(str(text)))


def _loop_flag(token_ids: list[int]) -> bool:
    if len(token_ids) < 3:
        return False
    if len(set(token_ids)) <= max(1, len(token_ids) // 2):
        return True
    for n in (1, 2):
        if len(token_ids) < n * 2:
            continue
        seen: set[tuple[int, ...]] = set()
        for i in range(0, len(token_ids) - n + 1):
            gram = tuple(int(t) for t in token_ids[i : i + n])
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _build_symbiote_prefix(question: str, scratchpad_token: str, scratchpad_length: int) -> str:
    scratchpad = " ".join([scratchpad_token] * int(scratchpad_length))
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        f"{scratchpad}\n"
        "Final answer:"
    )


def _ensure_symbiote_token(model, tokenizer, scratchpad_token: str) -> int:
    vocab = tokenizer.get_vocab()
    if scratchpad_token in vocab:
        token_id = int(tokenizer.convert_tokens_to_ids(scratchpad_token))
        model.resize_token_embeddings(len(tokenizer))
        return token_id

    old_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [scratchpad_token]})
    model.resize_token_embeddings(len(tokenizer))
    token_id = int(tokenizer.convert_tokens_to_ids(scratchpad_token))

    input_emb = model.get_input_embeddings().weight
    with torch.no_grad():
        mean_in = input_emb[:old_size].mean(dim=0, keepdim=True)
        input_emb[old_size : len(tokenizer)] = mean_in.to(device=input_emb.device, dtype=input_emb.dtype)
        output_emb = model.get_output_embeddings()
        if output_emb is not None and hasattr(output_emb, "weight") and output_emb.weight.shape[0] >= len(tokenizer):
            mean_out = output_emb.weight[:old_size].mean(dim=0, keepdim=True)
            output_emb.weight[old_size : len(tokenizer)] = mean_out.to(device=output_emb.weight.device, dtype=output_emb.weight.dtype)
    return token_id


def _scratchpad_attention_mass(attentions: tuple[torch.Tensor, ...] | None, scratchpad_mask: torch.Tensor) -> float:
    if not attentions:
        return 0.0
    mask = scratchpad_mask[0].to(dtype=torch.bool)
    if int(mask.sum().item()) <= 0:
        return 0.0
    masses: list[float] = []
    for layer_attn in attentions:
        val = layer_attn[0, :, -1, mask].sum(dim=-1).mean()
        masses.append(float(val.detach().item()))
    return float(sum(masses) / max(1, len(masses)))


def _build_scratchpad_context(
    model,
    tokenizer,
    prompt: str,
    layer_index: int,
    symbiote_token_id: int,
    scratchpad_token: str,
    scratchpad_length: int,
    *,
    output_attentions: bool = False,
) -> dict[str, Any]:
    full_prompt = _build_symbiote_prefix(prompt, scratchpad_token, scratchpad_length)
    dev = _model_device(model)
    prefix_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(dev)
    attention_mask = torch.ones_like(prefix_ids, device=dev)
    with torch.no_grad():
        out = model(
            input_ids=prefix_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )
    scratchpad_mask = prefix_ids.eq(int(symbiote_token_id))
    scratchpad_count = int(scratchpad_mask.sum().item())
    if scratchpad_count != int(scratchpad_length):
        raise RuntimeError(f"Expected {scratchpad_length} scratchpad tokens, found {scratchpad_count} in prompt.")
    layer_hidden = out.hidden_states[int(layer_index)].detach()
    scratchpad_hidden = layer_hidden[:, scratchpad_mask[0], :].detach()
    return {
        "prompt_text": full_prompt,
        "prefix_ids": prefix_ids,
        "attention_mask": attention_mask,
        "scratchpad_mask": scratchpad_mask,
        "scratchpad_hidden": scratchpad_hidden,
        "clean_output": out,
        "attention_mass": _scratchpad_attention_mass(out.attentions if output_attentions else None, scratchpad_mask),
    }


@contextmanager
def _scratchpad_injection_hook(model, layer_index: int, scratchpad_mask: torch.Tensor, delta: torch.Tensor):
    layers = _resolve_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for scratchpad injection.")
    positions = torch.nonzero(scratchpad_mask[0], as_tuple=False).flatten()

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        hidden = hidden.clone()
        delta_t = delta.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, positions, :] = hidden[:, positions, :] + delta_t
        return (hidden, *rest) if rest is not None else hidden

    handle = layers[int(layer_index)].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _forward_with_optional_injection(
    model,
    seq_ids: torch.Tensor,
    layer_index: int,
    scratchpad_mask: torch.Tensor,
    delta: torch.Tensor | None,
    *,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
):
    ctx = _scratchpad_injection_hook(model, int(layer_index), scratchpad_mask, delta) if delta is not None else nullcontext()
    with ctx:
        return model(
            input_ids=seq_ids,
            attention_mask=torch.ones_like(seq_ids),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
        )


class SymbioteScratchpadBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down = torch.nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = torch.nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, scratchpad_hidden: torch.Tensor, advisor_states: torch.Tensor, *, alpha: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        q = self.q_proj(scratchpad_hidden)
        k = self.k_proj(advisor_states)
        v = self.v_proj(advisor_states)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        delta = self.up(F.relu(self.down(context)))
        gate = torch.sigmoid(self.gate) * float(alpha)
        delta = torch.tanh(delta) * gate
        stats = {
            "gate": float(gate.detach().item()),
            "residual_norm": float(torch.norm(delta, dim=-1).mean().detach().item()),
            "attn_entropy": float((-(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()).detach().item()),
        }
        return delta, stats


class SymbioteTokenBaseline(torch.nn.Module):
    def __init__(self, hidden_size: int, scratchpad_length: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.scratchpad_length = int(scratchpad_length)
        self.token_offsets = torch.nn.Parameter(torch.empty(1, self.scratchpad_length, self.hidden_size))
        torch.nn.init.normal_(self.token_offsets, mean=0.0, std=0.02)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, *, alpha: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        gate = torch.sigmoid(self.gate) * float(alpha)
        delta = torch.tanh(self.token_offsets) * gate
        stats = {
            "gate": float(gate.detach().item()),
            "residual_norm": float(torch.norm(delta, dim=-1).mean().detach().item()),
            "attn_entropy": 0.0,
        }
        return delta, stats


def _candidate_log_prob(logits: torch.Tensor, tokenizer, candidate: str) -> torch.Tensor:
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    return torch.log_softmax(logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _teacher_forced_ce(
    model,
    prefix_ids: torch.Tensor,
    target_ids: list[int],
    layer_index: int,
    symbiote_token_id: int,
    delta: torch.Tensor | None,
) -> torch.Tensor:
    if not target_ids:
        return torch.zeros((), device=prefix_ids.device, dtype=model.get_input_embeddings().weight.dtype)

    prefix_len = int(prefix_ids.shape[1])
    if len(target_ids) == 1:
        seq_ids = prefix_ids
    else:
        prev = torch.tensor(target_ids[:-1], device=prefix_ids.device, dtype=prefix_ids.dtype).unsqueeze(0)
        seq_ids = torch.cat([prefix_ids, prev], dim=1)
    scratchpad_mask = seq_ids.eq(int(symbiote_token_id))
    out = _forward_with_optional_injection(
        model,
        seq_ids,
        layer_index,
        scratchpad_mask,
        delta,
        output_hidden_states=False,
        output_attentions=False,
    )
    target = torch.tensor(target_ids, device=seq_ids.device, dtype=torch.long)
    logits_slice = out.logits[:, prefix_len - 1 : prefix_len - 1 + len(target_ids), :]
    return F.cross_entropy(logits_slice.reshape(-1, logits_slice.shape[-1]), target.reshape(-1))


def _comparison_snapshot(m318_report: Path | None, m319_report: Path | None, m11_manifest: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if m318_report and m318_report.exists():
        data = json.loads(m318_report.read_text(encoding="utf-8"))
        d_metrics = data.get("cells", {}).get("D", {}).get("metrics", {})
        payload["M3.18.D"] = {
            "report": str(m318_report).replace("\\", "/"),
            "overall_accuracy": d_metrics.get("overall_accuracy"),
            "resume_first_token_accuracy": d_metrics.get("resume_first_token_accuracy"),
            "english_fluency_score": d_metrics.get("english_fluency_score"),
            "loop_rate": d_metrics.get("loop_rate"),
            "mean_intervention_delta_gold": d_metrics.get("mean_intervention_delta_gold"),
        }
    if m319_report and m319_report.exists():
        data = json.loads(m319_report.read_text(encoding="utf-8"))
        cells = data.get("cells", {})
        payload["M3.19"] = {
            "report": str(m319_report).replace("\\", "/"),
            "cells": {
                cell_id: {
                    "overall_accuracy": cells.get(cell_id, {}).get("metrics", {}).get("overall_accuracy"),
                    "resume_first_token_accuracy": cells.get(cell_id, {}).get("metrics", {}).get("resume_first_token_accuracy"),
                    "english_fluency_score": cells.get(cell_id, {}).get("metrics", {}).get("english_fluency_score"),
                    "loop_rate": cells.get(cell_id, {}).get("metrics", {}).get("loop_rate"),
                    "mean_intervention_delta_gold": cells.get(cell_id, {}).get("metrics", {}).get("mean_intervention_delta_gold"),
                }
                for cell_id in ("D0", "D1", "D2", "D3")
                if cell_id in cells
            },
        }
    if m11_manifest and m11_manifest.exists():
        data = json.loads(m11_manifest.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        payload["M11.discriminative"] = {
            "manifest": str(m11_manifest).replace("\\", "/"),
            "headline_accuracy": metrics.get("mean_accuracy", metrics.get("accuracy")),
            "headline_macro_f1": metrics.get("macro_f1"),
            "num_samples": metrics.get("n_samples", metrics.get("num_samples")),
        }
    return payload


def _compute_delta_for_cell(
    cell: str,
    bridge: SymbioteScratchpadBridge | None,
    token_baseline: SymbioteTokenBaseline | None,
    scratchpad_hidden: torch.Tensor,
    advisor_states: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if cell == "A":
        return None, {"gate": 0.0, "residual_norm": 0.0, "attn_entropy": 0.0}
    if cell in {"B", "C", "D"}:
        assert bridge is not None and advisor_states is not None
        return bridge(scratchpad_hidden, advisor_states, alpha=float(args.scratchpad_alpha))
    if cell == "E":
        assert token_baseline is not None
        return token_baseline(alpha=float(args.scratchpad_alpha))
    raise ValueError(f"Unsupported cell {cell!r}")


def _cell_guard_threshold(cell: str, args: argparse.Namespace) -> float:
    if cell == "B":
        return float(args.b_guard_threshold)
    if cell == "C":
        return float(args.c_guard_threshold)
    if cell == "D":
        return float(args.d_guard_threshold)
    return 0.0


def _train_cell(
    cell: str,
    bridge: SymbioteScratchpadBridge | None,
    token_baseline: SymbioteTokenBaseline | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    symbiote_token_id: int,
    train_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    if cell == "A":
        return {
            "answer_path_loss": 0.0,
            "continuation_ce_loss": 0.0,
            "answer_delta": 0.0,
            "scratchpad_residual_norm": 0.0,
            "scratchpad_gate_mean": 0.0,
            "scratchpad_gate_overflow_rate": 0.0,
            "scratchpad_attention_mass": 0.0,
            "train_objective": "none",
        }

    params = list((bridge.parameters() if bridge is not None else [])) + list((token_baseline.parameters() if token_baseline is not None else []))
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.01)
    answer_loss_hist: list[float] = []
    ce_hist: list[float] = []
    delta_hist: list[float] = []
    norm_hist: list[float] = []
    overflow_hist: list[float] = []
    gate_hist: list[float] = []
    attn_mass_hist: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        target_ids = _resume_target_ids(tokenizer, item, int(args.continuation_target_max_tokens))

        with torch.no_grad():
            ctx = _build_scratchpad_context(
                model,
                tokenizer,
                prompt,
                int(args.layer_index),
                int(symbiote_token_id),
                str(args.scratchpad_token),
                int(args.scratchpad_length),
                output_attentions=True,
            )
            advisor_states = None
            if cell in {"B", "C", "D"}:
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
        delta, stats = _compute_delta_for_cell(cell, bridge, token_baseline, ctx["scratchpad_hidden"], advisor_states, args)
        out = _forward_with_optional_injection(
            model,
            ctx["prefix_ids"],
            int(args.layer_index),
            ctx["scratchpad_mask"],
            delta,
            output_hidden_states=False,
            output_attentions=False,
        )
        logp_gold = _candidate_log_prob(out.logits, tokenizer, gold)
        logp_foil = _candidate_log_prob(out.logits, tokenizer, foil)
        answer_delta = logp_gold - logp_foil
        answer_loss = torch.relu(torch.tensor(float(args.margin), device=answer_delta.device, dtype=answer_delta.dtype) - answer_delta)
        continuation_ce = _teacher_forced_ce(
            model,
            ctx["prefix_ids"],
            target_ids,
            int(args.layer_index),
            int(symbiote_token_id),
            delta,
        )
        norm = torch.norm(delta, dim=-1).mean() if delta is not None else torch.zeros_like(answer_loss)
        guard_threshold = _cell_guard_threshold(cell, args)
        scratchpad_guard = torch.relu(norm - guard_threshold) if cell in {"B", "C", "D"} else torch.zeros_like(answer_loss)
        loss = (
            float(args.answer_weight) * continuation_ce
            + float(args.return_norm_weight) * norm
            + float(args.residual_guard_weight) * scratchpad_guard
        )
        loss.backward()
        opt.step()

        answer_loss_hist.append(float(answer_loss.detach().item()))
        ce_hist.append(float(continuation_ce.detach().item()))
        delta_hist.append(float(answer_delta.detach().item()))
        norm_hist.append(float(stats["residual_norm"]))
        overflow_hist.append(float(stats["residual_norm"] > guard_threshold) if cell in {"B", "C", "D"} else 0.0)
        gate_hist.append(float(stats["gate"]))
        attn_mass_hist.append(float(ctx["attention_mass"]))

    return {
        "answer_path_loss": float(sum(answer_loss_hist) / max(1, len(answer_loss_hist))),
        "continuation_ce_loss": float(sum(ce_hist) / max(1, len(ce_hist))),
        "answer_delta": float(sum(delta_hist) / max(1, len(delta_hist))),
        "scratchpad_residual_norm": float(sum(norm_hist) / max(1, len(norm_hist))),
        "scratchpad_gate_mean": float(sum(gate_hist) / max(1, len(gate_hist))),
        "scratchpad_gate_overflow_rate": float(sum(overflow_hist) / max(1, len(overflow_hist))),
        "scratchpad_attention_mass": float(sum(attn_mass_hist) / max(1, len(attn_mass_hist))),
        "train_objective": "continuation_ce" if cell in {"B", "C", "D", "E"} else "none",
    }


def _evaluate_cell(
    cell: str,
    bridge: SymbioteScratchpadBridge | None,
    token_baseline: SymbioteTokenBaseline | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    symbiote_token_id: int,
    eval_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    answer_delta_hist: list[float] = []
    scope_hist: list[float] = []
    residual_norm_hist: list[float] = []
    gate_hist: list[float] = []
    attn_entropy_hist: list[float] = []
    attn_mass_hist: list[float] = []
    overflow_hist: list[float] = []

    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            ctx = _build_scratchpad_context(
                model,
                tokenizer,
                prompt,
                int(args.layer_index),
                int(symbiote_token_id),
                str(args.scratchpad_token),
                int(args.scratchpad_length),
                output_attentions=True,
            )
            base_logits = ctx["clean_output"].logits
            base_gold = _candidate_log_prob(base_logits, tokenizer, gold)
            advisor_states = None
            token_ids: list[int] = []
            if cell in {"B", "C", "D"}:
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
            delta, stats = _compute_delta_for_cell(cell, bridge, token_baseline, ctx["scratchpad_hidden"], advisor_states, args)
            out = _forward_with_optional_injection(
                model,
                ctx["prefix_ids"],
                int(args.layer_index),
                ctx["scratchpad_mask"],
                delta,
                output_hidden_states=False,
                output_attentions=False,
            )
            on_gold = _candidate_log_prob(out.logits, tokenizer, gold)
            on_foil = _candidate_log_prob(out.logits, tokenizer, foil)

        pred_idx = gi if float(on_gold.item()) >= float(on_foil.item()) else fi
        pred = str(candidates[pred_idx])
        correct = _answer_match(gold, pred)
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR")) if token_ids else {"scope_total": 0.0}
        advisor = _advisor_stats(token_ids) if token_ids else {
            "active_token_count": 0.0,
            "active_op_count": 0.0,
            "operator_entropy": 0.0,
            "operator_top1_share": 0.0,
        }
        answer_delta = float((on_gold - on_foil).item())
        intervention = float((on_gold - base_gold).item())
        guard_threshold = _cell_guard_threshold(cell, args)

        deltas.append(intervention)
        answer_delta_hist.append(answer_delta)
        scope_hist.append(float(scope.get("scope_total", 0.0)))
        residual_norm_hist.append(float(stats["residual_norm"]))
        gate_hist.append(float(stats["gate"]))
        attn_entropy_hist.append(float(stats["attn_entropy"]))
        attn_mass_hist.append(float(ctx["attention_mass"]))
        overflow_hist.append(float(stats["residual_norm"] > guard_threshold) if cell in {"B", "C", "D"} else 0.0)

        rows.append(
            {
                "item_id": item.get("item_id", ""),
                "pair_id": item.get("pair_id", ""),
                "family": item.get("family", "other"),
                "bucket": item.get("bucket", ""),
                "prompt": prompt,
                "candidates": candidates,
                "gold_answer": gold,
                "model_answer": pred,
                "correct": bool(correct),
                "active_token_count": float(advisor["active_token_count"]),
                "active_op_count": float(advisor["active_op_count"]),
                "operator_entropy": float(advisor["operator_entropy"]),
                "operator_top1_share": float(advisor["operator_top1_share"]),
                "scope": float(scope.get("scope_total", 0.0)),
                "score_on_gold": float(on_gold.item()),
                "score_on_foil": float(on_foil.item()),
                "score_off_gold": float(base_gold.item()),
                "answer_delta": float(answer_delta),
                "gold_delta_on_off": float(intervention),
                "scratchpad_attention_mass": float(ctx["attention_mass"]),
                "scratchpad_residual_norm": float(stats["residual_norm"]),
                "scratchpad_gate_mean": float(stats["gate"]),
                "scratchpad_gate_overflow": float(stats["residual_norm"] > guard_threshold) if cell in {"B", "C", "D"} else 0.0,
                "scratchpad_attn_entropy": float(stats["attn_entropy"]),
                "cell_mode": {
                    "A": "scratchpad_control",
                    "B": "strict_residual_scratchpad",
                    "C": "relaxed_residual_scratchpad",
                    "D": "severance_threshold_residual_scratchpad",
                    "E": "token_only_scratchpad",
                }[cell],
            }
        )

    n = max(1, len(rows))
    fam_adj = [r for r in rows if str(r["family"]) == "adjective_property"]
    fam_cau = [r for r in rows if str(r["family"]) == "causal_direction"]
    metrics = {
        "overall_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
        "adjective_accuracy": float(sum(1 for r in fam_adj if r["correct"]) / max(1, len(fam_adj))),
        "causal_accuracy": float(sum(1 for r in fam_cau if r["correct"]) / max(1, len(fam_cau))),
        "mean_answer_delta": float(sum(answer_delta_hist) / max(1, len(answer_delta_hist))),
        "mean_scope": float(sum(scope_hist) / max(1, len(scope_hist))),
        "mean_intervention_delta_gold": float(sum(deltas) / max(1, len(deltas))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_active_op_count": float(sum(float(r["active_op_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_top1_op_share": float(sum(float(r["operator_top1_share"]) for r in rows) / n),
        "scratchpad_attention_mass": float(sum(attn_mass_hist) / max(1, len(attn_mass_hist))),
        "scratchpad_residual_norm": float(sum(residual_norm_hist) / max(1, len(residual_norm_hist))),
        "scratchpad_gate_mean": float(sum(gate_hist) / max(1, len(gate_hist))),
        "scratchpad_gate_overflow_rate": float(sum(overflow_hist) / max(1, len(overflow_hist))),
        "mean_return_attn_entropy": float(sum(attn_entropy_hist) / max(1, len(attn_entropy_hist))),
        "first_token_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
    }
    return metrics, rows


def _generate_short_resumption(
    cell: str,
    bridge: SymbioteScratchpadBridge | None,
    token_baseline: SymbioteTokenBaseline | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    symbiote_token_id: int,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = str(item["prompt"])
    with torch.no_grad():
        ctx = _build_scratchpad_context(
            model,
            tokenizer,
            prompt,
            int(args.layer_index),
            int(symbiote_token_id),
            str(args.scratchpad_token),
            int(args.scratchpad_length),
            output_attentions=True,
        )
        advisor_states = None
        if cell in {"B", "C", "D"}:
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
        delta, stats = _compute_delta_for_cell(cell, bridge, token_baseline, ctx["scratchpad_hidden"], advisor_states, args)

        cur_ids = ctx["prefix_ids"]
        generated_ids: list[int] = []
        first_attention_mass = float(ctx["attention_mass"])
        max_new_tokens = max(1, int(args.continuation_eval_tokens))
        for _ in range(max_new_tokens):
            cur_mask = cur_ids.eq(int(symbiote_token_id))
            out = _forward_with_optional_injection(
                model,
                cur_ids,
                int(args.layer_index),
                cur_mask,
                delta,
                output_hidden_states=False,
                output_attentions=False,
            )
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_tok.item()))
            cur_ids = torch.cat([cur_ids, next_tok.to(device=cur_ids.device, dtype=cur_ids.dtype)], dim=1)
            if tokenizer.eos_token_id is not None and int(next_tok.item()) == int(tokenizer.eos_token_id):
                break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    target_text = _resume_target_text(item)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids[: max(1, int(args.continuation_eval_tokens))]
    target_first = int(target_ids[0]) if target_ids else None
    first_match = bool(target_first is not None and generated_ids and int(generated_ids[0]) == int(target_first))
    contaminated = _contains_lojban_bleed(generated_text)
    scratchpad_bleed = any(int(tok_id) == int(symbiote_token_id) for tok_id in generated_ids) or str(args.scratchpad_token) in generated_text
    looped = _loop_flag(generated_ids)
    target_norm = _normalize_text(target_text)
    pred_norm = _normalize_text(generated_text.replace(str(args.scratchpad_token), " "))
    gold_mentioned = bool(target_norm) and target_norm in pred_norm
    exact_match = bool(target_norm) and pred_norm == target_norm
    fluency = _english_fluency_score(generated_text, contaminated or scratchpad_bleed, looped)
    return {
        "item_id": item.get("item_id", ""),
        "pair_id": item.get("pair_id", ""),
        "prompt": prompt,
        "target_text": target_text,
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "resume_first_token_accuracy": float(1.0 if first_match else 0.0),
        "continuation_overlap_f1": float(_continuation_overlap_f1(generated_text, target_text)),
        "gold_mention_rate": float(1.0 if gold_mentioned else 0.0),
        "exact_match_rate": float(1.0 if exact_match else 0.0),
        "contamination_rate": float(1.0 if contaminated else 0.0),
        "loop_rate": float(1.0 if looped else 0.0),
        "scratchpad_bleed_rate": float(1.0 if scratchpad_bleed else 0.0),
        "english_stays_english_rate": float(1.0 if (not contaminated and not scratchpad_bleed and fluency >= 0.65) else 0.0),
        "english_fluency_score": float(fluency),
        "scratchpad_attention_mass": float(first_attention_mass),
        "scratchpad_residual_norm": float(stats["residual_norm"]),
        "scratchpad_gate_mean": float(stats["gate"]),
    }


def _evaluate_continuation_cell(
    cell: str,
    bridge: SymbioteScratchpadBridge | None,
    token_baseline: SymbioteTokenBaseline | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    symbiote_token_id: int,
    eval_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = [
        _generate_short_resumption(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, item, args)
        for item in eval_pack
    ]
    n = max(1, len(rows))
    metrics = {
        "english_fluency_score": float(sum(float(r["english_fluency_score"]) for r in rows) / n),
        "contamination_rate": float(sum(float(r["contamination_rate"]) for r in rows) / n),
        "loop_rate": float(sum(float(r["loop_rate"]) for r in rows) / n),
        "continuation_overlap_f1": float(sum(float(r["continuation_overlap_f1"]) for r in rows) / n),
        "gold_mention_rate": float(sum(float(r["gold_mention_rate"]) for r in rows) / n),
        "exact_match_rate": float(sum(float(r["exact_match_rate"]) for r in rows) / n),
        "english_stays_english_rate": float(sum(float(r["english_stays_english_rate"]) for r in rows) / n),
        "resume_first_token_accuracy": float(sum(float(r["resume_first_token_accuracy"]) for r in rows) / n),
        "scratchpad_bleed_rate": float(sum(float(r["scratchpad_bleed_rate"]) for r in rows) / n),
    }
    return metrics, rows


def _promotion_gates(report: dict[str, Any], min_intervention: float = 0.01) -> dict[str, Any]:
    a = report["cells"]["A"]["metrics"]
    gates: dict[str, Any] = {}
    for cell in ("B", "C", "D", "E"):
        m = report["cells"][cell]["metrics"]
        seed2 = report["cells"].get(f"{cell}_seed2_eval", {}).get("metrics", {})
        gates[cell] = {
            "positive_intervention_delta": float(m.get("mean_intervention_delta_gold", 0.0)) >= float(min_intervention),
            "resume_first_token_gain": float(m.get("resume_first_token_accuracy", 0.0)) >= float(a.get("resume_first_token_accuracy", 0.0)),
            "fluency_preserved": float(m.get("english_fluency_score", 0.0)) >= float(a.get("english_fluency_score", 0.0)) - 0.05,
            "contamination_below_threshold": float(m.get("contamination_rate", 1.0)) <= float(a.get("contamination_rate", 0.0)) + 0.05,
            "loop_rate_below_threshold": float(m.get("loop_rate", 1.0)) <= float(a.get("loop_rate", 0.0)) + 0.05,
            "scratchpad_bleed_below_threshold": float(m.get("scratchpad_bleed_rate", 1.0)) <= float(a.get("scratchpad_bleed_rate", 0.0)) + 0.05,
            "seed_stability": abs(float(seed2.get("overall_accuracy", m.get("overall_accuracy", 0.0))) - float(m.get("overall_accuracy", 0.0))) <= 0.05,
        }
        gates[cell]["promote_to_next"] = all(bool(v) for v in gates[cell].values())
    return gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M14 Symbiote Scratchpad: bounded scratchpad-token residual injection before English continuation.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--eval-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--return-norm-weight", type=float, default=0.01)
    p.add_argument("--continuation-target-max-tokens", type=int, default=5)
    p.add_argument("--continuation-eval-tokens", type=int, default=5)
    p.add_argument("--scratchpad-token", type=str, default="<symbiote>")
    p.add_argument("--scratchpad-length", type=int, default=4)
    p.add_argument("--scratchpad-alpha", type=float, default=1.0)
    p.add_argument("--b-guard-threshold", type=float, default=0.01)
    p.add_argument("--c-guard-threshold", type=float, default=0.05)
    p.add_argument("--d-guard-threshold", type=float, default=0.10)
    p.add_argument("--residual-guard-weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, default=Path(SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["baseline_manifest"]))
    p.add_argument("--upstream-m3-18-report", type=Path, default=None)
    p.add_argument("--upstream-m3-19-report", type=Path, default=None)
    p.add_argument("--upstream-m11-manifest", type=Path, default=None)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path(SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["output_root"]))
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

    upstream_m318 = args.upstream_m3_18_report
    if upstream_m318 is None:
        upstream_m318 = _latest_named_file(Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume"), "m3_18_report.json")
    upstream_m319 = args.upstream_m3_19_report
    if upstream_m319 is None:
        upstream_m319 = _latest_named_file(Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid"), "m3_19_grid_report.json")
    upstream_m11 = args.upstream_m11_manifest
    if upstream_m11 is None:
        suite_root = Path(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["search_root"])
        upstream_m11 = _latest_named_file(suite_root, str(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["manifest_name"]))
        if upstream_m11 is None:
            fallback = Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json")
            upstream_m11 = fallback if fallback.exists() else None

    adapter_path = Path(args.adapter)
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    tok_src = str(adapter_path) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Surgical Load using original Qwen config
    from transformers import AutoConfig
    base_qwen = "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct"
    config = AutoConfig.from_pretrained(base_qwen, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Force alignment with archival weight dimensions [153921, 896]
    backbone.resize_token_embeddings(153921)
    
    # Load synced weights from archival path
    weights_path = Path(args.base_model) / "adapter_model.safetensors"
    if not weights_path.exists():
        # Try finding standard weights in the dir
        weights_path = Path(args.base_model) / "model.safetensors"
    
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
        # Handle PEFT key prefixes if necessary
        clean_sd = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_sd, strict=False)
    
    print(f"M14: Backbone realigned to original Qwen config and archival weights.")
    
    # Load adapter with strict=False to ignore embedding size mismatch
    model = PeftModel.from_pretrained(backbone, str(adapter_path), local_files_only=args.local_files_only, is_trainable=True)
    # The embeddings are already resized in the backbone, so we just need to ensure the PeftModel respects it.
    symbiote_token_id = _ensure_symbiote_token(model, tokenizer, str(args.scratchpad_token))
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
        pack_size = max(int(args.eval_size) * 8, 64, int(args.train_steps) * 4)
        if bool(args.strict_balance):
            pack_size = max(8, ((pack_size + 7) // 8) * 8)
        base_pack = _round_robin_rich_pack(pack_size, int(args.seed), bool(args.strict_balance))
        pack_path = run_dir / "m14_rich_resumption_pack.jsonl"
        _write_jsonl(pack_path, base_pack)
        _pack_summary(base_pack, run_dir / "m14_rich_resumption_pack.summary.json")
    else:
        pack_path = args.pack_jsonl
        base_pack = _load_pack(args.pack_jsonl, size=max(int(args.eval_size) * 8, 64), seed=int(args.seed), strict_balance=bool(args.strict_balance))

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
    pack_train = [r for r in base_pack if str(r.get("pair_id", "")) in train_ids][: max(1, int(args.train_steps))]
    pack_val = [r for r in base_pack if str(r.get("pair_id", "")) in val_ids][: int(args.eval_size)]
    pack_eval = [r for r in base_pack if str(r.get("pair_id", "")) in eval_ids][: int(args.eval_size)]
    pack_eval_seed2 = _round_robin_rich_pack(max(8, int(args.eval_size) * 2), int(args.seed) + 2, bool(args.strict_balance))[: int(args.eval_size)]
    split_meta = {"train_pair_ids": sorted(train_ids), "val_pair_ids": sorted(val_ids), "eval_pair_ids": sorted(eval_ids)}
    split_meta_seed2 = {"seed": int(args.seed) + 1}

    (run_dir / "m14_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")

    report: dict[str, Any] = build_scratchpad_protocol_manifest(
        track="M14",
        run_id=run_id,
        baseline_manifest_path=args.baseline_manifest,
        baseline_id=str(baseline_manifest.get("baseline_id", "")),
        upstream_m318_report=str(upstream_m318).replace("\\", "/") if upstream_m318 else None,
        upstream_m319_report=str(upstream_m319).replace("\\", "/") if upstream_m319 else None,
        upstream_m11_manifest=str(upstream_m11).replace("\\", "/") if upstream_m11 else None,
        config={k: str(v) for k, v in vars(args).items()},
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["series"] = series_metadata("M", "M14", "scripts/run_m14_symbiote_scratchpad.py")
    report["lineage"] = lineage_metadata(
        "train",
        checkpoint_in=str(args.checkpoint).replace("\\", "/"),
        checkpoint_out=None,
        dataset_profile="m3_19_rich_resumption_split",
        difficulty_tier="mixed",
    )
    report["data_split"] = {**split_meta, "seed2_eval_meta": split_meta_seed2, "runtime_policy_source": "validation_metrics", "final_metrics_source": "eval_pack"}
    report["pack_path"] = str(pack_path).replace("\\", "/")
    report["comparison_references"] = _comparison_snapshot(upstream_m318, upstream_m319, upstream_m11)
    report["cells"] = {}

    def make_bridge() -> SymbioteScratchpadBridge:
        mod = SymbioteScratchpadBridge(hidden, bottleneck_dim=int(args.bottleneck_dim)).to(target_device, dtype=module_dtype)
        mod.eval()
        return mod

    def make_token_baseline() -> SymbioteTokenBaseline:
        mod = SymbioteTokenBaseline(hidden, scratchpad_length=int(args.scratchpad_length)).to(target_device, dtype=module_dtype)
        mod.eval()
        return mod

    train_a = _train_cell("A", None, None, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_train, args)
    met_a_val, _ = _evaluate_cell("A", None, None, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_val, args)
    met_a, rows_a = _evaluate_cell("A", None, None, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_eval, args)
    cont_a, cont_rows_a = _evaluate_continuation_cell("A", None, None, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_eval, args)
    met_a.update(cont_a)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val, "variant_spec": SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["cells"]["A"]}
    (run_dir / "m14_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")
    (run_dir / "m14_A_continuation_eval.json").write_text(json.dumps(cont_rows_a, indent=2), encoding="utf-8")

    for cell in ("B", "C", "D", "E"):
        bridge = make_bridge() if cell in {"B", "C", "D"} else None
        token_baseline = make_token_baseline() if cell == "E" else None
        train_cell = _train_cell(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_train, args)
        met_val, _ = _evaluate_cell(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_val, args)
        met_eval, rows_eval = _evaluate_cell(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_eval, args)
        cont_eval, cont_rows_eval = _evaluate_continuation_cell(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_eval, args)
        met_eval.update(cont_eval)
        met_seed2, rows_seed2 = _evaluate_cell(cell, bridge, token_baseline, model, tokenizer, codebook, arity_head, symbiote_token_id, pack_eval_seed2, args)
        report["cells"][cell] = {
            "train": train_cell,
            "metrics": met_eval,
            "validation_metrics": met_val,
            "variant_spec": {
                **SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["cells"][cell],
                "scratchpad_length": int(args.scratchpad_length),
                "layer_index": int(args.layer_index),
                "scratchpad_alpha": float(args.scratchpad_alpha),
                "scratchpad_token": str(args.scratchpad_token),
            },
        }
        report["cells"][f"{cell}_seed2_eval"] = {"metrics": met_seed2}
        (run_dir / f"m14_{cell}_eval.json").write_text(json.dumps(rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m14_{cell}_continuation_eval.json").write_text(json.dumps(cont_rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m14_{cell}_seed2_eval.json").write_text(json.dumps(rows_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(report, min_intervention=0.01)
    (run_dir / "m14_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = scratchpad_cell_labels("M14")
    md = [
        "# M14 Symbiote Scratchpad",
        "",
        "| Cell | Regime | Acc | FTok | Fluency | Contam | S-Bleed | Loop | Answer Delta | Gold On-Off | S-Attn | S-Norm | Scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("A", "B", "C", "D", "E"):
        m = report["cells"][c]["metrics"]
        md.append(
            f"| {c} | {labels[c]} | {m.get('overall_accuracy', 0.0):.3f} | {m.get('resume_first_token_accuracy', 0.0):.3f} | "
            f"{m.get('english_fluency_score', 0.0):.3f} | {m.get('contamination_rate', 0.0):.3f} | "
            f"{m.get('scratchpad_bleed_rate', 0.0):.3f} | {m.get('loop_rate', 0.0):.3f} | "
            f"{m.get('mean_answer_delta', 0.0):.4f} | {m.get('mean_intervention_delta_gold', 0.0):.4f} | "
            f"{m.get('scratchpad_attention_mass', 0.0):.4f} | {m.get('scratchpad_residual_norm', 0.0):.4f} | {m.get('mean_scope', 0.0):.4f} |"
        )
    md.extend(["", "## Comparison References"])
    for label, snapshot in report["comparison_references"].items():
        md.append(f"- {label}: `{json.dumps(snapshot, ensure_ascii=True)}`")
    md.extend(["", "## Promotion Gates"])
    for cell in ("B", "C", "D", "E"):
        md.append(f"- {cell}:")
        for key, value in report["promotion_gates"][cell].items():
            md.append(f"  - {key}: `{value}`")
    (run_dir / "m14_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M14 complete: {run_dir}")


if __name__ == "__main__":
    main()
