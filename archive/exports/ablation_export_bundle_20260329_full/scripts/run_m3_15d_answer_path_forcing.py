from __future__ import annotations

import argparse
import json
import math
import random
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

from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from lojban_evolution.m_bridge_ablation_family import build_bridge_report, finalize_bridge_report, track_cell_labels
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
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
CUE_TERMS = (
    "because",
    "since",
    "after",
    "before",
    "too big",
    "too small",
    "too heavy",
    "too weak",
    "very fast",
    "very slow",
    "helped",
    "needed help",
    "unauthorized",
    "confidential",
    "feared unrest",
    "supported revolt",
    "reviewed affidavit",
    "contradicted testimony",
)


_WINOGRAD_PAIR_BANK: list[dict[str, Any]] = [
    {
        "pair_id": "size_01",
        "family": "adjective_property",
        "variants": [
            ("Stone did not fit in Box because it was too big. What was too big?", ["Stone", "Box"], 0, "positive"),
            ("Stone did not fit in Box because it was too small. What was too small?", ["Stone", "Box"], 1, "negative"),
        ],
    },
    {
        "pair_id": "size_02",
        "family": "adjective_property",
        "variants": [
            ("Crane could not lift Beam because it was too heavy. What was too heavy?", ["Crane", "Beam"], 1, "positive"),
            ("Crane could not lift Beam because it was too weak. What was too weak?", ["Crane", "Beam"], 0, "negative"),
        ],
    },
    {
        "pair_id": "speed_01",
        "family": "causal_direction",
        "variants": [
            ("Truck passed Bus because it was very fast. What was fast?", ["Truck", "Bus"], 0, "positive"),
            ("Truck passed Bus because it was very slow. What was slow?", ["Truck", "Bus"], 1, "negative"),
        ],
    },
    {
        "pair_id": "social_01",
        "family": "causal_direction",
        "variants": [
            ("Alex thanked Riley because he helped. Who helped?", ["Alex", "Riley"], 1, "positive"),
            ("Alex thanked Riley because he needed help. Who needed help?", ["Alex", "Riley"], 0, "negative"),
        ],
    },
    {
        "pair_id": "social_02",
        "family": "causal_direction",
        "variants": [
            ("Mayor refused Protesters permit because they feared unrest. Who feared unrest?", ["Mayor", "Protesters"], 0, "positive"),
            ("Mayor refused Protesters permit because they supported revolt. Who supported revolt?", ["Mayor", "Protesters"], 1, "negative"),
        ],
    },
    {
        "pair_id": "role_01",
        "family": "causal_direction",
        "variants": [
            ("Lawyer questioned Witness after he reviewed affidavit. Who reviewed affidavit?", ["Lawyer", "Witness"], 0, "positive"),
            ("Lawyer questioned Witness after he contradicted testimony. Who contradicted testimony?", ["Lawyer", "Witness"], 1, "negative"),
        ],
    },
    {
        "pair_id": "policy_01",
        "family": "causal_direction",
        "variants": [
            ("Morgan did not forward Memo to Casey because he was unauthorized. Who was unauthorized?", ["Morgan", "Casey"], 1, "positive"),
            ("Morgan did not forward Memo to Casey because it was confidential. What was confidential?", ["Memo", "Casey"], 0, "negative"),
        ],
    },
]


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", str(text).strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and p == e


def _masked_logits(logits: torch.Tensor, legal_start: int, legal_end: int) -> torch.Tensor:
    out = logits.clone()
    if legal_start > 0:
        out[:, :legal_start] = -1e9
    if legal_end < out.shape[-1]:
        out[:, legal_end:] = -1e9
    return out


def _triples(token_ids: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for i in range(0, len(token_ids) - 2, 3):
        out.append((int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2])))
    return out


def _decode_logic_tokens(
    arity_head: AdvisorArityHead,
    z_st: torch.Tensor,
    relation_vocab: int,
    var_min_id: int,
    relation_bias: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    tokens: list[torch.Tensor] = []
    for i in range(z_st.shape[1]):
        z = z_st[:, i, :]
        l_rel = arity_head.head_rel(z)
        l_rel = _masked_logits(l_rel, 0, int(relation_vocab))
        if relation_bias is not None:
            rb = relation_bias.view(1, -1)
            l_rel[:, : int(relation_vocab)] = l_rel[:, : int(relation_vocab)] + rb[:, : int(relation_vocab)]

        l_v1_raw = arity_head.head_var1(z)
        l_v2_raw = arity_head.head_var2(z)
        l_v1 = _masked_logits(l_v1_raw, int(var_min_id), l_v1_raw.shape[-1])
        l_v2 = _masked_logits(l_v2_raw, int(var_min_id), l_v2_raw.shape[-1])

        tokens.extend(
            [
                torch.argmax(l_rel, dim=-1),
                torch.argmax(l_v1, dim=-1),
                torch.argmax(l_v2, dim=-1),
            ]
        )
    return tokens


def _masked_relation_logits(
    arity_head: AdvisorArityHead,
    z: torch.Tensor,
    relation_vocab: int,
    relation_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = _masked_logits(arity_head.head_rel(z), 0, int(relation_vocab))[:, : int(relation_vocab)]
    if relation_bias is not None:
        rb = relation_bias.view(1, -1)[:, : int(relation_vocab)]
        logits = logits + rb
    return logits


def _mean_relation_probs(
    arity_head: AdvisorArityHead,
    z_st: torch.Tensor,
    relation_vocab: int,
    relation_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    probs: list[torch.Tensor] = []
    for i in range(z_st.shape[1]):
        logits = _masked_relation_logits(arity_head, z_st[:, i, :], int(relation_vocab), relation_bias=relation_bias)
        probs.append(torch.softmax(logits, dim=-1))
    return torch.stack(probs, dim=0).mean(dim=(0, 1))


def _distribution_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-8)
    return -torch.sum(p * torch.log(p))


def _scaled_relation_bias(raw_bias: torch.Tensor, scale: float) -> torch.Tensor:
    centered = raw_bias - torch.mean(raw_bias)
    return torch.tanh(centered) * float(scale)


def _anti_collapse_loss(
    base_probs: torch.Tensor,
    on_probs: torch.Tensor,
    *,
    entropy_floor_ratio: float,
    top1_margin: float,
    top1_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    base_entropy = _distribution_entropy(base_probs)
    on_entropy = _distribution_entropy(on_probs)
    entropy_floor_loss = torch.relu((float(entropy_floor_ratio) * base_entropy) - on_entropy)
    base_top1 = torch.max(base_probs)
    on_top1 = torch.max(on_probs)
    top1_loss = torch.relu(on_top1 - (base_top1 + float(top1_margin)))
    drift_kl = torch.sum(on_probs.clamp(min=1e-8) * (torch.log(on_probs.clamp(min=1e-8)) - torch.log(base_probs.clamp(min=1e-8))))
    loss = entropy_floor_loss + (float(top1_weight) * top1_loss) + (float(kl_weight) * drift_kl)
    stats = {
        "base_entropy": float(base_entropy.detach().item()),
        "on_entropy": float(on_entropy.detach().item()),
        "base_top1_share": float(base_top1.detach().item()),
        "on_top1_share": float(on_top1.detach().item()),
        "entropy_floor_loss": float(entropy_floor_loss.detach().item()),
        "top1_loss": float(top1_loss.detach().item()),
        "drift_kl": float(drift_kl.detach().item()),
    }
    return loss, stats


def _candidate_first_token_id(tokenizer, candidate: str) -> int:
    ids = tokenizer(" " + str(candidate), add_special_tokens=False, return_tensors="pt").input_ids
    if int(ids.numel()) == 0:
        return int(tokenizer.eos_token_id or 0)
    return int(ids[0, 0].item())


def _find_char_span(text: str, needle: str) -> tuple[int, int] | None:
    hay = str(text)
    nd = str(needle).strip()
    if not nd:
        return None
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(nd)}(?![A-Za-z0-9])", re.IGNORECASE)
    match = pattern.search(hay)
    if match is not None:
        return int(match.start()), int(match.end())
    idx = hay.lower().find(nd.lower())
    if idx >= 0:
        return int(idx), int(idx + len(nd))
    return None


def _span_mean(hidden: torch.Tensor, offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> torch.Tensor | None:
    if span is None:
        return None
    start, end = int(span[0]), int(span[1])
    idxs = [i for i, (s, e) in enumerate(offsets) if e > s and not (e <= start or s >= end)]
    if not idxs:
        return None
    return hidden[:, idxs, :].mean(dim=1)


def _extract_relation_local_states(model, tokenizer, prompt: str, candidates: list[str], layer_index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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
    eng_hidden = out.hidden_states[int(layer_index)].detach()

    pair_nodes: list[torch.Tensor] = []
    node_labels: list[str] = []
    spans: dict[str, list[int] | None] = {}
    for idx, candidate in enumerate(candidates[:2]):
        span = _find_char_span(prefix, candidate)
        spans[f"candidate_{idx}"] = list(span) if span is not None else None
        state = _span_mean(eng_hidden, offsets, span)
        if state is not None:
            pair_nodes.append(state)
            node_labels.append(f"candidate_{idx}")

    cue_state = None
    cue_label = ""
    for term in CUE_TERMS:
        span = _find_char_span(prefix, term)
        if span is None:
            continue
        state = _span_mean(eng_hidden, offsets, span)
        if state is None:
            continue
        cue_state = state
        cue_label = term
        spans["cue"] = list(span)
        break

    if cue_state is not None and len(pair_nodes) >= 2:
        local_nodes = [pair_nodes[0], cue_state, pair_nodes[1]]
        local_labels = [node_labels[0], f"cue:{cue_label}", node_labels[1]]
    elif pair_nodes:
        local_nodes = pair_nodes
        local_labels = node_labels
    else:
        valid = [i for i, (s, e) in enumerate(offsets) if e > s]
        fallback = valid[-2:] if len(valid) >= 2 else valid[-1:]
        local_nodes = [eng_hidden[:, i : i + 1, :].mean(dim=1) for i in fallback]
        local_labels = [f"fallback_{i}" for i in fallback]

    local_hidden = torch.stack(local_nodes, dim=1)
    meta = {
        "prefix": prefix,
        "node_labels": local_labels,
        "spans": spans,
    }
    return eng_hidden, local_hidden, meta


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _score_candidate_first_token(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    advisor_states: torch.Tensor,
    advisor_ids: torch.Tensor,
    layer_index: int,
) -> float:
    dev = _model_device(model)
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(dev)
    cur_emb = model.get_input_embeddings()(p_ids)
    advisor_states = advisor_states.to(device=dev, dtype=cur_emb.dtype)
    advisor_ids = advisor_ids.to(device=dev, dtype=torch.long)
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    ptr_ids = torch.zeros((1, cur_emb.shape[1]), dtype=torch.long, device=dev)
    with persistent_advisor_hook(model, int(layer_index), None, advisor_states, advisor_ids, ptr_ids, 1.0):
        out = model(inputs_embeds=cur_emb, return_dict=True, use_cache=False)
    logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]
    return float(logp.item())


def _score_candidate_first_token_with_adapter(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    advisor_adapter: CouncilCrossAttentionAdapter,
    advisor_states: torch.Tensor,
    advisor_ids: torch.Tensor,
    layer_index: int,
) -> float:
    dev = _model_device(model)
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(dev)
    cur_emb = model.get_input_embeddings()(p_ids)
    advisor_states = advisor_states.to(device=dev, dtype=cur_emb.dtype)
    advisor_ids = advisor_ids.to(device=dev, dtype=torch.long)
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    ptr_ids = torch.zeros((1, cur_emb.shape[1]), dtype=torch.long, device=dev)
    with persistent_advisor_hook(model, int(layer_index), advisor_adapter, advisor_states, advisor_ids, ptr_ids, 1.0):
        out = model(inputs_embeds=cur_emb, return_dict=True, use_cache=False)
    logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]
    return float(logp.item())


def _build_attention_mask_for_final_prefix(tokenizer, prompt: str, dev: torch.device, blindfold_question: bool) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = build_final_prefix(prompt)
    try:
        enc = tokenizer(prefix, return_tensors="pt", return_offsets_mapping=True)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    except NotImplementedError:
        enc = tokenizer(prefix, return_tensors="pt")
        offsets = [(0, 0) for _ in range(int(enc["input_ids"].shape[1]))]
    p_ids = enc["input_ids"].to(dev)
    am = torch.ones_like(p_ids, device=dev)
    if not blindfold_question:
        return p_ids, am
    q_span = _find_char_span(prefix, prompt)
    if q_span is None:
        return p_ids, am
    q_start, q_end = q_span
    masked_any = False
    for i, (s, e) in enumerate(offsets):
        if e > s and not (e <= q_start or s >= q_end):
            am[0, i] = 0
            masked_any = True
    if masked_any:
        am[0, -1] = 1
    return p_ids, am


def _score_candidate_first_token_with_adapter_tensor(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    advisor_adapter: CouncilCrossAttentionAdapter,
    advisor_states: torch.Tensor,
    advisor_ids: torch.Tensor,
    layer_index: int,
    *,
    blindfold_question: bool = False,
) -> torch.Tensor:
    dev = _model_device(model)
    p_ids, am = _build_attention_mask_for_final_prefix(tokenizer, prompt, dev, blindfold_question=blindfold_question)
    cur_emb = model.get_input_embeddings()(p_ids)
    advisor_states = advisor_states.to(device=dev, dtype=cur_emb.dtype)
    advisor_ids = advisor_ids.to(device=dev, dtype=torch.long)
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    ptr_ids = torch.zeros((1, cur_emb.shape[1]), dtype=torch.long, device=dev)
    with persistent_advisor_hook(model, int(layer_index), advisor_adapter, advisor_states, advisor_ids, ptr_ids, 1.0):
        out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
    return torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _to_complex_phase(x: torch.Tensor) -> torch.Tensor:
    d2 = (int(x.shape[-1]) // 2) * 2
    x = x[..., :d2]
    r = x[..., : d2 // 2]
    i = x[..., d2 // 2 :]
    return torch.atan2(i, r + 1e-8)


class RelationLocalRotaryBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, relation_vocab: int, max_nodes: int = 12) -> None:
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.relation_vocab = int(relation_vocab)
        self.eng_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.adv_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_head = torch.nn.Linear(hidden_size, relation_vocab, bias=False)
        self.cue_head = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.cue_scale = torch.nn.Parameter(torch.tensor(0.1))

    def _match_bias_input_dim(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = int(self.bias_head.in_features)
        cur_dim = int(x.shape[-1])
        if cur_dim == in_dim:
            return x
        if cur_dim > in_dim:
            return x[..., :in_dim]
        return F.pad(x, (0, in_dim - cur_dim))

    def _phase_alignment(self, local_nodes: torch.Tensor, adv_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(int(local_nodes.shape[1]), int(adv_nodes.shape[1]))
        if k < 2:
            z = torch.zeros((), device=local_nodes.device, dtype=local_nodes.dtype)
            return z, z
        e = self.eng_proj(local_nodes[:, :k, :])
        a = self.adv_proj(adv_nodes[:, :k, :])
        pe = _to_complex_phase(e)
        pa = _to_complex_phase(a)
        de = pe[:, 1:, :] - pe[:, :-1, :]
        da = pa[:, 1:, :] - pa[:, :-1, :]
        d = torch.atan2(torch.sin(de - da), torch.cos(de - da))
        loss = torch.mean(d * d)
        sim = F.cosine_similarity(de.reshape(1, -1), da.reshape(1, -1), dim=-1)[0]
        return loss, sim

    def relation_bias(self, local_nodes: torch.Tensor, adv_rel_nodes: torch.Tensor) -> torch.Tensor:
        en = local_nodes
        an = adv_rel_nodes
        k = min(int(en.shape[1]), int(an.shape[1]))
        if k <= 0:
            return torch.zeros((self.relation_vocab,), device=local_nodes.device, dtype=local_nodes.dtype)
        e = self.eng_proj(en[:, :k, :])
        a = self.adv_proj(an[:, :k, :])
        fe = torch.mean(torch.cos(_to_complex_phase(e)), dim=1)
        fa = torch.mean(torch.cos(_to_complex_phase(a)), dim=1)
        fused = torch.tanh(fe - fa)
        fused = self._match_bias_input_dim(fused)
        return self.bias_head(fused)[0]

    def runtime_cue(self, local_nodes: torch.Tensor, cue_norm_cap: float) -> torch.Tensor:
        pooled = torch.mean(self.eng_proj(local_nodes), dim=1, keepdim=True)
        cue = torch.tanh(self.cue_head(pooled)) * torch.sigmoid(self.cue_scale)
        cap = float(cue_norm_cap)
        if cap > 0.0:
            n = torch.norm(cue, dim=-1, keepdim=True).clamp(min=1e-8)
            cue = cue * torch.clamp(cap / n, max=1.0)
        return cue

    def alignment_metrics(self, local_nodes: torch.Tensor, adv_rel_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._phase_alignment(local_nodes, adv_rel_nodes)


class CandidatePointerHead(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.cand_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.rel_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, candidate_states: torch.Tensor, relation_nodes: torch.Tensor) -> torch.Tensor:
        cand = candidate_states[:, :2, :]
        if int(cand.shape[1]) == 0:
            cand = torch.zeros((relation_nodes.shape[0], 2, relation_nodes.shape[-1]), device=relation_nodes.device, dtype=relation_nodes.dtype)
        elif int(cand.shape[1]) == 1:
            cand = torch.cat([cand, cand], dim=1)
        rel_summary = relation_nodes.mean(dim=1)
        fused = torch.tanh(self.cand_proj(cand) + self.rel_proj(rel_summary).unsqueeze(1))
        return self.out_proj(fused).squeeze(-1)


def _candidate_states_from_local(local_hidden: torch.Tensor, local_meta: dict[str, Any]) -> torch.Tensor:
    labels = list(local_meta.get("node_labels", []))
    idxs = [i for i, label in enumerate(labels) if str(label).startswith("candidate_")]
    if not idxs:
        return local_hidden[:, :0, :]
    return local_hidden[:, idxs, :]


def _pair_variants(pair_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for p in pair_bank:
        for vid, v in enumerate(p["variants"]):
            variants.append(
                {
                    "pair_id": p["pair_id"],
                    "variant_id": vid,
                    "family": p["family"],
                    "prompt": v[0],
                    "candidates": v[1],
                    "gold_index": int(v[2]),
                    "polarity": v[3],
                    "causal_direction": p["family"] == "causal_direction",
                }
            )
    return variants


def _materialize_pack(variants: list[dict[str, Any]], size: int, seed: int, strict_balance: bool, item_prefix: str) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    out: list[dict[str, Any]] = []
    tgt = max(2, int(size))
    c0 = 0
    c1 = 0
    i = 0
    if not variants:
        raise ValueError("Cannot build pack from empty variant set.")
    while len(out) < tgt:
        row = dict(variants[i % len(variants)])
        i += 1
        # Lexical mutation while preserving minimal structure.
        prompt = row["prompt"].replace("because", rng.choice(["because", "since"]))
        row["prompt"] = prompt
        g = int(row["gold_index"])
        if strict_balance:
            if g == 0 and c0 > c1:
                continue
            if g == 1 and c1 > c0:
                continue
        row["item_id"] = f"{item_prefix}_{len(out):06d}"
        out.append(row)
        if g == 0:
            c0 += 1
        else:
            c1 += 1
    return out


def _build_winograd_pack(size: int, seed: int, strict_balance: bool = True) -> list[dict[str, Any]]:
    return _materialize_pack(_pair_variants(_WINOGRAD_PAIR_BANK), size=size, seed=seed, strict_balance=strict_balance, item_prefix="m3_15d")


def _build_winograd_split_packs(size: int, seed: int, strict_balance: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    rng = random.Random(int(seed))
    pair_bank = list(_WINOGRAD_PAIR_BANK)
    rng.shuffle(pair_bank)
    total = len(pair_bank)
    eval_count = max(1, round(total * 0.30))
    val_count = max(1, round(total * 0.15))
    train_count = max(1, total - eval_count - val_count)
    if train_count + val_count + eval_count > total:
        train_count = max(1, total - eval_count - val_count)
    train_pairs = pair_bank[:train_count]
    val_pairs = pair_bank[train_count : train_count + val_count]
    eval_pairs = pair_bank[train_count + val_count :]
    if not eval_pairs:
        eval_pairs = val_pairs[-1:]
        val_pairs = val_pairs[:-1] or train_pairs[-1:]
    split_ids = {
        "train_pair_ids": [str(p["pair_id"]) for p in train_pairs],
        "val_pair_ids": [str(p["pair_id"]) for p in val_pairs],
        "eval_pair_ids": [str(p["pair_id"]) for p in eval_pairs],
    }
    train_pack = _materialize_pack(_pair_variants(train_pairs), size=size, seed=seed, strict_balance=strict_balance, item_prefix="m3_15d_train")
    val_pack = _materialize_pack(_pair_variants(val_pairs), size=size, seed=seed + 101, strict_balance=strict_balance, item_prefix="m3_15d_val")
    eval_pack = _materialize_pack(_pair_variants(eval_pairs), size=size, seed=seed + 202, strict_balance=strict_balance, item_prefix="m3_15d_eval")
    return train_pack, val_pack, eval_pack, split_ids


def _load_pack(path: Path | None, size: int, seed: int, strict_balance: bool) -> list[dict[str, Any]]:
    if path is None:
        return _build_winograd_pack(size=size, seed=seed, strict_balance=strict_balance)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    if len(rows) < int(size):
        ext = _build_winograd_pack(size=int(size) - len(rows), seed=seed + 9, strict_balance=strict_balance)
        rows.extend(ext)
    return rows[: int(size)]


def _train_answer_path_cell(
    cell: str,
    bridge: RelationLocalRotaryBridge,
    pointer_head: CandidatePointerHead | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    train_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    if cell == "A":
        return {
            "answer_path_loss": 0.0,
            "answer_delta": 0.0,
            "alignment_similarity": 0.0,
            "anti_collapse_loss": 0.0,
            "base_operator_entropy": 0.0,
            "on_operator_entropy": 0.0,
            "base_top1_share": 1.0,
            "on_top1_share": 1.0,
        }

    bridge.train()
    if pointer_head is not None:
        pointer_head.train()
    params = list(bridge.parameters()) + (list(pointer_head.parameters()) if pointer_head is not None else [])
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.01)
    answer_hist: list[float] = []
    delta_hist: list[float] = []
    sim_hist: list[float] = []
    anti_hist: list[float] = []
    base_entropy_hist: list[float] = []
    on_entropy_hist: list[float] = []
    base_top1_hist: list[float] = []
    on_top1_hist: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            _eng_hidden, local_hidden, local_meta = _extract_relation_local_states(
                model,
                tokenizer,
                prompt,
                candidates,
                int(args.layer_index),
            )
            h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)
            base_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
            rel_nodes = base_states[:, 0::3, :]

        opt.zero_grad()
        _align_loss, align_sim = bridge.alignment_metrics(local_hidden, rel_nodes)
        cue = bridge.runtime_cue(local_hidden, cue_norm_cap=float(args.runtime_cue_norm_cap))
        gate = torch.sigmoid(bridge.gate)
        if float(args.bridge_train_gate_cap) > 0.0:
            gate = torch.clamp(gate, max=float(args.bridge_train_gate_cap))
        z_use = z_st + gate * cue

        base_probs = _mean_relation_probs(arity_head, z_st, int(args.relation_vocab), relation_bias=None)
        on_probs = _mean_relation_probs(arity_head, z_use, int(args.relation_vocab), relation_bias=None)
        anti_collapse_loss, anti_stats = _anti_collapse_loss(
            base_probs,
            on_probs,
            entropy_floor_ratio=float(args.collapse_entropy_floor_ratio),
            top1_margin=float(args.collapse_top1_margin),
            top1_weight=float(args.collapse_top1_weight),
            kl_weight=float(args.collapse_kl_weight),
        )

        on_tokens = _decode_logic_tokens(arity_head, z_use, int(args.relation_vocab), int(args.var_min_id))
        on_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in on_tokens], dim=1)
        on_ids = torch.stack(on_tokens, dim=1)
        blindfold = cell == "C"

        if cell == "D":
            assert pointer_head is not None
            candidate_states = _candidate_states_from_local(local_hidden, local_meta)
            logits = pointer_head(candidate_states, on_states[:, 0::3, :])
            target = torch.tensor([gi], device=logits.device, dtype=torch.long)
            answer_loss = F.cross_entropy(logits, target)
            answer_delta = logits[0, gi] - logits[0, fi]
        else:
            logp_gold = _score_candidate_first_token_with_adapter_tensor(
                model, tokenizer, prompt, gold, advisor_adapter, on_states, on_ids, int(args.layer_index), blindfold_question=blindfold
            )
            logp_foil = _score_candidate_first_token_with_adapter_tensor(
                model, tokenizer, prompt, foil, advisor_adapter, on_states, on_ids, int(args.layer_index), blindfold_question=blindfold
            )
            answer_delta = logp_gold - logp_foil
            answer_loss = torch.relu(torch.tensor(float(args.margin), device=answer_delta.device, dtype=answer_delta.dtype) - answer_delta)

        reg = 0.01 * torch.mean(cue * cue)
        loss = float(args.answer_weight) * answer_loss + float(args.anti_collapse_weight) * anti_collapse_loss + reg
        loss.backward()
        opt.step()
        answer_hist.append(float(answer_loss.detach().item()))
        delta_hist.append(float(answer_delta.detach().item()))
        sim_hist.append(float(align_sim.detach().item()))
        anti_hist.append(float(anti_collapse_loss.detach().item()))
        base_entropy_hist.append(float(anti_stats["base_entropy"]))
        on_entropy_hist.append(float(anti_stats["on_entropy"]))
        base_top1_hist.append(float(anti_stats["base_top1_share"]))
        on_top1_hist.append(float(anti_stats["on_top1_share"]))

    return {
        "answer_path_loss": float(sum(answer_hist) / max(1, len(answer_hist))),
        "answer_delta": float(sum(delta_hist) / max(1, len(delta_hist))),
        "alignment_similarity": float(sum(sim_hist) / max(1, len(sim_hist))),
        "anti_collapse_loss": float(sum(anti_hist) / max(1, len(anti_hist))),
        "base_operator_entropy": float(sum(base_entropy_hist) / max(1, len(base_entropy_hist))),
        "on_operator_entropy": float(sum(on_entropy_hist) / max(1, len(on_entropy_hist))),
        "base_top1_share": float(sum(base_top1_hist) / max(1, len(base_top1_hist))),
        "on_top1_share": float(sum(on_top1_hist) / max(1, len(on_top1_hist))),
    }


def _evaluate_answer_path_cell(
    cell: str,
    bridge: RelationLocalRotaryBridge,
    pointer_head: CandidatePointerHead | None,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    eval_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge.eval()
    if pointer_head is not None:
        pointer_head.eval()
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    answer_delta_hist: list[float] = []
    sim_eval_hist: list[float] = []

    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        with torch.no_grad():
            _eng_hidden, local_hidden, local_meta = _extract_relation_local_states(
                model,
                tokenizer,
                prompt,
                candidates,
                int(args.layer_index),
            )
            h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)
            base_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
            base_ids = torch.stack(base_tokens, dim=1)
            rel_nodes = base_states[:, 0::3, :]
            _eval_align, eval_sim = bridge.alignment_metrics(local_hidden, rel_nodes)
            sim_eval_hist.append(float(eval_sim.item()))

            if cell == "A":
                z_use = z_st
            else:
                cue = bridge.runtime_cue(local_hidden, cue_norm_cap=float(args.runtime_cue_norm_cap))
                gate = torch.sigmoid(bridge.gate)
                if float(args.runtime_gate_cap) > 0.0:
                    gate = torch.clamp(gate, max=float(args.runtime_gate_cap))
                z_use = z_st + gate * cue

            on_tokens = _decode_logic_tokens(arity_head, z_use, int(args.relation_vocab), int(args.var_min_id))
            on_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in on_tokens], dim=1)
            on_ids = torch.stack(on_tokens, dim=1)

            if cell == "D":
                assert pointer_head is not None
                candidate_states = _candidate_states_from_local(local_hidden, local_meta)
                logits_on = pointer_head(candidate_states, on_states[:, 0::3, :])
                logits_off = pointer_head(candidate_states, base_states[:, 0::3, :])
                pred_idx = int(torch.argmax(logits_on, dim=-1)[0].item())
                score_on_gold = float(logits_on[0, gi].item())
                score_on_foil = float(logits_on[0, fi].item())
                answer_delta_hist.append(float((logits_on[0, gi] - logits_on[0, fi]).item()))
                deltas.append(float((logits_on[0, gi] - logits_off[0, gi]).item()))
            else:
                blindfold = cell == "C"
                logp_on_gold = _score_candidate_first_token_with_adapter_tensor(
                    model, tokenizer, prompt, gold, advisor_adapter, on_states, on_ids, int(args.layer_index), blindfold_question=blindfold
                )
                logp_on_foil = _score_candidate_first_token_with_adapter_tensor(
                    model, tokenizer, prompt, foil, advisor_adapter, on_states, on_ids, int(args.layer_index), blindfold_question=blindfold
                )
                logp_off_gold = _score_candidate_first_token_with_adapter_tensor(
                    model, tokenizer, prompt, gold, advisor_adapter, base_states, base_ids, int(args.layer_index), blindfold_question=blindfold
                )
                pred_idx = gi if float(logp_on_gold.item()) >= float(logp_on_foil.item()) else fi
                score_on_gold = float(logp_on_gold.item())
                score_on_foil = float(logp_on_foil.item())
                answer_delta_hist.append(float((logp_on_gold - logp_on_foil).item()))
                deltas.append(float((logp_on_gold - logp_off_gold).item()))

        pred = str(candidates[pred_idx])
        correct = _answer_match(gold, pred)
        token_ids = [int(t[0].detach().item()) for t in on_tokens]
        rel_ids = token_ids[0::3]
        counts = Counter(rel_ids)
        total = max(1, len(rel_ids))
        probs = [float(c) / float(total) for c in counts.values()]
        entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
        active_op_count = int(sum(1 for _op, c in counts.items() if int(c) >= 1))
        top1_op_share = float(max(counts.values())) / float(total) if counts else 1.0
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        rows.append(
            {
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
                "score_on_gold": float(score_on_gold),
                "score_on_foil": float(score_on_foil),
                "answer_delta": float(answer_delta_hist[-1]),
                "gold_delta_on_off": float(deltas[-1]),
                "cell_mode": {"A": "control", "B": "direct_drive", "C": "blindfold", "D": "pointer"}[cell],
                "local_node_labels": list(local_meta.get("node_labels", [])),
                "local_spans": dict(local_meta.get("spans", {})),
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
        "mean_alignment_similarity": float(sum(sim_eval_hist) / max(1, len(sim_eval_hist))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_active_op_count": float(sum(float(r["active_op_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_top1_op_share": float(sum(float(r["operator_top1_share"]) for r in rows) / n),
        "mean_scope": float(sum(float(r["scope"]) for r in rows) / n),
        "mean_intervention_delta_gold": float(sum(deltas) / max(1, len(deltas))),
    }
    return metrics, rows


def _promotion_gates(report: dict[str, Any], min_acc_gain: float, scope_tol: float, entropy_floor_ratio: float) -> dict[str, Any]:
    a = report["cells"]["A"]["metrics"]
    gates: dict[str, Any] = {}
    for cell in ("B", "C", "D"):
        m = report["cells"][cell]["metrics"]
        gates[cell] = {
            "accuracy_up": float(m["overall_accuracy"]) >= float(a["overall_accuracy"]) + float(min_acc_gain),
            "no_entropy_collapse": float(m["mean_operator_entropy"]) >= float(a["mean_operator_entropy"]) * float(entropy_floor_ratio),
            "no_scope_regression": float(m["mean_scope"]) <= float(a["mean_scope"]) + float(scope_tol),
            "positive_answer_delta": float(m["mean_answer_delta"]) > 0.0,
            "positive_intervention_delta": float(m["mean_intervention_delta_gold"]) > 0.0,
        }
        gates[cell]["promote_to_next"] = all(bool(v) for v in gates[cell].values())
    return gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.15d Answer-Path Forcing: direct-drive, blindfolded, and pointer-head answer-path ablation.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=60)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=12)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--anti-collapse-weight", type=float, default=1.0)
    p.add_argument("--collapse-entropy-floor-ratio", type=float, default=0.85)
    p.add_argument("--collapse-top1-margin", type=float, default=0.10)
    p.add_argument("--collapse-top1-weight", type=float, default=1.0)
    p.add_argument("--collapse-kl-weight", type=float, default=0.5)
    p.add_argument("--bridge-train-gate-cap", type=float, default=0.08)
    p.add_argument("--runtime-gate-cap", type=float, default=0.03)
    p.add_argument("--runtime-cue-norm-cap", type=float, default=1.0)
    p.add_argument("--runtime-enable-min-acc-gain", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing"))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(args.seed))
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
    advisor_adapter = CouncilCrossAttentionAdapter(hidden, use_boolean_surgery=True).to(target_device, dtype=module_dtype)
    advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    advisor_adapter.eval()
    for param in advisor_adapter.parameters():
        param.requires_grad = False

    if args.pack_jsonl is None:
        pack_train, pack_val, pack_eval, split_meta = _build_winograd_split_packs(
            size=int(args.eval_size),
            seed=int(args.seed),
            strict_balance=bool(args.strict_balance),
        )
        _unused_train2, _unused_val2, pack_eval_seed2, split_meta_seed2 = _build_winograd_split_packs(
            size=int(args.eval_size),
            seed=int(args.seed) + 1,
            strict_balance=bool(args.strict_balance),
        )
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
        pack_eval_seed2 = _load_pack(args.pack_jsonl, size=int(args.eval_size), seed=int(args.seed) + 2, strict_balance=bool(args.strict_balance))
        split_meta = {
            "train_pair_ids": sorted(train_ids),
            "val_pair_ids": sorted(val_ids),
            "eval_pair_ids": sorted(eval_ids),
        }
        split_meta_seed2 = {"seed": int(args.seed) + 1}
    (run_dir / "m3_15d_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")

    cells = track_cell_labels("M3.15d")
    report: dict[str, Any] = build_bridge_report(
        track="M3.15d",
        script_path="scripts/run_m3_15d_answer_path_forcing.py",
        args=args,
        baseline_manifest_path=args.baseline_manifest,
        baseline_id=str(baseline_manifest.get("baseline_id", "")),
        checkpoint_in=str(args.checkpoint),
        split_meta=split_meta,
        seed2_meta=split_meta_seed2,
        runtime_policy_source="n/a",
        final_metrics_source="eval_pack",
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["series"] = series_metadata("M", "M3.15d", "scripts/run_m3_15d_answer_path_forcing.py")
    report["lineage"] = lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed")

    bridge_a = RelationLocalRotaryBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_a = _train_answer_path_cell("A", bridge_a, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_a_val, _rows_a_val = _evaluate_answer_path_cell("A", bridge_a, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args)
    met_a, rows_a = _evaluate_answer_path_cell("A", bridge_a, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val}
    (run_dir / "m3_15d_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")
    print(f"A: {met_a}")

    bridge_b = RelationLocalRotaryBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_b = _train_answer_path_cell("B", bridge_b, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_b_val, _rows_b_val = _evaluate_answer_path_cell("B", bridge_b, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args)
    met_b, rows_b = _evaluate_answer_path_cell("B", bridge_b, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args)
    report["cells"]["B"] = {"train": train_b, "metrics": met_b, "validation_metrics": met_b_val}
    (run_dir / "m3_15d_B_eval.json").write_text(json.dumps(rows_b, indent=2), encoding="utf-8")
    print(f"B: {met_b}")

    bridge_c = RelationLocalRotaryBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_c = _train_answer_path_cell("C", bridge_c, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_c_val, _rows_c_val = _evaluate_answer_path_cell("C", bridge_c, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args)
    met_c, rows_c = _evaluate_answer_path_cell("C", bridge_c, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args)
    report["cells"]["C"] = {"train": train_c, "metrics": met_c, "validation_metrics": met_c_val}
    (run_dir / "m3_15d_C_eval.json").write_text(json.dumps(rows_c, indent=2), encoding="utf-8")
    print(f"C: {met_c}")

    bridge_d = RelationLocalRotaryBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    pointer_d = CandidatePointerHead(hidden).to(target_device, dtype=module_dtype)
    train_d = _train_answer_path_cell("D", bridge_d, pointer_d, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_d_val, _rows_d_val = _evaluate_answer_path_cell("D", bridge_d, pointer_d, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args)
    met_d, rows_d = _evaluate_answer_path_cell("D", bridge_d, pointer_d, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args)
    report["cells"]["D"] = {"train": train_d, "metrics": met_d, "validation_metrics": met_d_val}
    (run_dir / "m3_15d_D_eval.json").write_text(json.dumps(rows_d, indent=2), encoding="utf-8")
    print(f"D: {met_d}")

    met_b_seed2, rows_b_seed2 = _evaluate_answer_path_cell("B", bridge_b, None, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval_seed2, args)
    report["cells"]["B_seed2_eval"] = {"metrics": met_b_seed2}
    (run_dir / "m3_15d_B_seed2_eval.json").write_text(json.dumps(rows_b_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(
        report,
        min_acc_gain=float(args.runtime_enable_min_acc_gain),
        scope_tol=0.02,
        entropy_floor_ratio=0.8,
    )

    report = finalize_bridge_report(report, "M3.15d")
    (run_dir / "m3_15d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# M3.15d Answer-Path Forcing",
        "",
        "| Cell | Regime | Acc | Adj Acc | Causal Acc | Ans Delta | Delta(gold on-off) | Active Ops | Entropy | Scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("A", "B", "C", "D"):
        m = report["cells"][c]["metrics"]
        md.append(
            f"| {c} | {cells[c]} | {m['overall_accuracy']:.3f} | {m['adjective_accuracy']:.3f} | {m['causal_accuracy']:.3f} | "
            f"{m['mean_answer_delta']:.4f} | {m['mean_intervention_delta_gold']:.4f} | "
            f"{m['mean_active_op_count']:.2f} | {m['mean_operator_entropy']:.4f} | {m['mean_scope']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Regimes",
            "- B: differentiable gold-vs-foil answer-path loss through the LM answer logits.",
            "- C: same answer-path loss, but with question tokens attention-masked out during final answer scoring.",
            "- D: no LM answer logits; a candidate-pointer head chooses between the active candidates directly from relation-local states and advisor relation nodes.",
            "",
            "## Anti-Collapse Controls",
            f"- anti_collapse_weight: `{float(args.anti_collapse_weight):.3f}`",
            f"- collapse_entropy_floor_ratio: `{float(args.collapse_entropy_floor_ratio):.3f}`",
            f"- collapse_top1_margin: `{float(args.collapse_top1_margin):.3f}`",
            f"- collapse_top1_weight: `{float(args.collapse_top1_weight):.3f}`",
            f"- collapse_kl_weight: `{float(args.collapse_kl_weight):.3f}`",
            "",
            "## Promotion Gates",
        ]
    )
    for cell in ("B", "C", "D"):
        md.append(f"- {cell}:")
        for key, value in report["promotion_gates"][cell].items():
            md.append(f"  - {key}: `{value}`")
    (run_dir / "m3_15d_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M3.15d complete: {run_dir}")


if __name__ == "__main__":
    main()


