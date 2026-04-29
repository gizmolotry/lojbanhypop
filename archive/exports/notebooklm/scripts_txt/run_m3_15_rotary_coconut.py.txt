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


def _candidate_first_token_id(tokenizer, candidate: str) -> int:
    ids = tokenizer(" " + str(candidate), add_special_tokens=False, return_tensors="pt").input_ids
    if int(ids.numel()) == 0:
        return int(tokenizer.eos_token_id or 0)
    return int(ids[0, 0].item())


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


def _to_complex_phase(x: torch.Tensor) -> torch.Tensor:
    d2 = (int(x.shape[-1]) // 2) * 2
    x = x[..., :d2]
    r = x[..., : d2 // 2]
    i = x[..., d2 // 2 :]
    return torch.atan2(i, r + 1e-8)


class RotaryCoconutBridge(torch.nn.Module):
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

    def _select_nodes(self, h: torch.Tensor) -> torch.Tensor:
        scores = torch.norm(h, dim=-1)
        k = max(2, min(int(self.max_nodes), int(h.shape[1])))
        idx = torch.topk(scores[0], k=k, largest=True).indices
        idx, _ = torch.sort(idx)
        return h[:, idx, :]

    def _phase_alignment(self, eng_nodes: torch.Tensor, adv_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(int(eng_nodes.shape[1]), int(adv_nodes.shape[1]))
        if k < 2:
            z = torch.zeros((), device=eng_nodes.device, dtype=eng_nodes.dtype)
            return z, z
        e = self.eng_proj(eng_nodes[:, :k, :])
        a = self.adv_proj(adv_nodes[:, :k, :])
        pe = _to_complex_phase(e)
        pa = _to_complex_phase(a)
        de = pe[:, 1:, :] - pe[:, :-1, :]
        da = pa[:, 1:, :] - pa[:, :-1, :]
        d = torch.atan2(torch.sin(de - da), torch.cos(de - da))
        loss = torch.mean(d * d)
        sim = F.cosine_similarity(de.reshape(1, -1), da.reshape(1, -1), dim=-1)[0]
        return loss, sim

    def relation_bias(self, eng_hidden: torch.Tensor, adv_rel_nodes: torch.Tensor) -> torch.Tensor:
        en = self._select_nodes(eng_hidden)
        an = adv_rel_nodes
        k = min(int(en.shape[1]), int(an.shape[1]))
        if k <= 0:
            return torch.zeros((self.relation_vocab,), device=eng_hidden.device, dtype=eng_hidden.dtype)
        e = self.eng_proj(en[:, :k, :])
        a = self.adv_proj(an[:, :k, :])
        fe = torch.mean(torch.cos(_to_complex_phase(e)), dim=1)
        fa = torch.mean(torch.cos(_to_complex_phase(a)), dim=1)
        fused = torch.tanh(fe - fa)
        fused = self._match_bias_input_dim(fused)
        return self.bias_head(fused)[0]

    def runtime_cue(self, eng_hidden: torch.Tensor, cue_norm_cap: float) -> torch.Tensor:
        nodes = self._select_nodes(eng_hidden)
        pooled = torch.mean(self.eng_proj(nodes), dim=1, keepdim=True)
        cue = torch.tanh(self.cue_head(pooled)) * torch.sigmoid(self.cue_scale)
        cap = float(cue_norm_cap)
        if cap > 0.0:
            n = torch.norm(cue, dim=-1, keepdim=True).clamp(min=1e-8)
            cue = cue * torch.clamp(cap / n, max=1.0)
        return cue

    def alignment_metrics(self, eng_hidden: torch.Tensor, adv_rel_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._phase_alignment(self._select_nodes(eng_hidden), adv_rel_nodes)


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
    return _materialize_pack(_pair_variants(_WINOGRAD_PAIR_BANK), size=size, seed=seed, strict_balance=strict_balance, item_prefix="m3_15")


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
    train_pack = _materialize_pack(_pair_variants(train_pairs), size=size, seed=seed, strict_balance=strict_balance, item_prefix="m3_15_train")
    val_pack = _materialize_pack(_pair_variants(val_pairs), size=size, seed=seed + 101, strict_balance=strict_balance, item_prefix="m3_15_val")
    eval_pack = _materialize_pack(_pair_variants(eval_pairs), size=size, seed=seed + 202, strict_balance=strict_balance, item_prefix="m3_15_eval")
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


def _train_bridge_cell(
    cell: str,
    bridge: RotaryCoconutBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    train_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    if cell == "A":
        return {"alignment_loss": 0.0, "margin_loss": 0.0, "alignment_similarity": 0.0}

    bridge.train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=float(args.lr), weight_decay=0.01)
    align_hist: list[float] = []
    marg_hist: list[float] = []
    sim_hist: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(model.device)
            with adapter_disabled(model):
                out = model(input_ids=p_ids, output_hidden_states=True, use_cache=False)
            eng_hidden = out.hidden_states[int(args.layer_index)].detach()

            h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)
            base_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
            rel_nodes = base_states[:, 0::3, :]

        opt.zero_grad()
        align_loss, align_sim = bridge.alignment_metrics(eng_hidden, rel_nodes)
        rb = bridge.relation_bias(eng_hidden, rel_nodes)
        on_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id), relation_bias=rb)
        on_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in on_tokens], dim=1)
        on_ids = torch.stack(on_tokens, dim=1)
        s_gold = _score_candidate_first_token_with_adapter(
            model, tokenizer, prompt, gold, advisor_adapter, on_states, on_ids, int(args.layer_index)
        )
        s_foil = _score_candidate_first_token_with_adapter(
            model, tokenizer, prompt, foil, advisor_adapter, on_states, on_ids, int(args.layer_index)
        )
        margin_loss = torch.relu(torch.tensor(float(args.margin), device=model.device, dtype=model.dtype) - (torch.tensor(s_gold, device=model.device, dtype=model.dtype) - torch.tensor(s_foil, device=model.device, dtype=model.dtype)))
        reg = 0.01 * torch.mean(rb * rb)
        loss = float(args.align_weight) * align_loss + float(args.margin_weight) * margin_loss + reg
        loss.backward()
        opt.step()
        align_hist.append(float(align_loss.detach().item()))
        marg_hist.append(float(margin_loss.detach().item()))
        sim_hist.append(float(align_sim.detach().item()))

    return {
        "alignment_loss": float(sum(align_hist) / max(1, len(align_hist))),
        "margin_loss": float(sum(marg_hist) / max(1, len(marg_hist))),
        "alignment_similarity": float(sum(sim_hist) / max(1, len(sim_hist))),
    }


def _evaluate_cell(
    cell: str,
    bridge: RotaryCoconutBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    advisor_adapter: CouncilCrossAttentionAdapter,
    eval_pack: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime_enabled: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge.eval()
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    align_eval_hist: list[float] = []
    sim_eval_hist: list[float] = []

    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        with torch.no_grad():
            p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(model.device)
            with adapter_disabled(model):
                out = model(input_ids=p_ids, output_hidden_states=True, use_cache=False)
            eng_hidden = out.hidden_states[int(args.layer_index)].detach()

            h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
            z_st, _idx, _cb, _commit = codebook.quantize(h_t)
            base_tokens = _decode_logic_tokens(arity_head, z_st, int(args.relation_vocab), int(args.var_min_id))
            base_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in base_tokens], dim=1)
            base_ids = torch.stack(base_tokens, dim=1)
            rel_nodes = base_states[:, 0::3, :]
            eval_align, eval_sim = bridge.alignment_metrics(eng_hidden, rel_nodes)
            align_eval_hist.append(float(eval_align.item()))
            sim_eval_hist.append(float(eval_sim.item()))

            rb = bridge.relation_bias(eng_hidden, rel_nodes) if cell in {"B", "C"} else None
            if cell == "C" and runtime_enabled:
                cue = bridge.runtime_cue(eng_hidden, cue_norm_cap=float(args.runtime_cue_norm_cap))
                gate = torch.sigmoid(bridge.gate)
                if float(args.runtime_gate_cap) > 0.0:
                    gate = torch.clamp(gate, max=float(args.runtime_gate_cap))
                z_use = z_st + gate * cue
            else:
                cue = None
                z_use = z_st

            on_tokens = _decode_logic_tokens(arity_head, z_use, int(args.relation_vocab), int(args.var_min_id), relation_bias=rb)
            on_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in on_tokens], dim=1)
            on_ids = torch.stack(on_tokens, dim=1)

            score_on_gold = _score_candidate_first_token_with_adapter(
                model, tokenizer, prompt, gold, advisor_adapter, on_states, on_ids, int(args.layer_index)
            )
            score_on_foil = _score_candidate_first_token_with_adapter(
                model, tokenizer, prompt, foil, advisor_adapter, on_states, on_ids, int(args.layer_index)
            )
            pred_idx = gi if score_on_gold >= score_on_foil else fi
            pred = str(candidates[pred_idx])
            correct = _answer_match(gold, pred)

            score_off_gold = _score_candidate_first_token_with_adapter(
                model, tokenizer, prompt, gold, advisor_adapter, base_states, base_ids, int(args.layer_index)
            )
            deltas.append(float(score_on_gold - score_off_gold))

        token_ids = [int(t[0].detach().item()) for t in on_tokens]
        rel_ids = token_ids[0::3]
        counts = Counter(rel_ids)
        total = max(1, len(rel_ids))
        probs = [float(c) / float(total) for c in counts.values()]
        entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
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
                "operator_entropy": float(entropy),
                "scope": float(scope.get("scope_total", 1.0)),
                "score_on_gold": float(score_on_gold),
                "score_on_foil": float(score_on_foil),
                "gold_logit_delta_on_off": float(deltas[-1]),
                "runtime_cue_norm": float(torch.norm(cue).item()) if cue is not None else 0.0,
            }
        )

    n = max(1, len(rows))
    fam_adj = [r for r in rows if str(r["family"]) == "adjective_property"]
    fam_cau = [r for r in rows if str(r["family"]) == "causal_direction"]
    metrics = {
        "overall_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
        "adjective_accuracy": float(sum(1 for r in fam_adj if r["correct"]) / max(1, len(fam_adj))),
        "causal_accuracy": float(sum(1 for r in fam_cau if r["correct"]) / max(1, len(fam_cau))),
        "mean_alignment_loss": float(sum(align_eval_hist) / max(1, len(align_eval_hist))),
        "mean_alignment_similarity": float(sum(sim_eval_hist) / max(1, len(sim_eval_hist))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_scope": float(sum(float(r["scope"]) for r in rows) / n),
        "mean_intervention_delta_gold": float(sum(deltas) / max(1, len(deltas))),
        "runtime_enabled": bool(runtime_enabled),
    }
    return metrics, rows


def _promotion_gates(report: dict[str, Any], min_acc_gain: float, scope_tol: float, entropy_floor_ratio: float) -> dict[str, Any]:
    a = report["cells"]["A"]["metrics"]
    b = report["cells"]["B"]["metrics"]
    b_seed2 = report["cells"]["B_seed2_eval"]["metrics"]
    gates = {
        "accuracy_up": float(b["overall_accuracy"]) >= float(a["overall_accuracy"]) + float(min_acc_gain),
        "no_entropy_collapse": float(b["mean_operator_entropy"]) >= float(a["mean_operator_entropy"]) * float(entropy_floor_ratio),
        "no_scope_regression": float(b["mean_scope"]) <= float(a["mean_scope"]) + float(scope_tol),
        "positive_intervention_delta": float(b["mean_intervention_delta_gold"]) > 0.0,
        "seed_stability": float(b_seed2["overall_accuracy"]) >= float(a["overall_accuracy"]) + float(min_acc_gain),
    }
    gates["promote_to_next"] = all(bool(v) for v in gates.values())
    return gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.15 Rotary Coconut: structural alignment + decision margin + relation-head bias.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=12)
    p.add_argument("--align-weight", type=float, default=1.0)
    p.add_argument("--margin-weight", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--runtime-gate-cap", type=float, default=0.03)
    p.add_argument("--runtime-cue-norm-cap", type=float, default=1.0)
    p.add_argument("--runtime-enable-min-acc-gain", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut"))
    p.add_argument("--local-files-only", action="store_true")
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
    use_bf16 = False
    if target_device.type == "cuda":
        if use_bf16:
            model = model.to(device=target_device, dtype=torch.bfloat16)
        else:
            model = model.to(device=target_device)
    else:
        model = model.to(target_device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=target_device)
    hidden = int(model.config.hidden_size)
    module_dtype = next(model.parameters()).dtype
    codebook = BooleanAnchorTable(2000, hidden).to(target_device, dtype=module_dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    arity_head = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    advisor_adapter = CouncilCrossAttentionAdapter(hidden, use_boolean_surgery=True).to(target_device, dtype=module_dtype)
    advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)

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
    (run_dir / "m3_15_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")

    cells = {"A": "control", "B": "alignment+margin(no runtime cue)", "C": "alignment+margin(+runtime cue if enabled)"}
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M3.15", "scripts/run_m3_15_rotary_coconut.py"),
        "track": "M3.15",
        "lineage": lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed"),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {k: str(v) for k, v in vars(args).items()},
        "data_split": {
            **split_meta,
            "seed2_eval_meta": split_meta_seed2,
            "runtime_policy_source": "validation_pack",
            "final_metrics_source": "eval_pack",
        },
        "cells": {},
    }

    # A
    bridge_a = RotaryCoconutBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_a = _train_bridge_cell("A", bridge_a, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_a_val, _rows_a_val = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args, runtime_enabled=False)
    met_a, rows_a = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args, runtime_enabled=False)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val}
    (run_dir / "m3_15_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")
    print(f"A: {met_a}")

    # B
    bridge_b = RotaryCoconutBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_b = _train_bridge_cell("B", bridge_b, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_b_val, _rows_b_val = _evaluate_cell("B", bridge_b, model, tokenizer, codebook, arity_head, advisor_adapter, pack_val, args, runtime_enabled=False)
    met_b, rows_b = _evaluate_cell("B", bridge_b, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args, runtime_enabled=False)
    report["cells"]["B"] = {"train": train_b, "metrics": met_b, "validation_metrics": met_b_val}
    (run_dir / "m3_15_B_eval.json").write_text(json.dumps(rows_b, indent=2), encoding="utf-8")
    print(f"B: {met_b}")

    runtime_enabled = float(met_b_val["overall_accuracy"]) >= float(met_a_val["overall_accuracy"]) + float(args.runtime_enable_min_acc_gain)
    # C
    bridge_c = RotaryCoconutBridge(hidden, int(args.relation_vocab), max_nodes=int(args.max_nodes)).to(target_device, dtype=module_dtype)
    train_c = _train_bridge_cell("C", bridge_c, model, tokenizer, codebook, arity_head, advisor_adapter, pack_train, args)
    met_c, rows_c = _evaluate_cell("C", bridge_c, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval, args, runtime_enabled=bool(runtime_enabled))
    report["cells"]["C"] = {"train": train_c, "metrics": met_c}
    report["cells"]["C"]["runtime_enabled_by_policy"] = bool(runtime_enabled)
    (run_dir / "m3_15_C_eval.json").write_text(json.dumps(rows_c, indent=2), encoding="utf-8")
    print(f"C: {met_c} runtime_enabled={runtime_enabled}")

    # Seed stability check (B only)
    met_b_seed2, rows_b_seed2 = _evaluate_cell("B", bridge_b, model, tokenizer, codebook, arity_head, advisor_adapter, pack_eval_seed2, args, runtime_enabled=False)
    report["cells"]["B_seed2_eval"] = {"metrics": met_b_seed2}
    (run_dir / "m3_15_B_seed2_eval.json").write_text(json.dumps(rows_b_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(
        report,
        min_acc_gain=float(args.runtime_enable_min_acc_gain),
        scope_tol=0.02,
        entropy_floor_ratio=0.8,
    )

    (run_dir / "m3_15_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# M3.15 Rotary Coconut",
        "",
        "| Cell | Acc | Adj Acc | Causal Acc | Align Loss | Align Sim | Delta(gold on-off) | Active | Entropy | Scope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("A", "B", "C"):
        m = report["cells"][c]["metrics"]
        md.append(
            f"| {c} | {m['overall_accuracy']:.3f} | {m['adjective_accuracy']:.3f} | {m['causal_accuracy']:.3f} | "
            f"{m['mean_alignment_loss']:.4f} | {m['mean_alignment_similarity']:.4f} | {m['mean_intervention_delta_gold']:.4f} | "
            f"{m['mean_active_tokens']:.2f} | {m['mean_operator_entropy']:.4f} | {m['mean_scope']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Runtime Cue Policy",
            f"- B vs A accuracy threshold: `{float(args.runtime_enable_min_acc_gain):.3f}`",
            "- policy_selection_split: `validation`",
            f"- C runtime enabled: `{bool(runtime_enabled)}`",
            "",
            "## Promotion Gates",
        ]
    )
    for k, v in report["promotion_gates"].items():
        md.append(f"- {k}: `{v}`")
    (run_dir / "m3_15_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M3.15 complete: {run_dir}")


if __name__ == "__main__":
    main()
