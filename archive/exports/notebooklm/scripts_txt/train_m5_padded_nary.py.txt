from __future__ import annotations

import argparse
import json
import math
import random
import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import generate_dataset, split_dataset
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata
from train_h5_persistent_vq_advisor import BooleanAnchorTable, adapter_disabled, build_final_prefix, extract_trace_hidden_states
from build_m3_15_winograd_pack import _build_row, _specs, _validate_rows


PAD_TOKEN_ID = 2000
VAR_TOKEN_BASE = 2001


@dataclass
class TrainTelemetry:
    step: int
    total_loss: float
    task_loss: float
    semantic_ce: float
    winograd_ce: float
    slot_l1_loss: float
    uniformity_loss: float
    grl_loss: float
    invariance_loss: float
    cpc_loss: float
    slot_usage_mean: float
    operator_entropy_batch: float
    top1_op_share_batch: float
    winograd_accuracy_batch: float


@contextmanager
def _disable_adapter_ctx(model):
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
def padded_graph_bias_hook(model, layer_index: int, adapter_module, graph_batch: dict[str, torch.Tensor], scale: float = 1.0):
    layers = _resolve_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for padded graph hook.")

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        delta = adapter_module(
            hidden_states=hidden,
            op_embedding=graph_batch["op_embedding"],
            slot_tensors=graph_batch["slot_tensors"],
            slot_mask=graph_batch["slot_mask"],
        ).to(dtype=hidden.dtype, device=hidden.device)
        hidden = hidden + (float(scale) * delta)
        return (hidden, *rest) if rest is not None else hidden

    handle = layers[int(layer_index)].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * float(ctx.scale), None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradReverse.apply(x, float(scale))


class NounAdversary(torch.nn.Module):
    def __init__(self, hidden_size: int, buckets: int):
        super().__init__()
        mid = max(64, hidden_size // 2)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mid),
            torch.nn.SiLU(),
            torch.nn.Linear(mid, buckets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PaddedNaryEmitter(torch.nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int = 2000, max_slots: int = 10):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.codebook_size = int(codebook_size)
        self.max_slots = int(max_slots)
        self.slot_queries = torch.nn.Parameter(torch.randn(self.max_slots, self.hidden_size) * 0.02)
        self.logic_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.logic_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.logic_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.op_pre = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_gate = torch.nn.Linear(hidden_size, 1, bias=True)

    def _attend(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(float(queries.shape[-1]))
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, values)
        return ctx, weights

    def forward(
        self,
        logic_hidden: torch.Tensor,
        prompt_hidden: torch.Tensor,
        codebook: BooleanAnchorTable,
    ) -> dict[str, torch.Tensor]:
        b, _t, h = logic_hidden.shape
        logic_q = self.logic_q(self.slot_queries).unsqueeze(0).expand(b, -1, -1)
        logic_k = self.logic_k(logic_hidden)
        logic_v = self.logic_v(logic_hidden)
        slot_seed, logic_attn = self._attend(logic_q, logic_k, logic_v)

        op_seed = slot_seed[:, 0, :]
        op_pre = self.op_pre(op_seed)

        slot_q = self.prompt_q(slot_seed[:, 1:, :])
        prompt_k = self.prompt_k(prompt_hidden)
        prompt_v = self.prompt_v(prompt_hidden)
        slot_tensors, prompt_attn = self._attend(slot_q, prompt_k, prompt_v)
        slot_use_logits = self.slot_gate(slot_seed[:, 1:, :]).squeeze(-1)
        slot_use_probs = torch.sigmoid(slot_use_logits)

        emb = codebook.emb
        diff = op_pre.unsqueeze(1) - emb.unsqueeze(0)
        dist = torch.sum(diff * diff, dim=-1)
        op_logits = -dist
        op_probs = torch.softmax(op_logits, dim=-1)
        op_idx = torch.argmax(op_logits, dim=-1)
        op_embedding = emb[op_idx]
        slot_indices = torch.arange(self.max_slots - 1, device=slot_use_probs.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        slot_token_ids = torch.where(
            slot_use_probs >= 0.5,
            slot_indices + int(VAR_TOKEN_BASE),
            torch.full_like(slot_indices, int(PAD_TOKEN_ID)),
        )
        matrix_token_ids = torch.cat([op_idx.unsqueeze(-1), slot_token_ids], dim=-1)

        return {
            "op_pre": op_pre,
            "op_logits": op_logits,
            "op_probs": op_probs,
            "op_idx": op_idx,
            "op_embedding": op_embedding,
            "slot_tensors": slot_tensors,
            "slot_use_logits": slot_use_logits,
            "slot_use_probs": slot_use_probs,
            "slot_token_ids": slot_token_ids,
            "matrix_token_ids": matrix_token_ids,
            "logic_attention": logic_attn,
            "prompt_attention": prompt_attn,
        }


class SoftGraphBiasCouncilAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_val = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.op_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_states: torch.Tensor, op_embedding: torch.Tensor, slot_tensors: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden_states)
        k = self.slot_key(slot_tensors)
        v = self.slot_val(slot_tensors)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
        mask = slot_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(mask, attn, torch.zeros_like(attn))
        denom = torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-8)
        attn = attn / denom
        slot_ctx = torch.matmul(attn, v)
        op_ctx = self.op_proj(op_embedding).unsqueeze(1)
        fused = torch.tanh(op_ctx + self.slot_proj(slot_ctx) + self.hidden_proj(hidden_states))
        return self.out_proj(fused) * torch.sigmoid(self.gate) * self.scale


ENTITY_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","because","since","while","as","who","what","where","when",
    "which","does","did","do","is","was","were","are","be","been","being","to","of","in","on","at","for","from",
    "with","without","by","it","they","them","their","this","that","these","those","too","very","not","no","so",
}


def extract_entity_lexemes(prompt: str, max_terms: int = 4) -> List[str]:
    words = re.findall(r"[A-Za-z]+", str(prompt))
    out: List[str] = []
    seen: set[str] = set()
    for word in words:
        token = word.lower()
        if len(token) < 3 or token in ENTITY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= int(max_terms):
            break
    return out


def lexical_bucket(prompt: str, buckets: int) -> Optional[int]:
    import hashlib
    lex = extract_entity_lexemes(prompt)
    if not lex:
        return None
    payload = "|".join(lex).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16) % max(2, int(buckets))


def _build_winograd_rows(size: int, seed: int, strict_balance: bool = True) -> List[dict[str, Any]]:
    rng = random.Random(int(seed))
    specs = _specs()
    rows: List[dict[str, Any]] = []
    pair_idx = 0
    target_even = int(size) if (int(size) % 2 == 0) else (int(size) - 1)
    while len(rows) < target_even:
        spec = specs[pair_idx % len(specs)]
        rows.append(_build_row(spec, pair_id=pair_idx, variant_index=0))
        rows.append(_build_row(spec, pair_id=pair_idx, variant_index=1))
        pair_idx += 1
    if len(rows) > target_even:
        rows = rows[:target_even]
    if int(size) % 2 != 0:
        spec = specs[pair_idx % len(specs)]
        rows.append(_build_row(spec, pair_id=pair_idx, variant_index=rng.randint(0, 1)))
    rng.shuffle(rows)
    if len(rows) >= 500:
        _validate_rows(rows, strict_balance=bool(strict_balance))
    else:
        required = {"family", "polarity", "causal_direction", "pair_id", "variant_id", "candidates", "gold_index"}
        c0 = sum(1 for r in rows if int(r.get("gold_index", -1)) == 0)
        c1 = sum(1 for r in rows if int(r.get("gold_index", -1)) == 1)
        if strict_balance and c0 != c1:
            raise ValueError(f"strict balance violated: candidate_0={c0}, candidate_1={c1}")
        for i, row in enumerate(rows):
            missing = required.difference(row.keys())
            if missing:
                raise ValueError(f"row {i} missing required metadata fields: {sorted(missing)}")
            candidates = row["candidates"]
            gold_index = int(row["gold_index"])
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise ValueError(f"row {i} candidates must be a list of length 2")
            if gold_index not in (0, 1):
                raise ValueError(f"row {i} gold_index must be 0 or 1")
            if row.get("answer") != candidates[gold_index]:
                raise ValueError(f"row {i} answer must equal candidates[gold_index]")
            if row.get("variant_id") not in {"v0", "v1"}:
                raise ValueError(f"row {i} variant_id must be 'v0' or 'v1'")
    return rows


def _split_rows_by_pair(rows: Sequence[dict[str, Any]], seed: int) -> Tuple[List[dict[str, Any]], List[dict[str, Any]], List[dict[str, Any]], dict[str, List[str]]]:
    pair_ids = sorted({str(r.get("pair_id", "")).strip() for r in rows if str(r.get("pair_id", "")).strip()})
    rng = random.Random(int(seed))
    rng.shuffle(pair_ids)
    n_pairs = max(1, len(pair_ids))
    eval_count = max(1, round(n_pairs * 0.30))
    val_count = max(1, round(n_pairs * 0.15))
    train_cut = max(1, n_pairs - eval_count - val_count)
    val_cut = max(train_cut + 1, n_pairs - eval_count)
    train_ids = set(pair_ids[:train_cut])
    val_ids = set(pair_ids[train_cut:val_cut])
    eval_ids = set(pair_ids[val_cut:])
    if not eval_ids:
        eval_ids = set(pair_ids[-1:])
    if not val_ids:
        val_ids = set(pair_ids[-2:-1] or pair_ids[-1:])
    train_rows = [r for r in rows if str(r.get("pair_id", "")) in train_ids]
    val_rows = [r for r in rows if str(r.get("pair_id", "")) in val_ids]
    eval_rows = [r for r in rows if str(r.get("pair_id", "")) in eval_ids]
    meta = {
        "train_pair_ids": sorted(train_ids),
        "val_pair_ids": sorted(val_ids),
        "eval_pair_ids": sorted(eval_ids),
    }
    return train_rows, val_rows, eval_rows, meta


def _entity_counterfactual_from_row(row: dict[str, Any]) -> tuple[str, list[str], int]:
    candidates = list(row["candidates"])
    replacements: list[str] = []
    object_pool = ["stone", "apple", "jar", "shelf", "crate", "beam", "printer", "hallway", "candle", "backpack", "binder", "ladder"]
    person_pool = ["Alex", "Riley", "Morgan", "Casey", "Jordan", "Taylor", "Priya", "Mina", "Avery", "Drew"]
    for idx, cand in enumerate(candidates):
        pool = person_pool if cand[:1].isupper() else object_pool
        choices = [x for x in pool if x.lower() != cand.lower() and x.lower() not in {c.lower() for c in candidates}]
        replacements.append(choices[idx % len(choices)] if choices else cand)
    prompt = str(row["prompt"])
    for old, new in zip(candidates, replacements):
        prompt = re.sub(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])", new, prompt)
    return prompt, replacements, int(row["gold_index"])


def _candidate_first_token_id(tokenizer, candidate: str) -> int:
    ids = tokenizer(" " + str(candidate), add_special_tokens=False, return_tensors="pt").input_ids
    if int(ids.numel()) == 0:
        return int(tokenizer.eos_token_id or 0)
    return int(ids[0, 0].item())


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _build_final_prefix_inputs(tokenizer, prompt: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(device)
    am = torch.ones_like(p_ids)
    return p_ids, am


def _answer_token_ids(tokenizer, answer: str, device: torch.device, max_tokens: int) -> torch.Tensor:
    return tokenizer(" " + str(answer), return_tensors="pt", add_special_tokens=False).input_ids.to(device)[:, : int(max_tokens)]


def teacher_forced_answer_ce(model, tokenizer, prompt: str, answer: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: SoftGraphBiasCouncilAdapter, max_answer_tokens: int) -> torch.Tensor:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    tgt = _answer_token_ids(tokenizer, answer, device, max_tokens=max_answer_tokens)
    ce = torch.zeros((), device=device, dtype=cur_emb.dtype)
    with adapter_disabled(model):
        for t in range(tgt.shape[1]):
            with padded_graph_bias_hook(model, int(layer_index), adapter, graph_batch, 1.0):
                out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
            ce = ce + F.cross_entropy(out.logits[:, -1, :], tgt[:, t])
            next_emb = model.get_input_embeddings()(tgt[:, t : t + 1])
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype, device=am.device)], dim=1)
    return ce / max(1, tgt.shape[1])


def score_candidate_first_token(model, tokenizer, prompt: str, candidate: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: SoftGraphBiasCouncilAdapter) -> torch.Tensor:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    with padded_graph_bias_hook(model, int(layer_index), adapter, graph_batch, 1.0):
        out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
    return torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]


def greedy_answer(model, tokenizer, prompt: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: SoftGraphBiasCouncilAdapter, max_answer_tokens: int) -> str:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    toks: List[int] = []
    with adapter_disabled(model):
        for _ in range(int(max_answer_tokens)):
            with padded_graph_bias_hook(model, int(layer_index), adapter, graph_batch, 1.0):
                out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
            tok = int(torch.argmax(out.logits[:, -1, :], dim=-1)[0].item())
            if tokenizer.eos_token_id is not None and tok == int(tokenizer.eos_token_id):
                break
            toks.append(tok)
            cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(torch.tensor([[tok]], device=device))], dim=1)
            am = torch.cat([am, torch.ones((1,1), dtype=am.dtype, device=am.device)], dim=1)
    return tokenizer.decode(toks, skip_special_tokens=True).strip()


def _extract_prompt_hidden(model, tokenizer, prompt: str, max_tokens: int = 256) -> torch.Tensor:
    device = _model_device(model)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_tokens)).input_ids.to(device)
    with adapter_disabled(model):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, return_dict=True, use_cache=False)
    return out.hidden_states[-1].detach()


def _answer_text_embedding(model, tokenizer, answer: str) -> torch.Tensor:
    device = _model_device(model)
    ids = tokenizer(" " + str(answer), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    emb = model.get_input_embeddings()(ids)
    vec = emb.mean(dim=1)
    return F.normalize(vec, p=2, dim=-1)


def _binary_accuracy_from_candidates(model, tokenizer, prompt: str, candidates: list[str], gold_index: int, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: SoftGraphBiasCouncilAdapter) -> float:
    s0 = float(score_candidate_first_token(model, tokenizer, prompt, candidates[0], layer_index, graph_batch, adapter).detach().item())
    s1 = float(score_candidate_first_token(model, tokenizer, prompt, candidates[1], layer_index, graph_batch, adapter).detach().item())
    pred = 0 if s0 >= s1 else 1
    return 1.0 if pred == int(gold_index) else 0.0


def _uniformity_loss(op_probs_batch: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(torch.mean(op_probs_batch, dim=0), min=1e-8)
    ent = -torch.sum(p * torch.log(p))
    max_ent = math.log(float(p.shape[-1]))
    return (max_ent - ent) / max(1e-8, max_ent)


def _slot_sparsity_loss(slot_use_probs: torch.Tensor) -> torch.Tensor:
    pos = torch.arange(1, slot_use_probs.shape[-1] + 1, device=slot_use_probs.device, dtype=slot_use_probs.dtype)
    pos = pos / pos.sum()
    l1 = torch.sum(slot_use_probs * pos.unsqueeze(0))
    monotonic = torch.relu(slot_use_probs[:, 1:] - slot_use_probs[:, :-1]).mean() if slot_use_probs.shape[-1] > 1 else torch.zeros((), device=slot_use_probs.device, dtype=slot_use_probs.dtype)
    min_active = torch.relu(1.0 - slot_use_probs.sum(dim=-1)).mean()
    return l1 + monotonic + min_active


def _op_invariance_loss(op_logits_a: torch.Tensor, op_logits_b: torch.Tensor) -> torch.Tensor:
    a = op_logits_a - op_logits_a.mean(dim=-1, keepdim=True)
    b = op_logits_b - op_logits_b.mean(dim=-1, keepdim=True)
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return F.mse_loss(a, b)


def _build_graph_batch(emission: dict[str, torch.Tensor], pad_threshold: float = 0.5) -> dict[str, torch.Tensor]:
    mask = emission["slot_token_ids"] != int(PAD_TOKEN_ID)
    if int(mask.sum().item()) <= 0:
        top_idx = torch.argmax(emission["slot_use_probs"], dim=-1, keepdim=True)
        mask = torch.zeros_like(emission["slot_use_probs"], dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
    slot_token_ids = torch.where(
        mask,
        emission["slot_token_ids"],
        torch.full_like(emission["slot_token_ids"], int(PAD_TOKEN_ID)),
    )
    matrix_token_ids = torch.cat([emission["op_idx"].unsqueeze(-1), slot_token_ids], dim=-1)
    return {
        "op_embedding": emission["op_embedding"],
        "slot_tensors": emission["slot_tensors"],
        "slot_mask": mask,
        "slot_token_ids": slot_token_ids,
        "matrix_token_ids": matrix_token_ids,
    }


def _load_backbone(base_model: str, adapter: Path, local_files_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(str(base_model), local_files_only=local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(adapter), local_files_only=local_files_only, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M5 padded n-ary auto-formalization trainer.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("runs/l_series/m5_padded_nary"))
    p.add_argument("--train-steps", type=int, default=40)
    p.add_argument("--semantic-dataset-size", type=int, default=256)
    p.add_argument("--winograd-pack-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--max-slots", type=int, default=10)
    p.add_argument("--codebook-size", type=int, default=2000)
    p.add_argument("--slot-sparsity-weight", type=float, default=0.25)
    p.add_argument("--uniformity-weight", type=float, default=0.10)
    p.add_argument("--grl-weight", type=float, default=0.10)
    p.add_argument("--grl-scale", type=float, default=1.0)
    p.add_argument("--invariance-weight", type=float, default=0.20)
    p.add_argument("--cpc-weight", type=float, default=0.20)
    p.add_argument("--hash-buckets", type=int, default=64)
    p.add_argument("--dataset-profile", type=str, default="semantic_bench_v1")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_backbone(args.base_model, args.adapter, bool(args.local_files_only))
    device = _model_device(model)
    hidden_size = int(model.config.hidden_size)

    codebook = BooleanAnchorTable(int(args.codebook_size), hidden_size).to(device, dtype=model.dtype)
    emitter = PaddedNaryEmitter(hidden_size, codebook_size=int(args.codebook_size), max_slots=int(args.max_slots)).to(device, dtype=model.dtype)
    graph_adapter = SoftGraphBiasCouncilAdapter(hidden_size).to(device, dtype=model.dtype)
    noun_adv = NounAdversary(hidden_size, max(2, int(args.hash_buckets))).to(device, dtype=model.dtype)
    cpc_head = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device, dtype=model.dtype)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt.get("codebook_state"), dict):
            codebook.load_state_dict(ckpt["codebook_state"], strict=False)

    params = list(codebook.parameters()) + list(emitter.parameters()) + list(graph_adapter.parameters()) + list(noun_adv.parameters()) + list(cpc_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr))

    semantic_ds = generate_dataset(size=int(args.semantic_dataset_size), seed=int(args.seed), profile=str(args.dataset_profile), difficulty_tier=str(args.difficulty_tier))
    semantic_train, semantic_val, semantic_test = split_dataset(semantic_ds)
    semantic_train = list(semantic_train) if semantic_train else list(semantic_val) if semantic_val else list(semantic_test)
    winograd_rows = _build_winograd_rows(size=int(args.winograd_pack_size), seed=int(args.seed), strict_balance=True)
    wino_train, wino_val, wino_eval, split_ids = _split_rows_by_pair(winograd_rows, seed=int(args.seed))

    telemetry: List[TrainTelemetry] = []

    for step in range(int(args.train_steps)):
        semantic_item = semantic_train[step % len(semantic_train)]
        wino_item = wino_train[step % len(wino_train)]
        cf_prompt, cf_candidates, cf_gold = _entity_counterfactual_from_row(wino_item)

        prompts = [str(semantic_item.prompt), str(wino_item["prompt"]), str(cf_prompt)]
        prompt_hiddens = [_extract_prompt_hidden(model, tokenizer, p) for p in prompts]
        logic_hiddens = [extract_trace_hidden_states(model, tokenizer, p, int(args.max_logic_new_tokens)).to(model.dtype) for p in prompts]

        emissions: List[dict[str, torch.Tensor]] = []
        noun_losses: List[torch.Tensor] = []
        noun_accs: List[float] = []
        for prompt, logic_hidden, prompt_hidden in zip(prompts, logic_hiddens, prompt_hiddens):
            emission = emitter(logic_hidden, prompt_hidden, codebook)
            emissions.append(emission)
            bucket = lexical_bucket(prompt, buckets=int(args.hash_buckets))
            if bucket is not None and float(args.grl_weight) > 0.0:
                target = torch.tensor([int(bucket)], device=device, dtype=torch.long)
                adv_logits = noun_adv(grad_reverse(emission["op_pre"], float(args.grl_scale)))
                noun_losses.append(F.cross_entropy(adv_logits, target))
                noun_accs.append(float((torch.argmax(adv_logits, dim=-1) == target).float().mean().item()))

        semantic_graph = _build_graph_batch(emissions[0])
        wino_graph = _build_graph_batch(emissions[1])
        cf_graph = _build_graph_batch(emissions[2])

        semantic_ce = teacher_forced_answer_ce(model, tokenizer, str(semantic_item.prompt), str(semantic_item.answer), int(args.layer_index), semantic_graph, graph_adapter, int(args.max_answer_tokens))
        wino_answer = str(wino_item["candidates"][int(wino_item["gold_index"])])
        winograd_ce = teacher_forced_answer_ce(model, tokenizer, str(wino_item["prompt"]), wino_answer, int(args.layer_index), wino_graph, graph_adapter, int(args.max_answer_tokens))
        task_loss = semantic_ce + winograd_ce

        slot_l1_loss = _slot_sparsity_loss(torch.cat([emissions[0]["slot_use_probs"], emissions[1]["slot_use_probs"], emissions[2]["slot_use_probs"]], dim=0))
        uniformity_loss = _uniformity_loss(torch.cat([e["op_probs"] for e in emissions], dim=0))
        grl_loss = torch.stack(noun_losses).mean() if noun_losses else torch.zeros((), device=device, dtype=model.dtype)
        invariance_loss = _op_invariance_loss(emissions[1]["op_logits"], emissions[2]["op_logits"])

        gold_emb = _answer_text_embedding(model, tokenizer, wino_item["candidates"][int(wino_item["gold_index"])])
        foil_emb = _answer_text_embedding(model, tokenizer, wino_item["candidates"][1 - int(wino_item["gold_index"])])
        op_proj = F.normalize(cpc_head(emissions[1]["op_embedding"]), p=2, dim=-1)
        pos = torch.sum(op_proj * gold_emb, dim=-1)
        neg = torch.sum(op_proj * foil_emb, dim=-1)
        cpc_loss = -torch.log(torch.softmax(torch.stack([pos, neg], dim=-1), dim=-1)[:, 0]).mean()

        total_loss = task_loss
        total_loss = total_loss + float(args.slot_sparsity_weight) * slot_l1_loss
        total_loss = total_loss + float(args.uniformity_weight) * uniformity_loss
        total_loss = total_loss + float(args.grl_weight) * grl_loss
        total_loss = total_loss + float(args.invariance_weight) * invariance_loss
        total_loss = total_loss + float(args.cpc_weight) * cpc_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        codebook.enforce_anchor_values()

        with torch.no_grad():
            batch_probs = torch.cat([e["op_probs"] for e in emissions], dim=0)
            p_mean = batch_probs.mean(dim=0)
            p_safe = torch.clamp(p_mean, min=1e-8)
            op_entropy = float((-torch.sum(p_safe * torch.log(p_safe))).item())
            top1 = float(torch.max(p_mean).item())
            w_acc = _binary_accuracy_from_candidates(model, tokenizer, str(wino_item["prompt"]), list(wino_item["candidates"]), int(wino_item["gold_index"]), int(args.layer_index), wino_graph, graph_adapter)
            telemetry.append(TrainTelemetry(
                step=step + 1,
                total_loss=float(total_loss.item()),
                task_loss=float(task_loss.item()),
                semantic_ce=float(semantic_ce.item()),
                winograd_ce=float(winograd_ce.item()),
                slot_l1_loss=float(slot_l1_loss.item()),
                uniformity_loss=float(uniformity_loss.item()),
                grl_loss=float(grl_loss.item()),
                invariance_loss=float(invariance_loss.item()),
                cpc_loss=float(cpc_loss.item()),
                slot_usage_mean=float(torch.cat([e["slot_use_probs"] for e in emissions], dim=0).mean().item()),
                operator_entropy_batch=op_entropy,
                top1_op_share_batch=top1,
                winograd_accuracy_batch=float(w_acc),
            ))

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{int(args.train_steps)} | total={float(total_loss.item()):.4f} task={float(task_loss.item()):.4f} sem={float(semantic_ce.item()):.4f} wino={float(winograd_ce.item()):.4f} slot={float(slot_l1_loss.item()):.4f} uni={float(uniformity_loss.item()):.4f} grl={float(grl_loss.item()):.4f} inv={float(invariance_loss.item()):.4f} cpc={float(cpc_loss.item()):.4f}")

    def eval_winograd(rows: list[dict[str, Any]]) -> dict[str, float]:
        accs: List[float] = []
        for row in rows[: min(64, len(rows))]:
            prompt_hidden = _extract_prompt_hidden(model, tokenizer, str(row["prompt"]))
            logic_hidden = extract_trace_hidden_states(model, tokenizer, str(row["prompt"]), int(args.max_logic_new_tokens)).to(model.dtype)
            emission = emitter(logic_hidden, prompt_hidden, codebook)
            graph = _build_graph_batch(emission)
            accs.append(_binary_accuracy_from_candidates(model, tokenizer, str(row["prompt"]), list(row["candidates"]), int(row["gold_index"]), int(args.layer_index), graph, graph_adapter))
        return {"n": int(min(64, len(rows))), "accuracy": float(sum(accs) / max(1, len(accs)))}

    def eval_semantic(rows: list[Any]) -> dict[str, float]:
        hits = 0
        ce_hist: List[float] = []
        subset = rows[: min(32, len(rows))]
        for row in subset:
            prompt = str(row.prompt)
            answer = str(row.answer)
            prompt_hidden = _extract_prompt_hidden(model, tokenizer, prompt)
            logic_hidden = extract_trace_hidden_states(model, tokenizer, prompt, int(args.max_logic_new_tokens)).to(model.dtype)
            emission = emitter(logic_hidden, prompt_hidden, codebook)
            graph = _build_graph_batch(emission)
            ce = teacher_forced_answer_ce(model, tokenizer, prompt, answer, int(args.layer_index), graph, graph_adapter, int(args.max_answer_tokens))
            pred = greedy_answer(model, tokenizer, prompt, int(args.layer_index), graph, graph_adapter, int(args.max_answer_tokens))
            ce_hist.append(float(ce.detach().item()))
            if answer.strip().lower() == pred.strip().lower():
                hits += 1
        return {"n": int(len(subset)), "accuracy": float(hits / max(1, len(subset))), "mean_ce": float(sum(ce_hist) / max(1, len(ce_hist)))}

    wino_val_metrics = eval_winograd(wino_val)
    wino_eval_metrics = eval_winograd(wino_eval)
    semantic_val_metrics = eval_semantic(semantic_val if semantic_val else semantic_test)

    ckpt_path = run_dir / "m5_padded_nary_checkpoint.pt"
    torch.save({
        "codebook_state": codebook.state_dict(),
        "emitter_state": emitter.state_dict(),
        "graph_adapter_state": graph_adapter.state_dict(),
        "noun_adversary_state": noun_adv.state_dict(),
        "cpc_head_state": cpc_head.state_dict(),
    }, ckpt_path)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M5.padded_nary", "scripts/train_m5_padded_nary.py"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {
            "semantic_profile": str(args.dataset_profile),
            "semantic_train": int(len(semantic_train)),
            "semantic_val": int(len(semantic_val)),
            "semantic_test": int(len(semantic_test)),
            "winograd_train": int(len(wino_train)),
            "winograd_val": int(len(wino_val)),
            "winograd_eval": int(len(wino_eval)),
            "winograd_pair_ids": split_ids,
        },
        "final_step": asdict(telemetry[-1]) if telemetry else None,
        "eval": {
            "winograd_val": wino_val_metrics,
            "winograd_eval": wino_eval_metrics,
            "semantic_val": semantic_val_metrics,
        },
        "steps": [asdict(t) for t in telemetry],
    }
    summary_path = run_dir / "m5_padded_nary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {ckpt_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
