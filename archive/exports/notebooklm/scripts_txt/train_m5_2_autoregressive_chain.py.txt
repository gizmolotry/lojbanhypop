from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from build_m3_15_winograd_pack import _build_row, _specs, _validate_rows
from lojban_evolution.experiment import generate_dataset, split_dataset
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata
from train_h5_persistent_vq_advisor import BooleanAnchorTable, adapter_disabled, build_final_prefix

PAD_TOKEN_ID = 2000
VAR_TOKEN_BASE = 2001


@dataclass
class TrainTelemetry:
    step: int
    total_loss: float
    task_loss: float
    semantic_ce: float
    winograd_ce: float
    slot_loss: float
    stop_loss: float
    diversity_loss: float
    operator_entropy_batch: float
    top1_op_share_batch: float
    slot_usage_mean: float
    mean_chain_length: float
    mean_stop_prob: float
    winograd_accuracy_batch: float


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


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


def _load_backbone(base_model: str, adapter: Path, local_files_only: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        local_files_only=local_files_only,
        device_map="auto",
    )
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        backbone,
        str(adapter),
        local_files_only=local_files_only,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def _extract_prompt_hidden(model, tokenizer, prompt: str, max_tokens: int = 256) -> torch.Tensor:
    device = _model_device(model)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_tokens))
    ids = enc.input_ids.to(device)
    mask = torch.ones_like(ids)
    with adapter_disabled(model):
        out = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    return out.hidden_states[-1].detach()


def extract_trace_hidden_states(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    prompt = (
        "You are a rigid symbolic reasoner. "
        "Emit a matrix-chain TRACE before the final answer.\n\n"
        f"QUESTION: {question}\nTRACE:"
    )
    device = _model_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc.input_ids.to(device)
    with adapter_disabled(model):
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden = out.hidden_states[-1].detach()
    return hidden[:, -min(hidden.shape[1], int(max_logic_new_tokens)) :, :]


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        op_embedding: torch.Tensor,
        slot_tensors: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query(hidden_states)
        k = self.slot_key(slot_tensors)
        v = self.slot_val(slot_tensors)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
        expanded_mask = slot_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        scores = scores.masked_fill(~expanded_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(expanded_mask, attn, torch.zeros_like(attn))
        denom = torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-8)
        attn = attn / denom
        slot_ctx = torch.matmul(attn, v)
        op_ctx = self.op_proj(op_embedding).unsqueeze(1)
        fused = torch.tanh(op_ctx + self.slot_proj(slot_ctx) + self.hidden_proj(hidden_states))
        return self.out_proj(fused) * torch.sigmoid(self.gate) * self.scale


class AutoregressiveMatrixChain(torch.nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int = 2000, max_slots: int = 10, max_chain_steps: int = 4):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.codebook_size = int(codebook_size)
        self.max_slots = int(max_slots)
        self.max_chain_steps = int(max_chain_steps)
        self.init_proj = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.prompt_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_queries = torch.nn.Parameter(torch.randn(self.max_slots - 1, hidden_size) * 0.02)
        self.slot_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.op_pre = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_gate = torch.nn.Linear(hidden_size, 1, bias=True)
        self.stop_head = torch.nn.Linear(hidden_size * 2, 1, bias=True)
        self.recur = torch.nn.GRUCell(hidden_size, hidden_size)

    def _attend(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(float(queries.shape[-1]))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values), weights

    def forward(self, logic_hidden: torch.Tensor, prompt_hidden: torch.Tensor, codebook: BooleanAnchorTable) -> dict[str, Any]:
        batch_size = logic_hidden.shape[0]
        prompt_summary = prompt_hidden.mean(dim=1)
        logic_summary = logic_hidden.mean(dim=1)
        state = torch.tanh(self.init_proj(torch.cat([prompt_summary, logic_summary], dim=-1)))
        prompt_k = self.prompt_k(prompt_hidden)
        prompt_v = self.prompt_v(prompt_hidden)

        emissions: list[dict[str, torch.Tensor]] = []
        summaries: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        stop_probs: list[torch.Tensor] = []

        for _ in range(self.max_chain_steps):
            state_q = self.prompt_q(state).unsqueeze(1)
            state_ctx, _ = self._attend(state_q, prompt_k, prompt_v)
            state_ctx = state_ctx.squeeze(1)

            op_pre = self.op_pre(state_ctx)
            diff = op_pre.unsqueeze(1) - codebook.emb.unsqueeze(0)
            dist = torch.sum(diff * diff, dim=-1)
            op_logits = -dist
            op_probs = torch.softmax(op_logits, dim=-1)
            op_idx = torch.argmax(op_logits, dim=-1)
            op_embedding = codebook.emb[op_idx]

            slot_seed = state_ctx.unsqueeze(1) + self.slot_queries.unsqueeze(0)
            slot_q = self.slot_q(slot_seed)
            slot_tensors, _ = self._attend(slot_q, prompt_k, prompt_v)
            slot_use_logits = self.slot_gate(slot_seed).squeeze(-1)
            slot_use_probs = torch.sigmoid(slot_use_logits)

            slot_indices = (
                torch.arange(self.max_slots - 1, device=slot_use_probs.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            slot_token_ids = torch.where(
                slot_use_probs >= 0.5,
                slot_indices + int(VAR_TOKEN_BASE),
                torch.full_like(slot_indices, int(PAD_TOKEN_ID)),
            )
            slot_mask = slot_token_ids != int(PAD_TOKEN_ID)
            if int(slot_mask.sum().item()) <= 0:
                top_idx = torch.argmax(slot_use_probs, dim=-1, keepdim=True)
                slot_mask = torch.zeros_like(slot_use_probs, dtype=torch.bool)
                slot_mask.scatter_(1, top_idx, True)
                slot_token_ids = torch.where(
                    slot_mask,
                    slot_indices + int(VAR_TOKEN_BASE),
                    torch.full_like(slot_indices, int(PAD_TOKEN_ID)),
                )
            matrix_token_ids = torch.cat([op_idx.unsqueeze(-1), slot_token_ids], dim=-1)

            slot_mask_f = slot_mask.to(slot_tensors.dtype)
            slot_summary = torch.sum(slot_tensors * slot_mask_f.unsqueeze(-1), dim=1) / torch.clamp(
                slot_mask_f.sum(dim=-1, keepdim=True), min=1.0
            )
            matrix_summary = torch.tanh(op_embedding + slot_summary)
            stop_input = torch.cat([state_ctx, matrix_summary], dim=-1)
            stop_logit = self.stop_head(stop_input).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logit)

            emissions.append(
                {
                    "op_logits": op_logits,
                    "op_probs": op_probs,
                    "op_idx": op_idx,
                    "op_embedding": op_embedding,
                    "slot_tensors": slot_tensors,
                    "slot_use_probs": slot_use_probs,
                    "slot_token_ids": slot_token_ids,
                    "slot_mask": slot_mask,
                    "matrix_token_ids": matrix_token_ids,
                    "matrix_summary": matrix_summary,
                }
            )
            summaries.append(matrix_summary)
            stop_logits.append(stop_logit)
            stop_probs.append(stop_prob)
            state = self.recur(matrix_summary, state)

        stop_logits_t = torch.stack(stop_logits, dim=1)
        stop_probs_t = torch.stack(stop_probs, dim=1)
        summary_stack = torch.stack(summaries, dim=1)
        stop_hits = stop_probs_t >= 0.5
        chain_lengths = torch.argmax(stop_hits.to(torch.int64), dim=1) + 1
        chain_lengths = torch.where(
            stop_hits.sum(dim=1) == 0,
            torch.full_like(chain_lengths, self.max_chain_steps),
            chain_lengths,
        )
        return {
            "emissions": emissions,
            "stop_logits": stop_logits_t,
            "stop_probs": stop_probs_t,
            "summary_stack": summary_stack,
            "chain_lengths": chain_lengths,
        }


class MultiMatrixGraphAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.base = SoftGraphBiasCouncilAdapter(hidden_size)

    def build_graph_batch(self, chain_out: dict[str, Any]) -> dict[str, torch.Tensor]:
        emissions = chain_out["emissions"]
        chain_lengths = chain_out["chain_lengths"]
        op_embeddings = torch.stack([e["op_embedding"] for e in emissions], dim=1)
        step_mask = (
            torch.arange(op_embeddings.shape[1], device=op_embeddings.device).unsqueeze(0)
            < chain_lengths.unsqueeze(1)
        )
        step_mask_f = step_mask.to(op_embeddings.dtype)
        op_embedding = torch.sum(op_embeddings * step_mask_f.unsqueeze(-1), dim=1) / torch.clamp(
            step_mask_f.sum(dim=1, keepdim=True), min=1.0
        )
        slot_tensors = torch.cat([e["slot_tensors"] for e in emissions], dim=1)
        slot_mask = torch.cat(
            [e["slot_mask"] & step_mask[:, i : i + 1] for i, e in enumerate(emissions)],
            dim=1,
        )
        if int(slot_mask.sum().item()) <= 0:
            slot_mask[:, 0] = True
        matrix_token_ids = torch.cat([e["matrix_token_ids"] for e in emissions], dim=1)
        return {
            "op_embedding": op_embedding,
            "slot_tensors": slot_tensors,
            "slot_mask": slot_mask,
            "matrix_token_ids": matrix_token_ids,
        }


class _LayerHook:
    def __init__(self, model, layer_index: int, adapter: MultiMatrixGraphAdapter, graph_batch: dict[str, torch.Tensor], scale: float = 1.0):
        self.model = model
        self.layer_index = int(layer_index)
        self.adapter = adapter
        self.graph_batch = graph_batch
        self.scale = float(scale)
        self.handle = None

    def __enter__(self):
        layers = _resolve_layers(self.model)
        if layers is None:
            raise RuntimeError("Unable to locate decoder layers for graph hook.")

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            delta = self.adapter.base(
                hidden_states=hidden,
                op_embedding=self.graph_batch["op_embedding"],
                slot_tensors=self.graph_batch["slot_tensors"],
                slot_mask=self.graph_batch["slot_mask"],
            ).to(dtype=hidden.dtype, device=hidden.device)
            hidden = hidden + (self.scale * delta)
            return (hidden, *rest) if rest is not None else hidden

        self.handle = layers[self.layer_index].register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


def _build_final_prefix_inputs(tokenizer, prompt: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(device)
    return ids, torch.ones_like(ids)


def _answer_token_ids(tokenizer, answer: str, device: torch.device, max_tokens: int) -> torch.Tensor:
    return tokenizer(" " + str(answer), return_tensors="pt", add_special_tokens=False).input_ids.to(device)[:, : int(max_tokens)]


def teacher_forced_answer_ce(model, tokenizer, prompt: str, answer: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: MultiMatrixGraphAdapter, max_answer_tokens: int) -> torch.Tensor:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    tgt = _answer_token_ids(tokenizer, answer, device, max_tokens=max_answer_tokens)
    ce = torch.zeros((), device=device, dtype=cur_emb.dtype)
    with adapter_disabled(model):
        for t in range(tgt.shape[1]):
            with _LayerHook(model, int(layer_index), adapter, graph_batch, 1.0):
                out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
            ce = ce + F.cross_entropy(out.logits[:, -1, :], tgt[:, t])
            next_emb = model.get_input_embeddings()(tgt[:, t : t + 1])
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype, device=am.device)], dim=1)
    return ce / max(1, tgt.shape[1])


def _candidate_first_token_id(tokenizer, candidate: str) -> int:
    ids = tokenizer(" " + str(candidate), add_special_tokens=False, return_tensors="pt").input_ids
    if int(ids.numel()) == 0:
        return int(tokenizer.eos_token_id or 0)
    return int(ids[0, 0].item())


def score_candidate_first_token(model, tokenizer, prompt: str, candidate: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: MultiMatrixGraphAdapter) -> torch.Tensor:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    tok_id = _candidate_first_token_id(tokenizer, candidate)
    with _LayerHook(model, int(layer_index), adapter, graph_batch, 1.0):
        out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
    return torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _binary_accuracy_from_candidates(model, tokenizer, prompt: str, candidates: list[str], gold_index: int, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: MultiMatrixGraphAdapter) -> float:
    s0 = float(score_candidate_first_token(model, tokenizer, prompt, candidates[0], layer_index, graph_batch, adapter).detach().item())
    s1 = float(score_candidate_first_token(model, tokenizer, prompt, candidates[1], layer_index, graph_batch, adapter).detach().item())
    return 1.0 if (0 if s0 >= s1 else 1) == int(gold_index) else 0.0


def greedy_answer(model, tokenizer, prompt: str, layer_index: int, graph_batch: dict[str, torch.Tensor], adapter: MultiMatrixGraphAdapter, max_answer_tokens: int) -> str:
    device = _model_device(model)
    p_ids, am = _build_final_prefix_inputs(tokenizer, prompt, device)
    cur_emb = model.get_input_embeddings()(p_ids)
    toks: List[int] = []
    with adapter_disabled(model):
        for _ in range(int(max_answer_tokens)):
            with _LayerHook(model, int(layer_index), adapter, graph_batch, 1.0):
                out = model(inputs_embeds=cur_emb, attention_mask=am, return_dict=True, use_cache=False)
            tok = int(torch.argmax(out.logits[:, -1, :], dim=-1)[0].item())
            if tokenizer.eos_token_id is not None and tok == int(tokenizer.eos_token_id):
                break
            toks.append(tok)
            cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(torch.tensor([[tok]], device=device))], dim=1)
            am = torch.cat([am, torch.ones((1, 1), dtype=am.dtype, device=am.device)], dim=1)
    return tokenizer.decode(toks, skip_special_tokens=True).strip()


def _build_winograd_rows(size: int, seed: int, strict_balance: bool = True) -> List[dict[str, Any]]:
    rng = random.Random(int(seed))
    specs = _specs()
    rows: List[dict[str, Any]] = []
    pair_idx = 0
    target_even = int(size) if int(size) % 2 == 0 else int(size) - 1
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
    return train_rows, val_rows, eval_rows, {
        "train_pair_ids": sorted(train_ids),
        "val_pair_ids": sorted(val_ids),
        "eval_pair_ids": sorted(eval_ids),
    }


def _slot_sparsity_loss(emissions: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    probs = torch.cat([e["slot_use_probs"] for e in emissions], dim=0)
    pos = torch.arange(1, probs.shape[-1] + 1, device=probs.device, dtype=probs.dtype)
    pos = pos / pos.sum()
    l1 = torch.sum(probs * pos.unsqueeze(0))
    monotonic = torch.relu(probs[:, 1:] - probs[:, :-1]).mean() if probs.shape[-1] > 1 else torch.zeros((), device=probs.device, dtype=probs.dtype)
    min_active = torch.relu(1.0 - probs.sum(dim=-1)).mean()
    return l1 + monotonic + min_active


def _stop_supervision_loss(stop_logits: torch.Tensor, target_steps: torch.Tensor) -> torch.Tensor:
    batch_size, steps = stop_logits.shape
    idx = torch.arange(steps, device=stop_logits.device).unsqueeze(0).expand(batch_size, -1)
    target = (idx == target_steps.unsqueeze(1)).to(stop_logits.dtype)
    return F.binary_cross_entropy_with_logits(stop_logits, target)


def _chain_diversity_loss(summary_stack: torch.Tensor) -> torch.Tensor:
    if summary_stack.shape[1] <= 1:
        return torch.zeros((), device=summary_stack.device, dtype=summary_stack.dtype)
    a = F.normalize(summary_stack[:, :-1, :], p=2, dim=-1)
    b = F.normalize(summary_stack[:, 1:, :], p=2, dim=-1)
    similarity = torch.sum(a * b, dim=-1)
    return torch.relu(similarity - 0.80).mean()


def _target_chain_length_from_trace(trace: Sequence[str], max_chain_steps: int) -> int:
    if not trace:
        return min(2, int(max_chain_steps))
    return max(2, min(int(max_chain_steps), math.ceil(len(trace) / 3)))


def _collect_batch_probs(*chains: dict[str, Any]) -> torch.Tensor:
    return torch.cat([e["op_probs"] for chain in chains for e in chain["emissions"]], dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bounded autoregressive M5.2 matrix-chain.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/l_series/m5_2_autoregressive_chain"))
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--semantic-dataset-size", type=int, default=96)
    p.add_argument("--winograd-pack-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--max-slots", type=int, default=10)
    p.add_argument("--codebook-size", type=int, default=2000)
    p.add_argument("--max-chain-steps", type=int, default=4)
    p.add_argument("--slot-sparsity-weight", type=float, default=0.25)
    p.add_argument("--stop-weight", type=float, default=0.20)
    p.add_argument("--diversity-weight", type=float, default=0.10)
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
    chain = AutoregressiveMatrixChain(
        hidden_size,
        codebook_size=int(args.codebook_size),
        max_slots=int(args.max_slots),
        max_chain_steps=int(args.max_chain_steps),
    ).to(device, dtype=model.dtype)
    graph_adapter = MultiMatrixGraphAdapter(hidden_size).to(device, dtype=model.dtype)
    opt = torch.optim.AdamW(
        list(codebook.parameters()) + list(chain.parameters()) + list(graph_adapter.parameters()),
        lr=float(args.lr),
    )

    semantic_ds = generate_dataset(
        size=int(args.semantic_dataset_size),
        seed=int(args.seed),
        profile=str(args.dataset_profile),
        difficulty_tier=str(args.difficulty_tier),
    )
    semantic_train, semantic_val_rows, semantic_test = split_dataset(semantic_ds)
    semantic_train = list(semantic_train) if semantic_train else list(semantic_val_rows) if semantic_val_rows else list(semantic_test)
    winograd_rows = _build_winograd_rows(size=int(args.winograd_pack_size), seed=int(args.seed), strict_balance=True)
    wino_train, wino_val, wino_eval, split_ids = _split_rows_by_pair(winograd_rows, seed=int(args.seed))

    telemetry: list[TrainTelemetry] = []
    for step in range(int(args.train_steps)):
        semantic_item = semantic_train[step % len(semantic_train)]
        wino_item = wino_train[step % len(wino_train)]

        semantic_prompt_hidden = _extract_prompt_hidden(model, tokenizer, str(semantic_item.prompt))
        wino_prompt_hidden = _extract_prompt_hidden(model, tokenizer, str(wino_item["prompt"]))
        semantic_logic_hidden = extract_trace_hidden_states(model, tokenizer, str(semantic_item.prompt), int(args.max_logic_new_tokens)).to(model.dtype)
        wino_logic_hidden = extract_trace_hidden_states(model, tokenizer, str(wino_item["prompt"]), int(args.max_logic_new_tokens)).to(model.dtype)

        semantic_chain = chain(semantic_logic_hidden, semantic_prompt_hidden, codebook)
        wino_chain = chain(wino_logic_hidden, wino_prompt_hidden, codebook)
        semantic_graph = graph_adapter.build_graph_batch(semantic_chain)
        wino_graph = graph_adapter.build_graph_batch(wino_chain)

        semantic_ce = teacher_forced_answer_ce(
            model,
            tokenizer,
            str(semantic_item.prompt),
            str(semantic_item.answer),
            int(args.layer_index),
            semantic_graph,
            graph_adapter,
            int(args.max_answer_tokens),
        )
        wino_answer = str(wino_item["candidates"][int(wino_item["gold_index"])])
        winograd_ce = teacher_forced_answer_ce(
            model,
            tokenizer,
            str(wino_item["prompt"]),
            wino_answer,
            int(args.layer_index),
            wino_graph,
            graph_adapter,
            int(args.max_answer_tokens),
        )
        task_loss = semantic_ce + winograd_ce

        semantic_target = torch.tensor(
            [_target_chain_length_from_trace(tuple(getattr(semantic_item, "trace", ())), int(args.max_chain_steps)) - 1],
            device=device,
            dtype=torch.long,
        )
        wino_target = torch.tensor([1], device=device, dtype=torch.long)
        stop_loss = _stop_supervision_loss(semantic_chain["stop_logits"], semantic_target) + _stop_supervision_loss(
            wino_chain["stop_logits"], wino_target
        )
        diversity_loss = _chain_diversity_loss(semantic_chain["summary_stack"]) + _chain_diversity_loss(wino_chain["summary_stack"])
        slot_loss = _slot_sparsity_loss(semantic_chain["emissions"]) + _slot_sparsity_loss(wino_chain["emissions"])

        total_loss = (
            task_loss
            + float(args.slot_sparsity_weight) * slot_loss
            + float(args.stop_weight) * stop_loss
            + float(args.diversity_weight) * diversity_loss
        )

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        codebook.enforce_anchor_values()

        with torch.no_grad():
            batch_probs = _collect_batch_probs(semantic_chain, wino_chain)
            p_mean = torch.clamp(batch_probs.mean(dim=0), min=1e-8)
            op_entropy = float((-torch.sum(p_mean * torch.log(p_mean))).item())
            top1 = float(torch.max(p_mean).item())
            batch_acc = _binary_accuracy_from_candidates(
                model,
                tokenizer,
                str(wino_item["prompt"]),
                list(wino_item["candidates"]),
                int(wino_item["gold_index"]),
                int(args.layer_index),
                wino_graph,
                graph_adapter,
            )
            slot_usage_mean = float(
                torch.cat([e["slot_use_probs"] for e in semantic_chain["emissions"] + wino_chain["emissions"]], dim=0)
                .mean()
                .item()
            )
            mean_chain_length = float(
                torch.cat([semantic_chain["chain_lengths"], wino_chain["chain_lengths"]], dim=0).float().mean().item()
            )
            mean_stop_prob = float(
                torch.cat([semantic_chain["stop_probs"], wino_chain["stop_probs"]], dim=0).mean().item()
            )
            telemetry.append(
                TrainTelemetry(
                    step=step + 1,
                    total_loss=float(total_loss.item()),
                    task_loss=float(task_loss.item()),
                    semantic_ce=float(semantic_ce.item()),
                    winograd_ce=float(winograd_ce.item()),
                    slot_loss=float(slot_loss.item()),
                    stop_loss=float(stop_loss.item()),
                    diversity_loss=float(diversity_loss.item()),
                    operator_entropy_batch=op_entropy,
                    top1_op_share_batch=top1,
                    slot_usage_mean=slot_usage_mean,
                    mean_chain_length=mean_chain_length,
                    mean_stop_prob=mean_stop_prob,
                    winograd_accuracy_batch=float(batch_acc),
                )
            )

    def eval_winograd(rows: list[dict[str, Any]]) -> dict[str, float]:
        accs: list[float] = []
        lengths: list[float] = []
        subset = rows[: min(32, len(rows))]
        for row in subset:
            prompt_hidden = _extract_prompt_hidden(model, tokenizer, str(row["prompt"]))
            logic_hidden = extract_trace_hidden_states(model, tokenizer, str(row["prompt"]), int(args.max_logic_new_tokens)).to(model.dtype)
            out = chain(logic_hidden, prompt_hidden, codebook)
            graph = graph_adapter.build_graph_batch(out)
            accs.append(
                _binary_accuracy_from_candidates(
                    model,
                    tokenizer,
                    str(row["prompt"]),
                    list(row["candidates"]),
                    int(row["gold_index"]),
                    int(args.layer_index),
                    graph,
                    graph_adapter,
                )
            )
            lengths.append(float(out["chain_lengths"].float().mean().item()))
        return {
            "n": int(len(subset)),
            "accuracy": float(sum(accs) / max(1, len(accs))),
            "mean_chain_length": float(sum(lengths) / max(1, len(lengths))),
        }

    def eval_semantic(rows: list[Any]) -> dict[str, float]:
        hits = 0
        lengths: list[float] = []
        subset = rows[: min(16, len(rows))]
        for row in subset:
            prompt = str(row.prompt)
            answer = str(row.answer)
            prompt_hidden = _extract_prompt_hidden(model, tokenizer, prompt)
            logic_hidden = extract_trace_hidden_states(model, tokenizer, prompt, int(args.max_logic_new_tokens)).to(model.dtype)
            out = chain(logic_hidden, prompt_hidden, codebook)
            graph = graph_adapter.build_graph_batch(out)
            pred = greedy_answer(model, tokenizer, prompt, int(args.layer_index), graph, graph_adapter, int(args.max_answer_tokens))
            if pred.strip().lower() == answer.strip().lower():
                hits += 1
            lengths.append(float(out["chain_lengths"].float().mean().item()))
        return {
            "n": int(len(subset)),
            "accuracy": float(hits / max(1, len(subset))),
            "mean_chain_length": float(sum(lengths) / max(1, len(lengths))),
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M5.2.autoregressive_chain", "scripts/train_m5_2_autoregressive_chain.py"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {
            "semantic_profile": str(args.dataset_profile),
            "semantic_train": int(len(semantic_train)),
            "semantic_val": int(len(semantic_val_rows)),
            "semantic_test": int(len(semantic_test)),
            "winograd_train": int(len(wino_train)),
            "winograd_val": int(len(wino_val)),
            "winograd_eval": int(len(wino_eval)),
            "winograd_pair_ids": split_ids,
        },
        "final_step": asdict(telemetry[-1]) if telemetry else None,
        "eval": {
            "winograd_val": eval_winograd(wino_val),
            "winograd_eval": eval_winograd(wino_eval),
            "semantic_val": eval_semantic(semantic_val_rows if semantic_val_rows else semantic_test),
        },
        "steps": [asdict(x) for x in telemetry],
    }

    ckpt_path = run_dir / "m5_2_autoregressive_chain_checkpoint.pt"
    torch.save(
        {
            "codebook_state": codebook.state_dict(),
            "chain_state": chain.state_dict(),
            "graph_adapter_state": graph_adapter.state_dict(),
        },
        ckpt_path,
    )
    summary_path = run_dir / "m5_2_autoregressive_chain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {ckpt_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
