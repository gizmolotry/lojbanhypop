from __future__ import annotations

import random
from collections import defaultdict
from inspect import signature
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lojban_evolution.bridi_substrate import (
    assert_symbolic_trace_contract,
    budget_packed_trace_symbols,
    pack_symbolic_trace_from_batch,
    pack_symbolic_trace_from_outputs,
    packed_trace_component_accuracy,
    packed_trace_exact_accuracy,
    packed_trace_symbol_counts,
    packed_trace_spec,
    random_packed_trace_like,
    shuffled_packed_trace_like,
    zero_packed_trace_like,
)
from lojban_evolution.m21.bridi import ANSWER_LABELS, CMAVO, DEFAULT_MAX_ENTITIES, DEFAULT_MAX_PLACES, GISMU, tokenize
from lojban_evolution.m23.relevance import M23RelevanceDataset, M23RelevanceExample, m23_collate, train_m23_relevance_router


DEFAULT_M24_MDL_WEIGHT = 0.01


M24_LOCKS: dict[str, str] = {
    "substrate_first": "the generator is reused from M23/M21 dynamic bridi and is frozen before advisor training",
    "symbolic_trace_only": "the advisor consumes packed integer bridi traces, not frame_repr/trace_state/prompt_state",
    "compression_controls": "predicted, oracle, shuffled, random, zero/no-trace, and prompt-only paths are evaluated side-by-side",
}

DISALLOWED_ADVISOR_TRACE_INPUTS = ("frame_repr", "trace_state", "prompt_state")


def m24_2_promotion_gate_metrics(
    *,
    hard_bottleneck_configured: bool,
    strict_accuracy: float,
    predicted_vs_shuffled_delta: float,
    predicted_vs_random_delta: float,
    hard_bottleneck_trace_accuracy: float,
    effective_packed_symbol_to_prompt_ratio: float,
    symbol_budget_respected: bool,
    advisor_vs_prompt_delta: float = 0.0,
) -> dict[str, float]:
    """Return per-run M24.2 promotion gates as float metrics."""

    gates = {
        "hard_bottleneck_configured": bool(hard_bottleneck_configured),
        "strict_accuracy_retained": float(strict_accuracy) >= 0.68,
        "trace_beats_shuffled_strong": float(predicted_vs_shuffled_delta) >= 0.60,
        "trace_beats_random_strong": float(predicted_vs_random_delta) >= 0.60,
        "trace_exact_floor": float(hard_bottleneck_trace_accuracy) >= 0.05,
        "token_reduction_positive": float(effective_packed_symbol_to_prompt_ratio) < 1.0,
        "symbol_budget_respected": bool(symbol_budget_respected),
    }
    pass_rate = sum(1.0 for passed in gates.values() if passed) / max(1, len(gates))
    return {
        "m24_2_promotion_gate_pass_rate": pass_rate,
        "m24_2_promotion_candidate": 1.0 if all(gates.values()) else 0.0,
        "m24_2_gate_hard_bottleneck_configured": 1.0 if gates["hard_bottleneck_configured"] else 0.0,
        "m24_2_gate_strict_accuracy_retained": 1.0 if gates["strict_accuracy_retained"] else 0.0,
        "m24_2_gate_trace_beats_shuffled_strong": 1.0 if gates["trace_beats_shuffled_strong"] else 0.0,
        "m24_2_gate_trace_beats_random_strong": 1.0 if gates["trace_beats_random_strong"] else 0.0,
        "m24_2_gate_trace_exact_floor": 1.0 if gates["trace_exact_floor"] else 0.0,
        "m24_2_gate_symbol_budget_respected": 1.0 if gates["symbol_budget_respected"] else 0.0,
        "m24_2_gate_hard_trace_beats_random": 1.0 if gates["trace_beats_random_strong"] else 0.0,
        "m24_2_gate_hard_trace_beats_prompt_only": 1.0 if float(advisor_vs_prompt_delta) >= 0.0 else 0.0,
        "m24_2_gate_token_reduction_positive": 1.0 if gates["token_reduction_positive"] else 0.0,
    }


class PackedTraceAdvisor(nn.Module):
    """Trace-only answer advisor over packed integer bridi symbols."""

    primary_trace_input = "packed_symbolic_trace"
    disallowed_primary_inputs = DISALLOWED_ADVISOR_TRACE_INPUTS

    def __init__(
        self,
        *,
        max_frames: int = 6,
        max_places: int = DEFAULT_MAX_PLACES,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        hidden_dim: int = 64,
        active_frame_budget: int | None = None,
        trace_symbol_budget: int | None = None,
    ) -> None:
        super().__init__()
        self.max_frames = int(max_frames)
        self.max_places = int(max_places)
        self.max_entities = int(max_entities)
        if active_frame_budget is not None and int(active_frame_budget) < 0:
            raise ValueError("active_frame_budget must be non-negative or None.")
        if trace_symbol_budget is not None and int(trace_symbol_budget) < 0:
            raise ValueError("trace_symbol_budget must be non-negative or None.")
        self.active_frame_budget = int(active_frame_budget) if active_frame_budget is not None and int(active_frame_budget) > 0 else None
        self.trace_symbol_budget = int(trace_symbol_budget) if trace_symbol_budget is not None and int(trace_symbol_budget) > 0 else None
        self.spec = packed_trace_spec(max_frames=self.max_frames, max_places=self.max_places)
        dim = int(hidden_dim)
        self.active_embed = nn.Embedding(2, dim)
        self.stop_embed = nn.Embedding(2, dim)
        self.gismu_embed = nn.Embedding(len(GISMU), dim)
        self.cmavo_bit_embed = nn.Embedding(2, dim)
        self.judri_embed = nn.Embedding(self.max_entities + 1, dim)
        self.frame_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh())
        self.answer_head = nn.Linear(dim, len(ANSWER_LABELS))

    def forward(self, packed_trace: torch.Tensor) -> torch.Tensor:
        assert_symbolic_trace_contract(packed_trace)
        packed_trace = budget_packed_trace_symbols(
            packed_trace,
            symbol_budget=self.trace_symbol_budget,
            active_frame_budget=self.active_frame_budget,
        )
        x = packed_trace.long()
        if x.shape[1] != self.max_frames:
            x = x[:, : self.max_frames]
        active_ids = x[..., self.spec.active_col].clamp(0, 1)
        stop_ids = x[..., self.spec.stop_col].clamp(0, 1)
        gismu_ids = x[..., self.spec.gismu_col].clamp(0, len(GISMU) - 1)
        cmavo_bits = x[..., self.spec.cmavo_start : self.spec.cmavo_start + len(CMAVO)].clamp(0, 1)
        judri_ids = x[..., self.spec.judri_start : self.spec.judri_start + self.max_places].clamp(0, self.max_entities)
        cmavo_state = self.cmavo_bit_embed(cmavo_bits).sum(dim=-2)
        if judri_ids.shape[-1] > 0:
            judri_state = self.judri_embed(judri_ids).mean(dim=-2)
        else:
            judri_state = self.gismu_embed(gismu_ids).new_zeros(self.gismu_embed(gismu_ids).shape)
        frame_state = (
            self.active_embed(active_ids)
            + self.stop_embed(stop_ids)
            + self.gismu_embed(gismu_ids)
            + cmavo_state
            + judri_state
        )
        active_mask = active_ids.float().unsqueeze(-1)
        encoded = self.frame_mlp(frame_state) * active_mask
        trace_summary = encoded.sum(dim=1) / active_mask.sum(dim=1).clamp_min(1.0)
        return self.answer_head(trace_summary)


class PromptOnlyControl(nn.Module):
    """Prompt-only bag-of-words control, deliberately separate from trace advisor."""

    def __init__(self, *, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.classifier = nn.Sequential(nn.Linear(int(embedding_dim), int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), len(ANSWER_LABELS)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(input_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    return float((torch.argmax(logits, dim=-1) == target).float().mean().detach().cpu().item())


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _advisor_symbolic_contract_status(advisor: PackedTraceAdvisor) -> dict[str, bool]:
    params = list(signature(advisor.forward).parameters)
    forbidden = set(getattr(advisor, "disallowed_primary_inputs", ()))
    uses_only_packed = params == ["packed_trace"]
    primary_symbolic = getattr(advisor, "primary_trace_input", "") == "packed_symbolic_trace"
    no_forbidden_params = not any(name in forbidden for name in params)
    return {
        "uses_only_packed_trace_param": uses_only_packed,
        "primary_trace_input_is_symbolic": primary_symbolic,
        "no_forbidden_continuous_params": no_forbidden_params,
    }


def _train_trace_advisor(
    *,
    generator: nn.Module,
    advisor: PackedTraceAdvisor,
    examples: Sequence[M23RelevanceExample],
    vocab: dict[str, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    trace_source: str = "predicted",
) -> list[dict[str, float]]:
    source = str(trace_source).strip().lower()
    if source not in {"predicted", "oracle"}:
        raise ValueError(f"trace_source must be 'predicted' or 'oracle', got {trace_source!r}.")
    dataset = M23RelevanceDataset(examples, vocab, max_frames=advisor.max_frames)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m23_collate)
    optimizer = torch.optim.AdamW(advisor.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    advisor.train()
    for epoch in range(int(epochs)):
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["answer_id"].to(device)
            with torch.no_grad():
                if source == "oracle":
                    packed = pack_symbolic_trace_from_batch(batch, max_frames=advisor.max_frames, max_places=advisor.max_places)
                else:
                    outputs = generator(input_ids)
                    packed = pack_symbolic_trace_from_outputs(outputs, max_frames=advisor.max_frames, max_places=advisor.max_places)
            optimizer.zero_grad(set_to_none=True)
            logits = advisor(packed.to(device))
            loss = F.cross_entropy(logits, target)
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError("M24 advisor loss became non-finite.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(advisor.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            totals["accuracy"] += _accuracy(logits.detach(), target.detach())
            totals["grad_norm"] += float(grad_norm.detach().cpu().item())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    advisor.eval()
    return history


def _train_prompt_control(
    *,
    control: PromptOnlyControl,
    examples: Sequence[M23RelevanceExample],
    vocab: dict[str, int],
    max_frames: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    dataset = M23RelevanceDataset(examples, vocab, max_frames=int(max_frames))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m23_collate)
    optimizer = torch.optim.AdamW(control.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    control.train()
    for epoch in range(int(epochs)):
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["answer_id"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = control(input_ids)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(control.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            totals["accuracy"] += _accuracy(logits.detach(), target.detach())
            totals["grad_norm"] += float(grad_norm.detach().cpu().item())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    control.eval()
    return history


@torch.no_grad()
def evaluate_m24_substrate_compression(
    *,
    generator: nn.Module,
    advisor: PackedTraceAdvisor,
    oracle_advisor: PackedTraceAdvisor | None = None,
    prompt_control: PromptOnlyControl,
    examples: Sequence[M23RelevanceExample],
    vocab: dict[str, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    generator.eval()
    advisor.eval()
    if oracle_advisor is not None:
        oracle_advisor.eval()
    prompt_control.eval()
    dataset = M23RelevanceDataset(examples, vocab, max_frames=advisor.max_frames)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m23_collate)
    merged_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    predicted_traces: list[torch.Tensor] = []
    oracle_traces: list[torch.Tensor] = []
    predicted_traces_before_bottleneck: list[torch.Tensor] = []
    oracle_traces_before_bottleneck: list[torch.Tensor] = []
    prompt_token_counts: list[float] = []
    surfaces: list[str] = []
    active_frame_budget = int(advisor.active_frame_budget or 0)
    trace_symbol_budget = int(advisor.trace_symbol_budget or 0)
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device_obj)
        target = batch["answer_id"].to(device_obj)
        outputs = generator(input_ids)
        predicted_before = pack_symbolic_trace_from_outputs(outputs, max_frames=advisor.max_frames, max_places=advisor.max_places).to(device_obj)
        oracle_before = pack_symbolic_trace_from_batch(batch, max_frames=advisor.max_frames, max_places=advisor.max_places).to(device_obj)
        predicted = budget_packed_trace_symbols(
            predicted_before,
            symbol_budget=advisor.trace_symbol_budget,
            active_frame_budget=advisor.active_frame_budget,
        ).to(device_obj)
        oracle = budget_packed_trace_symbols(
            oracle_before,
            symbol_budget=advisor.trace_symbol_budget,
            active_frame_budget=advisor.active_frame_budget,
        ).to(device_obj)
        shuffled_trace = shuffled_packed_trace_like(predicted, seed=int(seed) + batch_idx).to(device_obj)
        random_trace = budget_packed_trace_symbols(
            random_packed_trace_like(oracle_before, seed=int(seed) + batch_idx, max_entities=advisor.max_entities),
            symbol_budget=advisor.trace_symbol_budget,
            active_frame_budget=advisor.active_frame_budget,
        ).to(device_obj)
        zero_trace = zero_packed_trace_like(oracle).to(device_obj)
        merged_logits["predicted_trace"].append(advisor(predicted).detach().cpu())
        merged_logits["oracle_trace"].append(advisor(oracle).detach().cpu())
        merged_logits["shuffled_trace"].append(advisor(shuffled_trace).detach().cpu())
        merged_logits["random_trace"].append(advisor(random_trace).detach().cpu())
        merged_logits["zero_trace"].append(advisor(zero_trace).detach().cpu())
        merged_logits["prompt_only"].append(prompt_control(input_ids).detach().cpu())
        if oracle_advisor is not None:
            merged_logits["oracle_trained_oracle_trace"].append(oracle_advisor(oracle).detach().cpu())
            merged_logits["oracle_trained_predicted_trace"].append(oracle_advisor(predicted).detach().cpu())
            merged_logits["oracle_trained_shuffled_trace"].append(oracle_advisor(shuffled_trace).detach().cpu())
            merged_logits["oracle_trained_random_trace"].append(oracle_advisor(random_trace).detach().cpu())
            merged_logits["oracle_trained_zero_trace"].append(oracle_advisor(zero_trace).detach().cpu())
        targets.append(target.detach().cpu())
        predicted_traces.append(predicted.detach().cpu())
        oracle_traces.append(oracle.detach().cpu())
        predicted_traces_before_bottleneck.append(predicted_before.detach().cpu())
        oracle_traces_before_bottleneck.append(oracle_before.detach().cpu())
        prompt_token_counts.extend([float(ids.ne(0).sum().detach().cpu().item()) for ids in input_ids])
        surfaces.extend(batch["surface"])
    target_all = torch.cat(targets, dim=0)
    logits_all = {key: torch.cat(value, dim=0) for key, value in merged_logits.items()}
    predicted_all = torch.cat(predicted_traces, dim=0)
    oracle_all = torch.cat(oracle_traces, dim=0)
    predicted_before_all = torch.cat(predicted_traces_before_bottleneck, dim=0)
    oracle_before_all = torch.cat(oracle_traces_before_bottleneck, dim=0)
    component_metrics = packed_trace_component_accuracy(predicted_all, oracle_all)
    predicted_accuracy = _accuracy(logits_all["predicted_trace"], target_all)
    oracle_accuracy = _accuracy(logits_all["oracle_trace"], target_all)
    shuffled_accuracy = _accuracy(logits_all["shuffled_trace"], target_all)
    random_accuracy = _accuracy(logits_all["random_trace"], target_all)
    zero_accuracy = _accuracy(logits_all["zero_trace"], target_all)
    prompt_accuracy = _accuracy(logits_all["prompt_only"], target_all)
    oracle_trained_oracle_accuracy = _accuracy(logits_all["oracle_trained_oracle_trace"], target_all) if "oracle_trained_oracle_trace" in logits_all else 0.0
    oracle_trained_predicted_accuracy = _accuracy(logits_all["oracle_trained_predicted_trace"], target_all) if "oracle_trained_predicted_trace" in logits_all else 0.0
    oracle_trained_shuffled_accuracy = _accuracy(logits_all["oracle_trained_shuffled_trace"], target_all) if "oracle_trained_shuffled_trace" in logits_all else 0.0
    oracle_trained_random_accuracy = _accuracy(logits_all["oracle_trained_random_trace"], target_all) if "oracle_trained_random_trace" in logits_all else 0.0
    oracle_trained_zero_accuracy = _accuracy(logits_all["oracle_trained_zero_trace"], target_all) if "oracle_trained_zero_trace" in logits_all else 0.0
    surface_metrics: dict[str, dict[str, float]] = {}
    pred_labels = torch.argmax(logits_all["predicted_trace"], dim=-1)
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([item == surface for item in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {
            "strict_accuracy": float((pred_labels[mask] == target_all[mask]).float().mean().item()) if bool(mask.any().item()) else 0.0,
            "count": float(mask.sum().item()),
        }
    mean_prompt_tokens = sum(prompt_token_counts) / max(1, len(prompt_token_counts))
    packed_symbols = float(advisor.max_frames * advisor.spec.width)
    diagnostic_mean_predicted_raw_nonzero_entries = float(predicted_all.ne(0).float().sum(dim=(-1, -2)).mean().item()) if predicted_all.numel() else 0.0
    diagnostic_mean_oracle_raw_nonzero_entries = float(oracle_all.ne(0).float().sum(dim=(-1, -2)).mean().item()) if oracle_all.numel() else 0.0
    predicted_symbol_counts_before = packed_trace_symbol_counts(predicted_before_all) if predicted_before_all.numel() else torch.zeros(0, dtype=torch.long)
    predicted_symbol_counts_after = packed_trace_symbol_counts(predicted_all) if predicted_all.numel() else torch.zeros(0, dtype=torch.long)
    oracle_symbol_counts_before = packed_trace_symbol_counts(oracle_before_all) if oracle_before_all.numel() else torch.zeros(0, dtype=torch.long)
    oracle_symbol_counts_after = packed_trace_symbol_counts(oracle_all) if oracle_all.numel() else torch.zeros(0, dtype=torch.long)
    mean_predicted_symbols_before = float(predicted_symbol_counts_before.float().mean().item()) if predicted_symbol_counts_before.numel() else 0.0
    mean_predicted_symbols_after = float(predicted_symbol_counts_after.float().mean().item()) if predicted_symbol_counts_after.numel() else 0.0
    mean_oracle_symbols_before = float(oracle_symbol_counts_before.float().mean().item()) if oracle_symbol_counts_before.numel() else 0.0
    mean_oracle_symbols_after = float(oracle_symbol_counts_after.float().mean().item()) if oracle_symbol_counts_after.numel() else 0.0
    if trace_symbol_budget > 0:
        predicted_symbol_budget_overflow_rate = float(predicted_symbol_counts_before.gt(trace_symbol_budget).float().mean().item()) if predicted_symbol_counts_before.numel() else 0.0
        oracle_symbol_budget_overflow_rate = float(oracle_symbol_counts_before.gt(trace_symbol_budget).float().mean().item()) if oracle_symbol_counts_before.numel() else 0.0
    else:
        predicted_symbol_budget_overflow_rate = 0.0
        oracle_symbol_budget_overflow_rate = 0.0
    predicted_dropped = (predicted_symbol_counts_before - predicted_symbol_counts_after).clamp_min(0)
    oracle_dropped = (oracle_symbol_counts_before - oracle_symbol_counts_after).clamp_min(0)
    predicted_bottleneck_symbol_drop_rate = (
        float(predicted_dropped.float().sum().item()) / max(1.0, float(predicted_symbol_counts_before.float().sum().item()))
        if predicted_symbol_counts_before.numel()
        else 0.0
    )
    oracle_bottleneck_symbol_drop_rate = (
        float(oracle_dropped.float().sum().item()) / max(1.0, float(oracle_symbol_counts_before.float().sum().item()))
        if oracle_symbol_counts_before.numel()
        else 0.0
    )
    packed_symbol_to_prompt_ratio = mean_predicted_symbols_after / max(1.0, mean_prompt_tokens)
    prompt_to_packed_symbol_ratio = mean_prompt_tokens / max(1.0, mean_predicted_symbols_after)
    token_reduction_ratio = 1.0 - packed_symbol_to_prompt_ratio
    effective_packed_symbol_to_prompt_ratio = mean_predicted_symbols_after / max(1.0, mean_prompt_tokens)
    effective_token_reduction_ratio = 1.0 - effective_packed_symbol_to_prompt_ratio
    advisor_vs_prompt_delta = predicted_accuracy - prompt_accuracy
    predicted_vs_shuffled_delta = predicted_accuracy - shuffled_accuracy
    predicted_vs_random_delta = predicted_accuracy - random_accuracy
    predicted_vs_zero_delta = predicted_accuracy - zero_accuracy
    oracle_trace_delta = oracle_accuracy - predicted_accuracy
    oracle_trained_trace_delta = oracle_trained_oracle_accuracy - oracle_trained_random_accuracy
    predicted_trace_gap_to_oracle_upper_bound = oracle_trained_oracle_accuracy - oracle_trained_predicted_accuracy
    cross_advisor_oracle_gap = oracle_trained_oracle_accuracy - predicted_accuracy
    substrate_claim_score = max(
        0.0,
        min(
            1.0,
            0.35 * float(component_metrics["bridi_trace_exact_accuracy"])
            + 0.15 * max(0.0, predicted_vs_shuffled_delta)
            + 0.10 * max(0.0, predicted_vs_random_delta)
            + 0.20 * max(0.0, advisor_vs_prompt_delta)
            + 0.10 * max(0.0, predicted_vs_zero_delta)
            + 0.10 * max(0.0, oracle_trained_trace_delta),
        ),
    )
    promotion_gates = {
        "trace_beats_shuffled": predicted_vs_shuffled_delta >= 0.10,
        "trace_beats_random": predicted_vs_random_delta >= 0.10,
        "trace_beats_zero": predicted_vs_zero_delta >= 0.10,
        "trace_matches_oracle_upper_bound": predicted_trace_gap_to_oracle_upper_bound <= 0.05,
        "trace_beats_prompt_only": advisor_vs_prompt_delta >= 0.0,
        "packed_trace_is_shorter_than_prompt": packed_symbol_to_prompt_ratio < 1.0,
        "nonzero_exact_trace_reconstruction": float(component_metrics["bridi_trace_exact_accuracy"]) > 0.0,
    }
    promotion_gate_pass_rate = sum(1.0 for passed in promotion_gates.values() if passed) / max(1, len(promotion_gates))
    hard_bottleneck_trace_accuracy = packed_trace_exact_accuracy(predicted_all, oracle_all)
    hard_bottleneck_vs_shuffled_delta = predicted_vs_shuffled_delta
    hard_bottleneck_vs_random_delta = predicted_vs_random_delta
    hard_bottleneck_configured = active_frame_budget > 0 or trace_symbol_budget > 0
    symbol_budget_respected = trace_symbol_budget <= 0 or (
        bool(predicted_symbol_counts_after.le(trace_symbol_budget).all().item())
        and bool(oracle_symbol_counts_after.le(trace_symbol_budget).all().item())
    )
    m24_2_gate_metrics = m24_2_promotion_gate_metrics(
        hard_bottleneck_configured=hard_bottleneck_configured,
        strict_accuracy=predicted_accuracy,
        predicted_vs_shuffled_delta=hard_bottleneck_vs_shuffled_delta,
        predicted_vs_random_delta=hard_bottleneck_vs_random_delta,
        hard_bottleneck_trace_accuracy=hard_bottleneck_trace_accuracy,
        effective_packed_symbol_to_prompt_ratio=effective_packed_symbol_to_prompt_ratio,
        symbol_budget_respected=symbol_budget_respected,
        advisor_vs_prompt_delta=advisor_vs_prompt_delta,
    )
    hard_bottleneck_accuracy_per_token = predicted_accuracy / max(1.0, mean_predicted_symbols_after)
    hard_bottleneck_symbol_error_rate = 1.0 - hard_bottleneck_trace_accuracy
    return {
        "strict_accuracy": predicted_accuracy,
        "predicted_trace_accuracy": predicted_accuracy,
        "oracle_trace_accuracy": oracle_accuracy,
        "shuffled_trace_accuracy": shuffled_accuracy,
        "random_trace_accuracy": random_accuracy,
        "zero_trace_accuracy": zero_accuracy,
        "no_trace_accuracy": zero_accuracy,
        "prompt_only_accuracy": prompt_accuracy,
        "advisor_vs_prompt_delta": advisor_vs_prompt_delta,
        "m24_strict_delta_vs_prompt_only": advisor_vs_prompt_delta,
        "predicted_vs_shuffled_delta": predicted_vs_shuffled_delta,
        "predicted_vs_random_delta": predicted_vs_random_delta,
        "predicted_vs_zero_delta": predicted_vs_zero_delta,
        "oracle_trace_delta": oracle_trace_delta,
        "oracle_trained_oracle_trace_accuracy": oracle_trained_oracle_accuracy,
        "oracle_trained_predicted_trace_accuracy": oracle_trained_predicted_accuracy,
        "oracle_trained_shuffled_trace_accuracy": oracle_trained_shuffled_accuracy,
        "oracle_trained_random_trace_accuracy": oracle_trained_random_accuracy,
        "oracle_trained_zero_trace_accuracy": oracle_trained_zero_accuracy,
        "oracle_trained_trace_delta": oracle_trained_trace_delta,
        "predicted_trace_gap_to_oracle_upper_bound": predicted_trace_gap_to_oracle_upper_bound,
        "cross_advisor_oracle_gap": cross_advisor_oracle_gap,
        "trace_advisor_delta": predicted_vs_random_delta,
        "surface_metrics": surface_metrics,
        "prompt_mean_tokens": mean_prompt_tokens,
        "reference_token_count": mean_prompt_tokens,
        "substrate_token_count": mean_predicted_symbols_after,
        "substrate_tokens": mean_predicted_symbols_after,
        "packed_trace_symbols": packed_symbols,
        "diagnostic_mean_predicted_raw_nonzero_entries": diagnostic_mean_predicted_raw_nonzero_entries,
        "diagnostic_mean_oracle_raw_nonzero_entries": diagnostic_mean_oracle_raw_nonzero_entries,
        "packed_symbol_to_prompt_ratio": packed_symbol_to_prompt_ratio,
        "prompt_to_packed_symbol_ratio": prompt_to_packed_symbol_ratio,
        "packed_to_prompt_ratio": packed_symbol_to_prompt_ratio,
        "prompt_to_packed_ratio": prompt_to_packed_symbol_ratio,
        "token_reduction_ratio": token_reduction_ratio,
        "active_frame_budget": float(active_frame_budget),
        "trace_symbol_budget": float(trace_symbol_budget),
        "hard_trace_length_bottleneck_active": 1.0 if active_frame_budget > 0 else 0.0,
        "hard_symbol_budget_active": 1.0 if trace_symbol_budget > 0 else 0.0,
        "mean_predicted_emitted_symbols_before_bottleneck": mean_predicted_symbols_before,
        "mean_predicted_emitted_symbols_after_bottleneck": mean_predicted_symbols_after,
        "mean_oracle_emitted_symbols_before_bottleneck": mean_oracle_symbols_before,
        "mean_oracle_emitted_symbols_after_bottleneck": mean_oracle_symbols_after,
        "predicted_symbol_budget_overflow_rate": predicted_symbol_budget_overflow_rate,
        "oracle_symbol_budget_overflow_rate": oracle_symbol_budget_overflow_rate,
        "predicted_bottleneck_symbol_drop_rate": predicted_bottleneck_symbol_drop_rate,
        "oracle_bottleneck_symbol_drop_rate": oracle_bottleneck_symbol_drop_rate,
        "effective_packed_symbol_to_prompt_ratio": effective_packed_symbol_to_prompt_ratio,
        "effective_token_reduction_ratio": effective_token_reduction_ratio,
        "hard_bottleneck_trace_accuracy": hard_bottleneck_trace_accuracy,
        "hard_bottleneck_vs_shuffled_delta": hard_bottleneck_vs_shuffled_delta,
        "hard_bottleneck_vs_random_delta": hard_bottleneck_vs_random_delta,
        "m24_2_hard_bottleneck_strict_accuracy": predicted_accuracy,
        "m24_2_hard_bottleneck_trace_exact_accuracy": hard_bottleneck_trace_accuracy,
        "m24_2_hard_bottleneck_token_count": mean_predicted_symbols_after,
        "m24_2_hard_bottleneck_compression_ratio": effective_packed_symbol_to_prompt_ratio,
        "m24_2_hard_bottleneck_accuracy_per_token": hard_bottleneck_accuracy_per_token,
        "m24_2_hard_bottleneck_delta_vs_prompt_only": advisor_vs_prompt_delta,
        "m24_2_hard_bottleneck_symbol_error_rate": hard_bottleneck_symbol_error_rate,
        "m24_2_hard_bottleneck_score": m24_2_gate_metrics["m24_2_promotion_gate_pass_rate"],
        **m24_2_gate_metrics,
        "compression_ratio": prompt_to_packed_symbol_ratio,
        "packed_symbol_compression_ratio": packed_symbol_to_prompt_ratio,
        "oracle_symbol_compression_ratio": mean_oracle_symbols_after / max(1.0, mean_prompt_tokens),
        "accuracy_per_packed_symbol": predicted_accuracy / max(1.0, mean_predicted_symbols_after),
        "accuracy_per_trace_token": predicted_accuracy / max(1.0, mean_predicted_symbols_after),
        "strict_accuracy_per_substrate_token": predicted_accuracy / max(1.0, mean_predicted_symbols_after),
        "substrate_claim_score": substrate_claim_score,
        "m24_promotion_gate_pass_rate": promotion_gate_pass_rate,
        "m24_promotion_candidate": 1.0 if all(promotion_gates.values()) else 0.0,
        "m24_gate_trace_beats_shuffled": 1.0 if promotion_gates["trace_beats_shuffled"] else 0.0,
        "m24_gate_trace_beats_random": 1.0 if promotion_gates["trace_beats_random"] else 0.0,
        "m24_gate_trace_beats_zero": 1.0 if promotion_gates["trace_beats_zero"] else 0.0,
        "m24_gate_trace_matches_oracle_upper_bound": 1.0 if promotion_gates["trace_matches_oracle_upper_bound"] else 0.0,
        "m24_gate_trace_beats_prompt_only": 1.0 if promotion_gates["trace_beats_prompt_only"] else 0.0,
        "m24_gate_packed_trace_shorter_than_prompt": 1.0 if promotion_gates["packed_trace_is_shorter_than_prompt"] else 0.0,
        "m24_gate_token_reduction_positive": 1.0 if promotion_gates["packed_trace_is_shorter_than_prompt"] else 0.0,
        "m24_gate_nonzero_exact_trace_reconstruction": 1.0 if promotion_gates["nonzero_exact_trace_reconstruction"] else 0.0,
        **component_metrics,
    }


def train_m24_substrate_compression(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    generator_epochs: int = 8,
    advisor_epochs: int = 8,
    prompt_epochs: int | None = None,
    batch_size: int = 128,
    generator_learning_rate: float = 2e-3,
    advisor_learning_rate: float = 2e-3,
    seed: int = 24,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    max_frames: int = 6,
    max_places: int = DEFAULT_MAX_PLACES,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    trace_weight: float = 2.5,
    answer_weight: float = 0.2,
    mdl_weight: float = DEFAULT_M24_MDL_WEIGHT,
    trace_exact_surrogate_weight: float = 0.5,
    active_frame_budget: int | None = None,
    trace_symbol_budget: int | None = None,
    clean_train_fraction: float = 0.35,
    clean_eval_fraction: float = 0.35,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_obj = torch.device(device)
    stage1 = train_m23_relevance_router(
        train_size=int(train_size),
        eval_size=int(eval_size),
        epochs=int(generator_epochs),
        batch_size=int(batch_size),
        learning_rate=float(generator_learning_rate),
        seed=int(seed),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        max_frames=int(max_frames),
        max_places=int(max_places),
        max_entities=int(max_entities),
        trace_weight=float(trace_weight),
        answer_weight=float(answer_weight),
        counterfactual_weight=1.25,
        brivi_lock_weight=1.5,
        frame_necessity_weight=1.0,
        mdl_weight=float(mdl_weight),
        relevance_rank_weight=0.0,
        trace_exact_surrogate_weight=float(trace_exact_surrogate_weight),
        use_relevance_router=False,
        clean_train_fraction=float(clean_train_fraction),
        clean_eval_fraction=float(clean_eval_fraction),
        judri_bridge_gate=True,
        device=device_obj,
    )
    generator = stage1["model"]
    _freeze_module(generator)
    frozen_parameter_count = sum(1 for parameter in generator.parameters() if not parameter.requires_grad)
    trainable_generator_parameters = sum(1 for parameter in generator.parameters() if parameter.requires_grad)
    frozen_snapshots = [parameter.detach().clone().cpu() for parameter in generator.parameters()]
    advisor = PackedTraceAdvisor(
        max_frames=int(max_frames),
        max_places=int(max_places),
        max_entities=int(max_entities),
        hidden_dim=int(advisor_hidden_dim),
        active_frame_budget=active_frame_budget,
        trace_symbol_budget=trace_symbol_budget,
    ).to(device_obj)
    oracle_advisor = PackedTraceAdvisor(
        max_frames=int(max_frames),
        max_places=int(max_places),
        max_entities=int(max_entities),
        hidden_dim=int(advisor_hidden_dim),
        active_frame_budget=active_frame_budget,
        trace_symbol_budget=trace_symbol_budget,
    ).to(device_obj)
    prompt_control = PromptOnlyControl(vocab_size=len(stage1["vocab"]), embedding_dim=max(8, int(embedding_dim) // 2), hidden_dim=int(advisor_hidden_dim)).to(device_obj)
    advisor_history = _train_trace_advisor(
        generator=generator,
        advisor=advisor,
        examples=stage1["train_examples"],
        vocab=stage1["vocab"],
        epochs=int(advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 1,
        trace_source="predicted",
    )
    oracle_advisor_history = _train_trace_advisor(
        generator=generator,
        advisor=oracle_advisor,
        examples=stage1["train_examples"],
        vocab=stage1["vocab"],
        epochs=int(advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 11,
        trace_source="oracle",
    )
    prompt_history = _train_prompt_control(
        control=prompt_control,
        examples=stage1["train_examples"],
        vocab=stage1["vocab"],
        max_frames=int(max_frames),
        epochs=int(prompt_epochs if prompt_epochs is not None else advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 2,
    )
    metrics = evaluate_m24_substrate_compression(
        generator=generator,
        advisor=advisor,
        oracle_advisor=oracle_advisor,
        prompt_control=prompt_control,
        examples=stage1["eval_examples"],
        vocab=stage1["vocab"],
        batch_size=int(batch_size),
        device=device_obj,
        seed=int(seed) + 3,
    )
    max_delta = 0.0
    for snapshot, parameter in zip(frozen_snapshots, generator.parameters(), strict=False):
        delta = (parameter.detach().cpu() - snapshot).abs().max().item()
        max_delta = max(max_delta, float(delta))
    advisor_contract = _advisor_symbolic_contract_status(advisor)
    metrics["generator_frozen_parameter_count"] = float(frozen_parameter_count)
    metrics["generator_trainable_parameter_count_after_freeze"] = float(trainable_generator_parameters)
    metrics["generator_parameter_max_delta_after_advisor"] = float(max_delta)
    metrics["generator_parameters_unchanged_after_advisor"] = 1.0 if max_delta <= 0.0 else 0.0
    metrics["advisor_primary_trace_is_symbolic"] = 1.0 if advisor_contract["primary_trace_input_is_symbolic"] else 0.0
    metrics["continuous_trace_smuggling_detected"] = 0.0 if all(advisor_contract.values()) else 1.0
    metrics["mdl_weight"] = float(mdl_weight)
    metrics["active_frame_budget"] = float(advisor.active_frame_budget or 0)
    metrics["trace_symbol_budget"] = float(advisor.trace_symbol_budget or 0)
    return {
        "generator": generator,
        "advisor": advisor,
        "oracle_advisor": oracle_advisor,
        "prompt_control": prompt_control,
        "vocab": stage1["vocab"],
        "train_examples": stage1["train_examples"],
        "eval_examples": stage1["eval_examples"],
        "stage1_history": stage1["history"],
        "advisor_history": advisor_history,
        "oracle_advisor_history": oracle_advisor_history,
        "prompt_history": prompt_history,
        "stage1_metrics": stage1["metrics"],
        "stage1_config": stage1["config"],
        "metrics": metrics,
        "config": {
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "generator_epochs": int(generator_epochs),
            "advisor_epochs": int(advisor_epochs),
            "prompt_epochs": int(prompt_epochs if prompt_epochs is not None else advisor_epochs),
            "batch_size": int(batch_size),
            "generator_learning_rate": float(generator_learning_rate),
            "advisor_learning_rate": float(advisor_learning_rate),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "max_frames": int(max_frames),
            "max_places": int(max_places),
            "max_entities": int(max_entities),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "trace_exact_surrogate_weight": float(trace_exact_surrogate_weight),
            "active_frame_budget": int(advisor.active_frame_budget or 0),
            "trace_symbol_budget": int(advisor.trace_symbol_budget or 0),
            "clean_train_fraction": float(clean_train_fraction),
            "clean_eval_fraction": float(clean_eval_fraction),
            "stage1_reused_m23_training_path": True,
            "advisor_primary_trace_input": PackedTraceAdvisor.primary_trace_input,
            "advisor_disallowed_primary_inputs": list(DISALLOWED_ADVISOR_TRACE_INPUTS),
            "oracle_trained_advisor_control": True,
        },
    }


def metric_lock_status(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "substrate_first": float(metrics.get("generator_trainable_parameter_count_after_freeze", 1.0)) == 0.0,
        "symbolic_trace_only": float(metrics.get("advisor_primary_trace_is_symbolic", 0.0)) == 1.0
        and float(metrics.get("continuous_trace_smuggling_detected", 1.0)) == 0.0,
        "generator_unchanged_after_advisor": float(metrics.get("generator_parameters_unchanged_after_advisor", 0.0)) == 1.0,
        "compression_controls": all(
            key in metrics
            for key in (
                "predicted_trace_accuracy",
                "oracle_trace_accuracy",
                "shuffled_trace_accuracy",
                "random_trace_accuracy",
                "zero_trace_accuracy",
                "prompt_only_accuracy",
            )
        ),
    }

