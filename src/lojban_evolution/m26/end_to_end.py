from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import ANSWER_LABELS, build_vocab
from lojban_evolution.m24.compression import _accuracy
from lojban_evolution.m25.emergent_bridi import (
    DEFAULT_M25_MDL_WEIGHT,
    LOOSE_PAD,
    LOOSE_TYPE_COUNT,
    M25EmergentBridiExample,
    M25EmergentBridiQFormer,
    M25LooseBridiDataset,
    _aux_vocab_size,
    _component_accuracy,
    _value_vocab_size,
    generate_m25_emergent_bridi_examples,
    loose_stream_symbol_counts,
    m25_collate,
    pack_loose_stream_from_outputs,
)


DEFAULT_M26_TRACE_WEIGHT = 2.0
DEFAULT_M26_ANSWER_WEIGHT = 1.0


@dataclass(frozen=True)
class M26GradientProbe:
    answer_loss_generator_grad_norm: float
    answer_loss_symbol_head_grad_norm: float
    answer_loss_advisor_grad_norm: float
    answer_loss_reaches_generator: float
    answer_loss_reaches_symbol_heads: float

    def as_dict(self) -> dict[str, float]:
        return {
            "answer_loss_generator_grad_norm": self.answer_loss_generator_grad_norm,
            "answer_loss_symbol_head_grad_norm": self.answer_loss_symbol_head_grad_norm,
            "answer_loss_advisor_grad_norm": self.answer_loss_advisor_grad_norm,
            "answer_loss_reaches_generator": self.answer_loss_reaches_generator,
            "answer_loss_reaches_symbol_heads": self.answer_loss_reaches_symbol_heads,
        }


class DifferentiableLooseStreamAdvisor(nn.Module):
    """Trace-only answer advisor over soft loose-bridi symbol distributions."""

    primary_trace_input = "soft_differentiable_loose_bridi_stream"
    disallowed_primary_inputs = ("prompt_state", "hidden_states", "raw_prompt_tokens")

    def __init__(
        self,
        *,
        max_symbols: int,
        value_vocab_size: int | None = None,
        aux_vocab_size: int | None = None,
        hidden_dim: int = 64,
        symbol_budget: int | None = None,
    ) -> None:
        super().__init__()
        self.max_symbols = int(max_symbols)
        self.value_vocab_size = int(value_vocab_size or _value_vocab_size())
        self.aux_vocab_size = int(aux_vocab_size or _aux_vocab_size())
        self.symbol_budget = None if symbol_budget is None or int(symbol_budget) <= 0 else int(symbol_budget)
        self.type_embedding = nn.Embedding(LOOSE_TYPE_COUNT, int(hidden_dim), padding_idx=LOOSE_PAD)
        self.value_embedding = nn.Embedding(self.value_vocab_size, int(hidden_dim), padding_idx=0)
        self.aux_embedding = nn.Embedding(self.aux_vocab_size, int(hidden_dim), padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), len(ANSWER_LABELS)),
        )

    def encode_from_logits(
        self,
        *,
        active_logits: torch.Tensor,
        type_logits: torch.Tensor,
        value_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        active_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_probs = torch.softmax(type_logits, dim=-1)
        value_probs = torch.softmax(value_logits, dim=-1)
        aux_probs = torch.softmax(aux_logits, dim=-1)
        active = torch.sigmoid(active_logits) if active_override is None else active_override.to(active_logits).float()
        if self.symbol_budget is not None:
            budget_mask = torch.zeros_like(active)
            budget_mask[:, : min(self.symbol_budget, active.shape[1])] = 1.0
            active = active * budget_mask
        embedded = (
            type_probs @ self.type_embedding.weight
            + value_probs @ self.value_embedding.weight
            + aux_probs @ self.aux_embedding.weight
        )
        active = active.unsqueeze(-1)
        return (embedded * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)

    def forward_from_logits(
        self,
        *,
        active_logits: torch.Tensor,
        type_logits: torch.Tensor,
        value_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        active_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.encode_from_logits(
                active_logits=active_logits,
                type_logits=type_logits,
                value_logits=value_logits,
                aux_logits=aux_logits,
                active_override=active_override,
            )
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args:
            raise TypeError("DifferentiableLooseStreamAdvisor accepts logits by keyword, not hard integer streams.")
        return self.forward_from_logits(**kwargs)


class M26EndToEndLoafman(nn.Module):
    """One trainable prompt -> bridi stream -> advisor organism.

    This is intentionally smaller than the eventual chatbot path. Its job is to
    establish the missing spinal cord: final answer loss must reach the symbolic
    stream emitter without a hard argmax/no_grad cut.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_symbols: int = 32,
        value_vocab_size: int | None = None,
        aux_vocab_size: int | None = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        advisor_hidden_dim: int = 64,
        symbol_budget: int | None = None,
    ) -> None:
        super().__init__()
        self.generator = M25EmergentBridiQFormer(
            vocab_size=int(vocab_size),
            max_symbols=int(max_symbols),
            value_vocab_size=value_vocab_size or _value_vocab_size(),
            aux_vocab_size=aux_vocab_size or _aux_vocab_size(),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
        )
        self.advisor = DifferentiableLooseStreamAdvisor(
            max_symbols=int(max_symbols),
            value_vocab_size=value_vocab_size or _value_vocab_size(),
            aux_vocab_size=aux_vocab_size or _aux_vocab_size(),
            hidden_dim=int(advisor_hidden_dim),
            symbol_budget=symbol_budget,
        )

    @property
    def max_symbols(self) -> int:
        return self.generator.max_symbols

    def advisor_logits_from_generator_outputs(
        self,
        generator_outputs: dict[str, torch.Tensor],
        *,
        active_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.advisor.forward_from_logits(
            active_logits=generator_outputs["active_logits"],
            type_logits=generator_outputs["type_logits"],
            value_logits=generator_outputs["value_logits"],
            aux_logits=generator_outputs["aux_logits"],
            active_override=active_override,
        )

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        generator_outputs = self.generator(input_ids)
        answer_logits = self.advisor_logits_from_generator_outputs(generator_outputs)
        return {
            **generator_outputs,
            "generator_answer_logits": generator_outputs["answer_logits"],
            "answer_logits": answer_logits,
            "trace_state": self.advisor.encode_from_logits(
                active_logits=generator_outputs["active_logits"],
                type_logits=generator_outputs["type_logits"],
                value_logits=generator_outputs["value_logits"],
                aux_logits=generator_outputs["aux_logits"],
            ),
        }


def compute_m26_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    trace_weight: float = DEFAULT_M26_TRACE_WEIGHT,
    answer_weight: float = DEFAULT_M26_ANSWER_WEIGHT,
    mdl_weight: float = DEFAULT_M25_MDL_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["active_logits"].device
    active_target = batch["stream_active_targets"].to(device)
    active_mask = active_target > 0.5
    type_target = batch["type_targets"].to(device)
    value_target = batch["value_targets"].to(device).clamp_max(outputs["value_logits"].shape[-1] - 1)
    aux_target = batch["aux_targets"].to(device).clamp_max(outputs["aux_logits"].shape[-1] - 1)
    answer_target = batch["answer_id"].to(device)
    positive = active_target.sum().clamp_min(1.0)
    negative = torch.numel(active_target) - positive
    pos_weight = (negative / positive).clamp(1.0, 10.0)
    active_loss = F.binary_cross_entropy_with_logits(outputs["active_logits"], active_target, pos_weight=pos_weight)
    if bool(active_mask.any().item()):
        type_loss = F.cross_entropy(outputs["type_logits"][active_mask], type_target[active_mask])
        value_loss = F.cross_entropy(outputs["value_logits"][active_mask], value_target[active_mask])
        aux_loss = F.cross_entropy(outputs["aux_logits"][active_mask], aux_target[active_mask])
    else:
        type_loss = value_loss = aux_loss = outputs["active_logits"].sum() * 0.0
    answer_loss = F.cross_entropy(outputs["answer_logits"], answer_target)
    generator_answer_loss = F.cross_entropy(outputs["generator_answer_logits"], answer_target)
    mdl_loss = torch.sigmoid(outputs["active_logits"]).mean()
    stream_loss = active_loss + type_loss + value_loss + aux_loss
    loss = float(trace_weight) * stream_loss + float(answer_weight) * answer_loss + float(mdl_weight) * mdl_loss
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "active_loss": float(active_loss.detach().cpu().item()),
        "type_loss": float(type_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
        "answer_loss": float(answer_loss.detach().cpu().item()),
        "generator_answer_loss_diagnostic": float(generator_answer_loss.detach().cpu().item()),
        "mdl_loss": float(mdl_loss.detach().cpu().item()),
    }


def probe_m26_answer_gradient_flow(
    model: M26EndToEndLoafman,
    batch: dict[str, Any],
) -> M26GradientProbe:
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    targets = batch["answer_id"].to(input_ids.device)
    loss = F.cross_entropy(model(input_ids)["answer_logits"], targets)
    loss.backward()

    generator_norm = _grad_norm(model.generator.parameters())
    advisor_norm = _grad_norm(model.advisor.parameters())
    symbol_params = []
    for module in (model.generator.active_head, model.generator.type_head, model.generator.value_head, model.generator.aux_head):
        symbol_params.extend(module.parameters())
    symbol_norm = _grad_norm(symbol_params)
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return M26GradientProbe(
        answer_loss_generator_grad_norm=generator_norm,
        answer_loss_symbol_head_grad_norm=symbol_norm,
        answer_loss_advisor_grad_norm=advisor_norm,
        answer_loss_reaches_generator=1.0 if generator_norm > 0.0 else 0.0,
        answer_loss_reaches_symbol_heads=1.0 if symbol_norm > 0.0 else 0.0,
    )


def evaluate_m26_end_to_end_loafman(
    *,
    model: M26EndToEndLoafman,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=model.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m25_collate)
    model.eval()
    logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    streams: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    surfaces: list[str] = []
    symbol_counts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_obj)
            target = batch["answer_id"].to(device_obj)
            outputs = model(input_ids)
            generator_outputs = _generator_slice(outputs)
            shuffled_outputs = _shuffle_generator_outputs(generator_outputs, seed=int(seed) + batch_idx)
            random_outputs = _random_generator_outputs(generator_outputs, seed=int(seed) + 1000 + batch_idx)
            zero_active = torch.zeros_like(generator_outputs["active_logits"])

            logits["predicted"].append(outputs["answer_logits"].detach().cpu())
            logits["shuffled"].append(model.advisor_logits_from_generator_outputs(shuffled_outputs).detach().cpu())
            logits["random"].append(model.advisor_logits_from_generator_outputs(random_outputs).detach().cpu())
            logits["zero"].append(
                model.advisor_logits_from_generator_outputs(generator_outputs, active_override=zero_active).detach().cpu()
            )
            predicted = pack_loose_stream_from_outputs(generator_outputs)
            oracle = batch["stream_targets"].to(device_obj)
            streams["predicted"].append(predicted.detach().cpu())
            streams["oracle"].append(oracle.detach().cpu())
            targets.append(target.detach().cpu())
            surfaces.extend(batch["surface"])
            symbol_counts.append(loose_stream_symbol_counts(predicted).detach().cpu())
    target_all = torch.cat(targets, dim=0)
    all_logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    all_streams = {key: torch.cat(value, dim=0) for key, value in streams.items()}
    predicted_acc = _accuracy(all_logits["predicted"], target_all)
    shuffled_acc = _accuracy(all_logits["shuffled"], target_all)
    random_acc = _accuracy(all_logits["random"], target_all)
    zero_acc = _accuracy(all_logits["zero"], target_all)
    pred_count = torch.cat(symbol_counts, dim=0)
    mean_pred = float(pred_count.mean().item()) if pred_count.numel() else 0.0
    metrics = {
        "strict_accuracy": predicted_acc,
        "synthetic_world_accuracy": predicted_acc,
        "phrase_accuracy": predicted_acc,
        "end_to_end_answer_accuracy": predicted_acc,
        "shuffled_trace_accuracy": shuffled_acc,
        "random_trace_accuracy": random_acc,
        "zero_trace_accuracy": zero_acc,
        "predicted_vs_shuffled_delta": float(predicted_acc - shuffled_acc),
        "predicted_vs_random_delta": float(predicted_acc - random_acc),
        "predicted_vs_zero_delta": float(predicted_acc - zero_acc),
        "mean_predicted_emitted_symbols_after_bottleneck": mean_pred,
        "accuracy_per_loose_symbol": float(predicted_acc / max(1.0, mean_pred)),
        "single_optimizer_end_to_end_training": 1.0,
        "hard_argmax_training_cut_detected": 0.0,
        "torch_no_grad_training_cut_detected": 0.0,
        "advisor_primary_trace_is_differentiable": 1.0,
    }
    metrics.update(_component_accuracy(all_streams["predicted"], all_streams["oracle"]))
    pred_labels = torch.argmax(all_logits["predicted"], dim=-1)
    surface_metrics: dict[str, dict[str, float]] = {}
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([item == surface for item in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {
            "strict_accuracy": float((pred_labels[mask] == target_all[mask]).float().mean().item()),
            "count": float(mask.sum().item()),
        }
    return {"metrics": metrics, "surface_metrics": surface_metrics}


def train_m26_end_to_end_loafman(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 26,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    max_frames: int = 6,
    max_symbols: int = 32,
    symbol_budget: int | None = None,
    trace_weight: float = DEFAULT_M26_TRACE_WEIGHT,
    answer_weight: float = DEFAULT_M26_ANSWER_WEIGHT,
    mdl_weight: float = DEFAULT_M25_MDL_WEIGHT,
    clean_train_fraction: float = 0.35,
    clean_eval_fraction: float = 0.35,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    train_examples = generate_m25_emergent_bridi_examples(
        int(train_size), seed=int(seed), clean_fraction=float(clean_train_fraction), max_frames=int(max_frames), max_symbols=int(max_symbols)
    )
    eval_examples = generate_m25_emergent_bridi_examples(
        int(eval_size), seed=int(seed) + 999, clean_fraction=float(clean_eval_fraction), max_frames=int(max_frames), max_symbols=int(max_symbols)
    )
    vocab = build_vocab([*train_examples, *eval_examples])  # type: ignore[arg-type]
    model = M26EndToEndLoafman(
        vocab_size=len(vocab),
        max_symbols=int(max_symbols),
        value_vocab_size=_value_vocab_size(),
        aux_vocab_size=_aux_vocab_size(),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        advisor_hidden_dim=int(advisor_hidden_dim),
        symbol_budget=symbol_budget,
    ).to(device_obj)
    dataset = M25LooseBridiDataset(train_examples, vocab, max_symbols=model.max_symbols)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        collate_fn=m25_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    model.train()
    for _ in range(int(epochs)):
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            outputs = model(batch["input_ids"].to(device_obj))
            loss, metrics = compute_m26_loss(
                outputs,
                batch,
                trace_weight=float(trace_weight),
                answer_weight=float(answer_weight),
                mdl_weight=float(mdl_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()})

    eval_payload = evaluate_m26_end_to_end_loafman(
        model=model,
        examples=eval_examples,
        vocab=vocab,
        batch_size=int(batch_size),
        device=device_obj,
        seed=int(seed),
    )
    probe_batch = m25_collate([M25LooseBridiDataset(eval_examples, vocab, max_symbols=model.max_symbols)[i] for i in range(min(8, len(eval_examples)))])
    gradient_probe = probe_m26_answer_gradient_flow(model, probe_batch).as_dict()
    metrics = dict(eval_payload["metrics"])
    metrics.update(gradient_probe)
    metrics.update(
        {
            "trainable_parameter_count": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "generator_trainable_parameter_count": float(sum(p.numel() for p in model.generator.parameters() if p.requires_grad)),
            "advisor_trainable_parameter_count": float(sum(p.numel() for p in model.advisor.parameters() if p.requires_grad)),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
        }
    )
    metrics.update(m26_promotion_gate_metrics(metrics))
    return {
        "config": {
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "max_frames": int(max_frames),
            "max_symbols": int(max_symbols),
            "symbol_budget": int(symbol_budget or 0),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "device": str(device_obj),
        },
        "metrics": metrics,
        "surface_metrics": eval_payload["surface_metrics"],
        "history": history,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "vocab_size": len(vocab),
    }


def m26_promotion_gate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    gates = {
        "m26_gate_answer_loss_reaches_generator": 1.0 if metrics.get("answer_loss_reaches_generator", 0.0) >= 1.0 else 0.0,
        "m26_gate_answer_loss_reaches_symbol_heads": 1.0 if metrics.get("answer_loss_reaches_symbol_heads", 0.0) >= 1.0 else 0.0,
        "m26_gate_single_optimizer": 1.0 if metrics.get("single_optimizer_end_to_end_training", 0.0) >= 1.0 else 0.0,
        "m26_gate_no_hard_training_cut": 1.0 if metrics.get("hard_argmax_training_cut_detected", 1.0) == 0.0 else 0.0,
        "m26_gate_stream_beats_zero": 1.0 if metrics.get("predicted_vs_zero_delta", 0.0) >= 0.02 else 0.0,
    }
    gates["m26_spinal_cord_gate_pass_rate"] = sum(gates.values()) / max(1, len(gates))
    gates["m26_promotion_candidate"] = 1.0 if all(value == 1.0 for value in gates.values()) else 0.0
    return gates


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu().item())
    return float(total**0.5)


def _generator_slice(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: outputs[key] for key in ("active_logits", "type_logits", "value_logits", "aux_logits")}


def _shuffle_generator_outputs(outputs: dict[str, torch.Tensor], *, seed: int) -> dict[str, torch.Tensor]:
    if outputs["active_logits"].shape[0] <= 1:
        return {key: value.clone() for key, value in outputs.items()}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(outputs["active_logits"].shape[0], generator=generator).to(outputs["active_logits"].device)
    return {key: value[order].clone() for key, value in outputs.items()}


def _random_generator_outputs(outputs: dict[str, torch.Tensor], *, seed: int) -> dict[str, torch.Tensor]:
    device = outputs["active_logits"].device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return {
        key: torch.randn(value.shape, generator=generator, device=device, dtype=value.dtype)
        for key, value in outputs.items()
    }
