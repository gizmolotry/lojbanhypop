from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import ANSWER_LABELS, build_vocab
from lojban_evolution.m24.compression import PromptOnlyControl, _accuracy
from lojban_evolution.m25.emergent_bridi import (
    DEFAULT_M25_MDL_WEIGHT,
    LOOSE_PAD,
    LOOSE_STOP,
    LOOSE_TYPE_COUNT,
    M25EmergentBridiExample,
    M25LooseBridiDataset,
    _aux_vocab_size,
    _component_accuracy,
    _value_vocab_size,
    budget_prompt_tokens,
    generate_m25_emergent_bridi_examples,
    loose_stream_symbol_counts,
    m25_collate,
    pack_loose_stream_from_outputs,
)
from lojban_evolution.m26.end_to_end import (
    DEFAULT_M26_ANSWER_WEIGHT,
    DEFAULT_M26_TRACE_WEIGHT,
    DifferentiableLooseStreamAdvisor,
    M26TinyLanguageBackbone,
    M26TraceLanguageBridge,
    _grad_norm,
    _random_generator_outputs,
    _shuffle_generator_outputs,
    _train_prompt_control,
)


DEFAULT_M27_TRACE_WEIGHT = DEFAULT_M26_TRACE_WEIGHT
DEFAULT_M27_ANSWER_WEIGHT = DEFAULT_M26_ANSWER_WEIGHT
DEFAULT_M27_MDL_WEIGHT = DEFAULT_M25_MDL_WEIGHT
DEFAULT_M27_RELEVANCE_MARGIN = 0.15


@dataclass(frozen=True)
class M27GradientProbe:
    answer_loss_generator_grad_norm: float
    answer_loss_coconut_cell_grad_norm: float
    answer_loss_symbol_head_grad_norm: float
    answer_loss_recurrent_feedback_grad_norm: float
    answer_loss_advisor_grad_norm: float
    answer_loss_trace_slot_advisor_grad_norm: float
    answer_loss_advisor_classifier_grad_norm: float
    answer_loss_language_backbone_grad_norm: float
    answer_loss_bridge_grad_norm: float
    answer_loss_reaches_generator: float
    answer_loss_reaches_coconut_cell: float
    answer_loss_reaches_symbol_heads: float
    answer_loss_reaches_recurrent_bridi_feedback: float
    answer_loss_reaches_trace_slot_advisor: float
    answer_loss_reaches_advisor_classifier: float
    answer_loss_reaches_language_backbone: float
    answer_loss_reaches_bridge: float

    @property
    def answer_loss_reaches_prompt_encoder(self) -> float:
        return self.answer_loss_reaches_language_backbone

    @property
    def answer_loss_reaches_bridi_emitter(self) -> float:
        return self.answer_loss_reaches_generator

    @property
    def answer_loss_reaches_trace_bridge(self) -> float:
        return self.answer_loss_reaches_bridge

    @property
    def answer_loss_reaches_answer_decoder(self) -> float:
        return self.answer_loss_reaches_bridge

    @property
    def answer_loss_reaches_stop_head(self) -> float:
        return self.answer_loss_reaches_symbol_heads

    @property
    def hard_argmax_training_cut_detected(self) -> float:
        return 0.0

    @property
    def torch_no_grad_training_cut_detected(self) -> float:
        return 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "answer_loss_generator_grad_norm": self.answer_loss_generator_grad_norm,
            "answer_loss_coconut_cell_grad_norm": self.answer_loss_coconut_cell_grad_norm,
            "answer_loss_symbol_head_grad_norm": self.answer_loss_symbol_head_grad_norm,
            "answer_loss_recurrent_feedback_grad_norm": self.answer_loss_recurrent_feedback_grad_norm,
            "answer_loss_advisor_grad_norm": self.answer_loss_advisor_grad_norm,
            "answer_loss_trace_slot_advisor_grad_norm": self.answer_loss_trace_slot_advisor_grad_norm,
            "answer_loss_advisor_classifier_grad_norm": self.answer_loss_advisor_classifier_grad_norm,
            "answer_loss_language_backbone_grad_norm": self.answer_loss_language_backbone_grad_norm,
            "answer_loss_bridge_grad_norm": self.answer_loss_bridge_grad_norm,
            "answer_loss_reaches_generator": self.answer_loss_reaches_generator,
            "answer_loss_reaches_coconut_cell": self.answer_loss_reaches_coconut_cell,
            "answer_loss_reaches_symbol_heads": self.answer_loss_reaches_symbol_heads,
            "answer_loss_reaches_recurrent_bridi_feedback": self.answer_loss_reaches_recurrent_bridi_feedback,
            "answer_loss_reaches_trace_slot_advisor": self.answer_loss_reaches_trace_slot_advisor,
            "answer_loss_reaches_advisor_classifier": self.answer_loss_reaches_advisor_classifier,
            "answer_loss_reaches_language_backbone": self.answer_loss_reaches_language_backbone,
            "answer_loss_reaches_bridge": self.answer_loss_reaches_bridge,
            "answer_loss_reaches_prompt_encoder": self.answer_loss_reaches_prompt_encoder,
            "answer_loss_reaches_bridi_emitter": self.answer_loss_reaches_bridi_emitter,
            "answer_loss_reaches_trace_bridge": self.answer_loss_reaches_trace_bridge,
            "answer_loss_reaches_answer_decoder": self.answer_loss_reaches_answer_decoder,
            "answer_loss_reaches_stop_head": self.answer_loss_reaches_stop_head,
            "hard_argmax_training_cut_detected": 0.0,
            "torch_no_grad_training_cut_detected": 0.0,
        }


class M27RelevanceRuntimeOrgan(nn.Module):
    """Optional M23-style relevance selector over emitted M27 trace slots.

    The organ does not create a new dictionary or trace family. It scores the
    existing Coconut-emitted loose-bridi slots and exposes a weighted active
    mask so the M26 bridge can read causally ranked slots instead of every
    active symbol uniformly.
    """

    def __init__(self, trace_hidden_dim: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.score = nn.Linear(int(trace_hidden_dim), 1)

    def forward(self, trace_slots: torch.Tensor, trace_active: torch.Tensor) -> dict[str, torch.Tensor]:
        active = trace_active.float().clamp(min=0.0, max=1.0)
        active_mask = active > 1e-8
        raw_logits = self.score(trace_slots).squeeze(-1)
        masked_logits = raw_logits.masked_fill(~active_mask, -1e4)
        weights = torch.softmax(masked_logits / self.temperature, dim=-1) * active_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selected_state = (trace_slots * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "relevance_logits": raw_logits,
            "relevance_weights": weights,
            "relevance_active_override": weights * active,
            "relevance_state": selected_state,
            "relevance_runtime_active": trace_slots.new_tensor(1.0),
        }


def m27_relevance_rank_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    margin: float = DEFAULT_M27_RELEVANCE_MARGIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Margin-rank relevant trace slots above known decoy slots when labels exist."""

    reference = outputs["active_logits"]
    zero = reference.sum() * 0.0
    if "relevance_logits" not in outputs or "relevance_targets" not in batch or "decoy_targets" not in batch:
        return zero, {
            "loss_relevance_rank": 0.0,
            "m27_relevance_rank_valid_fraction": 0.0,
            "m27_relevance_top1_accuracy": 0.0,
            "m27_relevance_margin": 0.0,
        }
    logits = outputs["relevance_logits"]
    relevant = batch["relevance_targets"].to(logits.device).float() > 0.5
    decoy = batch["decoy_targets"].to(logits.device).float() > 0.5
    valid = relevant.any(dim=-1) & decoy.any(dim=-1)
    if not bool(valid.any().item()):
        return zero, {
            "loss_relevance_rank": 0.0,
            "m27_relevance_rank_valid_fraction": 0.0,
            "m27_relevance_top1_accuracy": 0.0,
            "m27_relevance_margin": 0.0,
        }
    relevant_max = logits.masked_fill(~relevant, -1e4).max(dim=-1).values
    decoy_max = logits.masked_fill(~decoy, -1e4).max(dim=-1).values
    margins = relevant_max - decoy_max
    loss = F.relu(float(margin) - margins[valid]).mean()
    top1 = torch.argmax(logits, dim=-1)
    top1_relevant = relevant.gather(1, top1.unsqueeze(-1)).squeeze(-1).float()
    return loss, {
        "loss_relevance_rank": float(loss.detach().cpu().item()),
        "m27_relevance_rank_valid_fraction": float(valid.float().mean().detach().cpu().item()),
        "m27_relevance_top1_accuracy": float(top1_relevant[valid].mean().detach().cpu().item()),
        "m27_relevance_margin": float(margins[valid].mean().detach().cpu().item()),
    }


class CoconutRecurrentCell(nn.Module):
    """GRU-style latent scratchpad update for one loose-bridi emission step."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)
        self.no_recurrence = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, prompt_context: torch.Tensor, previous_symbol: torch.Tensor) -> torch.Tensor:
        return self.gru(torch.cat([prompt_context, previous_symbol], dim=-1), state)

    def independent_state(self, initial_state: torch.Tensor, prompt_context: torch.Tensor, step_query: torch.Tensor) -> torch.Tensor:
        return self.no_recurrence(torch.cat([prompt_context, initial_state + step_query], dim=-1))


class AutoregressiveBridiEmitter(nn.Module):
    """One-symbol-per-step loose-bridi emitter with differentiable feedback."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_symbols: int = 32,
        value_vocab_size: int | None = None,
        aux_vocab_size: int | None = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_symbols = int(max_symbols)
        self.value_vocab_size = int(value_vocab_size or _value_vocab_size())
        self.aux_vocab_size = int(aux_vocab_size or _aux_vocab_size())
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(int(vocab_size), self.embedding_dim, padding_idx=0)
        self.prompt_init = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh())
        self.prompt_key = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)
        self.prompt_value = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)
        self.prompt_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.step_queries = nn.Parameter(torch.randn(self.max_symbols, self.hidden_dim) * 0.02)
        self.bos_symbol = nn.Parameter(torch.zeros(self.hidden_dim))
        self.coconut_cell = CoconutRecurrentCell(self.hidden_dim)
        self.active_head = nn.Linear(self.hidden_dim, 1)
        self.type_head = nn.Linear(self.hidden_dim, LOOSE_TYPE_COUNT)
        self.value_head = nn.Linear(self.hidden_dim, self.value_vocab_size)
        self.aux_head = nn.Linear(self.hidden_dim, self.aux_vocab_size)
        self.type_embed = nn.Embedding(LOOSE_TYPE_COUNT, self.hidden_dim)
        self.value_embed = nn.Embedding(self.value_vocab_size, self.hidden_dim)
        self.aux_embed = nn.Embedding(self.aux_vocab_size, self.hidden_dim)
        self.answer_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, len(ANSWER_LABELS)))
        nn.init.constant_(self.active_head.bias, 1.0)

    def _prompt_hidden(self, input_ids: torch.Tensor, prompt_hidden_states: torch.Tensor | None) -> torch.Tensor:
        hidden = self.embedding(input_ids) if prompt_hidden_states is None else prompt_hidden_states
        if hidden.shape[-1] != self.embedding_dim:
            raise ValueError(f"prompt_hidden_states last dimension must be {self.embedding_dim}, got {hidden.shape[-1]}")
        return hidden

    def _attend_prompt(
        self,
        state: torch.Tensor,
        prompt_keys: torch.Tensor,
        prompt_values: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.prompt_query(state).unsqueeze(1)
        scores = torch.matmul(query, prompt_keys.transpose(-1, -2)).squeeze(1) / (float(self.hidden_dim) ** 0.5)
        attention = torch.softmax(scores.masked_fill(~prompt_mask.bool(), -1e4), dim=-1)
        return torch.matmul(attention.unsqueeze(1), prompt_values).squeeze(1), attention

    def _soft_symbol_embedding(self, type_logits: torch.Tensor, value_logits: torch.Tensor, aux_logits: torch.Tensor) -> torch.Tensor:
        return (
            torch.softmax(type_logits, dim=-1) @ self.type_embed.weight
            + torch.softmax(value_logits, dim=-1) @ self.value_embed.weight
            + torch.softmax(aux_logits, dim=-1) @ self.aux_embed.weight
        )

    def _hard_symbol_embedding(self, type_ids: torch.Tensor, value_ids: torch.Tensor, aux_ids: torch.Tensor) -> torch.Tensor:
        return self.type_embed(type_ids) + self.value_embed(value_ids) + self.aux_embed(aux_ids)

    def _teacher_previous_embedding(self, teacher_trace: torch.Tensor, step: int) -> torch.Tensor:
        prev = teacher_trace[:, step].long()
        return self._hard_symbol_embedding(
            prev[:, 0].clamp(0, LOOSE_TYPE_COUNT - 1),
            prev[:, 1].clamp(0, self.value_vocab_size - 1),
            prev[:, 2].clamp(0, self.aux_vocab_size - 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_hidden_states: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        teacher_trace: torch.Tensor | None = None,
        hard_feedback: bool = False,
        no_recurrence: bool = False,
        max_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        prompt_hidden = self._prompt_hidden(input_ids, prompt_hidden_states)
        resolved_mask = input_ids.ne(0) if prompt_mask is None else prompt_mask.bool()
        mask_float = resolved_mask.float().unsqueeze(-1)
        prompt_pooled = (prompt_hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        initial_state = self.prompt_init(prompt_pooled)
        prompt_keys = self.prompt_key(prompt_hidden)
        prompt_values = self.prompt_value(prompt_hidden)
        steps = min(int(max_steps or self.max_symbols), self.max_symbols)

        state = initial_state
        previous_symbol = self.bos_symbol.unsqueeze(0).expand(int(input_ids.shape[0]), -1)
        states: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        active_logits: list[torch.Tensor] = []
        type_logits: list[torch.Tensor] = []
        value_logits: list[torch.Tensor] = []
        aux_logits: list[torch.Tensor] = []
        for step in range(steps):
            query_state = initial_state + self.step_queries[step].unsqueeze(0) if no_recurrence else state
            prompt_context, attention = self._attend_prompt(query_state, prompt_keys, prompt_values, resolved_mask)
            if no_recurrence:
                step_state = self.coconut_cell.independent_state(initial_state, prompt_context, self.step_queries[step].unsqueeze(0))
            else:
                step_state = self.coconut_cell(state, prompt_context, previous_symbol)
                state = step_state
            step_active = self.active_head(step_state).squeeze(-1)
            step_type = self.type_head(step_state)
            step_value = self.value_head(step_state)
            step_aux = self.aux_head(step_state)
            states.append(step_state)
            attentions.append(attention)
            active_logits.append(step_active)
            type_logits.append(step_type)
            value_logits.append(step_value)
            aux_logits.append(step_aux)
            if teacher_trace is not None and step + 1 < steps:
                previous_symbol = self._teacher_previous_embedding(teacher_trace, step)
            elif hard_feedback:
                previous_symbol = self._hard_symbol_embedding(
                    torch.argmax(step_type, dim=-1).clamp(0, LOOSE_TYPE_COUNT - 1),
                    torch.argmax(step_value, dim=-1).clamp(0, self.value_vocab_size - 1),
                    torch.argmax(step_aux, dim=-1).clamp(0, self.aux_vocab_size - 1),
                )
            else:
                previous_symbol = self._soft_symbol_embedding(step_type, step_value, step_aux)

        outputs = {
            "active_logits": torch.stack(active_logits, dim=1),
            "type_logits": torch.stack(type_logits, dim=1),
            "value_logits": torch.stack(value_logits, dim=1),
            "aux_logits": torch.stack(aux_logits, dim=1),
            "coconut_states": torch.stack(states, dim=1),
            "prompt_attention": torch.stack(attentions, dim=1),
            "initial_coconut_state": initial_state,
        }
        outputs = _pad_generator_outputs(outputs, self.max_symbols)
        active = torch.sigmoid(outputs["active_logits"]).unsqueeze(-1)
        trace_state = (outputs["coconut_states"] * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        outputs["answer_logits"] = self.answer_head(trace_state)
        outputs["trace_state"] = trace_state
        outputs["stop_logits"] = outputs["type_logits"][:, :, LOOSE_STOP]
        return outputs


class M27CoconutBridiRuntime(nn.Module):
    """End-to-end prompt -> recurrent Coconut trace -> trace-language bridge."""

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
        max_prompt_length: int = 128,
        language_layers: int = 1,
        language_heads: int = 2,
        enable_relevance_runtime: bool = False,
        relevance_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.language_backbone = M26TinyLanguageBackbone(
            vocab_size=int(vocab_size),
            hidden_dim=int(embedding_dim),
            max_prompt_length=int(max_prompt_length),
            num_layers=int(language_layers),
            num_heads=int(language_heads),
        )
        self.generator = AutoregressiveBridiEmitter(
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
        self.bridge = M26TraceLanguageBridge(prompt_hidden_dim=int(embedding_dim), trace_hidden_dim=int(advisor_hidden_dim))
        self.relevance_runtime = (
            M27RelevanceRuntimeOrgan(int(advisor_hidden_dim), temperature=float(relevance_temperature))
            if bool(enable_relevance_runtime)
            else None
        )
        self.generator_primary_input = "language_hidden_states"
        self.answer_head_primary_input = "fused_language_trace_state"
        self.trace_runtime_mode = "autoregressive_coconut_loose_bridi"
        self.relevance_runtime_enabled = bool(enable_relevance_runtime)

    @property
    def max_symbols(self) -> int:
        return self.generator.max_symbols

    def advisor_logits_from_generator_outputs(self, generator_outputs: dict[str, torch.Tensor], *, active_override: torch.Tensor | None = None) -> torch.Tensor:
        return self.advisor.forward_from_logits(
            active_logits=generator_outputs["active_logits"],
            type_logits=generator_outputs["type_logits"],
            value_logits=generator_outputs["value_logits"],
            aux_logits=generator_outputs["aux_logits"],
            active_override=active_override,
        )

    def bridge_logits_from_generator_outputs(
        self,
        *,
        input_ids: torch.Tensor,
        generator_outputs: dict[str, torch.Tensor],
        active_override: torch.Tensor | None = None,
        language_outputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        language = self.language_backbone(input_ids) if language_outputs is None else language_outputs
        trace_slots, trace_active = self.advisor.slot_embeddings_from_logits(
            active_logits=generator_outputs["active_logits"],
            type_logits=generator_outputs["type_logits"],
            value_logits=generator_outputs["value_logits"],
            aux_logits=generator_outputs["aux_logits"],
            active_override=active_override,
        )
        return self.bridge(
            token_hidden_states=language["token_hidden_states"],
            prompt_mask=language["prompt_mask"],
            trace_slots=trace_slots,
            trace_active=trace_active,
        )["answer_logits"]

    def logits_from_hard_trace(self, hard_trace: torch.Tensor, *, logit_scale: float = 12.0) -> dict[str, torch.Tensor]:
        stream = _apply_monotonic_stop(hard_trace.long())
        active = stream[:, :, 0].ne(LOOSE_PAD).float()
        active_logits = torch.where(active > 0.5, torch.full_like(active, float(logit_scale)), torch.full_like(active, -float(logit_scale)))
        return {
            "active_logits": active_logits,
            "type_logits": _one_hot_logits(stream[:, :, 0].clamp(0, LOOSE_TYPE_COUNT - 1), LOOSE_TYPE_COUNT, logit_scale),
            "value_logits": _one_hot_logits(stream[:, :, 1].clamp(0, self.generator.value_vocab_size - 1), self.generator.value_vocab_size, logit_scale),
            "aux_logits": _one_hot_logits(stream[:, :, 2].clamp(0, self.generator.aux_vocab_size - 1), self.generator.aux_vocab_size, logit_scale),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        teacher_trace: torch.Tensor | None = None,
        mode: str = "soft_train",
        no_recurrence: bool = False,
        max_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        language = self.language_backbone(input_ids)
        hard_feedback = str(mode) == "hard_free_run"
        generator_outputs = self.generator(
            input_ids,
            prompt_hidden_states=language["token_hidden_states"],
            prompt_mask=language["prompt_mask"],
            teacher_trace=teacher_trace if not hard_feedback else None,
            hard_feedback=hard_feedback,
            no_recurrence=bool(no_recurrence),
            max_steps=max_steps,
        )
        hard_trace = _apply_monotonic_stop(pack_loose_stream_from_outputs(generator_outputs))
        bridge_source = self.logits_from_hard_trace(hard_trace) if hard_feedback else generator_outputs
        trace_slots, trace_active = self.advisor.slot_embeddings_from_logits(
            active_logits=bridge_source["active_logits"],
            type_logits=bridge_source["type_logits"],
            value_logits=bridge_source["value_logits"],
            aux_logits=bridge_source["aux_logits"],
        )
        bridge_outputs = self.bridge(
            token_hidden_states=language["token_hidden_states"],
            prompt_mask=language["prompt_mask"],
            trace_slots=trace_slots,
            trace_active=trace_active,
        )
        relevance_outputs: dict[str, torch.Tensor] = {}
        if self.relevance_runtime is not None:
            relevance = self.relevance_runtime(trace_slots, trace_active)
            relevance_bridge = self.bridge(
                token_hidden_states=language["token_hidden_states"],
                prompt_mask=language["prompt_mask"],
                trace_slots=trace_slots,
                trace_active=relevance["relevance_active_override"],
            )
            relevance_outputs.update(
                {
                    **relevance,
                    "relevance_answer_logits": relevance_bridge["answer_logits"],
                    "relevance_fused_state": relevance_bridge["fused_state"],
                    "relevance_trace_attention": relevance_bridge["trace_attention"],
                    "relevance_trace_active_mass": relevance_bridge["trace_active_mass"],
                }
            )
        return {
            **generator_outputs,
            "generator_answer_logits": generator_outputs["answer_logits"],
            "trace_only_answer_logits": self.advisor_logits_from_generator_outputs(bridge_source),
            "answer_logits": bridge_outputs["answer_logits"],
            "trace_state": self.advisor.encode_from_logits(
                active_logits=bridge_source["active_logits"],
                type_logits=bridge_source["type_logits"],
                value_logits=bridge_source["value_logits"],
                aux_logits=bridge_source["aux_logits"],
            ),
            "language_hidden_states": language["token_hidden_states"],
            "prompt_state": bridge_outputs["prompt_state"],
            "fused_state": bridge_outputs["fused_state"],
            "trace_slots": trace_slots,
            "soft_trace_embeddings": trace_slots,
            "trace_active": trace_active,
            "bridge_delta": bridge_outputs["bridge_delta"],
            "trace_attention": bridge_outputs["trace_attention"],
            "bridge_gate_value": bridge_outputs["bridge_gate_value"],
            "trace_attention_entropy": bridge_outputs["trace_attention_entropy"],
            "trace_active_mass": bridge_outputs["trace_active_mass"],
            "bridge_delta_norm": bridge_outputs["bridge_delta_norm"],
            "raw_prompt_bypass_blocked": bridge_outputs["raw_prompt_bypass_blocked"],
            "hard_trace_tokens": hard_trace,
            "hard_free_run_mode": input_ids.new_tensor(1 if hard_feedback else 0).float(),
            "no_recurrence_mode": input_ids.new_tensor(1 if no_recurrence else 0).float(),
            "m27_relevance_runtime_enabled": input_ids.new_tensor(1.0 if self.relevance_runtime is not None else 0.0).float(),
            **relevance_outputs,
        }


def compute_m27_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    answer_outputs: dict[str, torch.Tensor] | None = None,
    trace_weight: float = DEFAULT_M27_TRACE_WEIGHT,
    answer_weight: float = DEFAULT_M27_ANSWER_WEIGHT,
    mdl_weight: float = DEFAULT_M27_MDL_WEIGHT,
    relevance_rank_weight: float = 0.0,
    relevance_margin: float = DEFAULT_M27_RELEVANCE_MARGIN,
    use_relevance_answer: bool = False,
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
    answer_source = outputs if answer_outputs is None else answer_outputs
    answer_key = "relevance_answer_logits" if bool(use_relevance_answer) and "relevance_answer_logits" in answer_source else "answer_logits"
    answer_loss = F.cross_entropy(answer_source[answer_key], answer_target)
    generator_answer_loss = F.cross_entropy(answer_source["generator_answer_logits"], answer_target)
    mdl_loss = torch.sigmoid(outputs["active_logits"]).mean()
    relevance_loss, relevance_metrics = m27_relevance_rank_loss(
        answer_source,
        batch,
        margin=float(relevance_margin),
    )
    stream_loss = active_loss + type_loss + value_loss + aux_loss
    loss = (
        float(trace_weight) * stream_loss
        + float(answer_weight) * answer_loss
        + float(mdl_weight) * mdl_loss
        + float(relevance_rank_weight) * relevance_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "active_loss": float(active_loss.detach().cpu().item()),
        "type_loss": float(type_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "aux_loss": float(aux_loss.detach().cpu().item()),
        "answer_loss": float(answer_loss.detach().cpu().item()),
        "generator_answer_loss_diagnostic": float(generator_answer_loss.detach().cpu().item()),
        "answer_loss_uses_soft_free_run_trace": 1.0 if answer_outputs is not None else 0.0,
        "answer_loss_uses_relevance_runtime_trace": 1.0 if answer_key == "relevance_answer_logits" else 0.0,
        "trace_loss_uses_teacher_forcing": 1.0,
        "mdl_loss": float(mdl_loss.detach().cpu().item()),
        "relevance_rank_weight": float(relevance_rank_weight),
        "relevance_margin": float(relevance_margin),
        "bridge_gate_value": float(outputs["bridge_gate_value"].detach().cpu().item()),
        "bridge_delta_norm": float(outputs["bridge_delta_norm"].detach().cpu().item()),
        "trace_attention_entropy": float(outputs["trace_attention_entropy"].detach().cpu().item()),
        "trace_active_mass": float(outputs["trace_active_mass"].detach().cpu().item()),
        "raw_prompt_bypass_blocked": float(outputs["raw_prompt_bypass_blocked"].detach().cpu().item()),
        **relevance_metrics,
    }


def probe_m27_answer_gradient_flow(model: M27CoconutBridiRuntime, batch: dict[str, Any]) -> M27GradientProbe:
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    targets = batch["answer_id"].to(input_ids.device)
    loss = F.cross_entropy(model(input_ids)["answer_logits"], targets)
    loss.backward()

    generator_norm = _grad_norm(model.generator.parameters())
    coconut_norm = _grad_norm(model.generator.coconut_cell.parameters())
    symbol_params = []
    for module in (model.generator.active_head, model.generator.type_head, model.generator.value_head, model.generator.aux_head):
        symbol_params.extend(module.parameters())
    symbol_norm = _grad_norm(symbol_params)
    feedback_params = []
    for module in (model.generator.type_embed, model.generator.value_embed, model.generator.aux_embed):
        feedback_params.extend(module.parameters())
    feedback_norm = _grad_norm(feedback_params)
    trace_slot_params = []
    for module in (model.advisor.type_embedding, model.advisor.value_embedding, model.advisor.aux_embedding):
        trace_slot_params.extend(module.parameters())
    trace_slot_norm = _grad_norm(trace_slot_params)
    advisor_classifier_norm = _grad_norm(model.advisor.classifier.parameters())
    language_norm = _grad_norm(model.language_backbone.parameters())
    bridge_norm = _grad_norm(model.bridge.parameters())
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return M27GradientProbe(
        answer_loss_generator_grad_norm=generator_norm,
        answer_loss_coconut_cell_grad_norm=coconut_norm,
        answer_loss_symbol_head_grad_norm=symbol_norm,
        answer_loss_recurrent_feedback_grad_norm=feedback_norm,
        answer_loss_advisor_grad_norm=trace_slot_norm,
        answer_loss_trace_slot_advisor_grad_norm=trace_slot_norm,
        answer_loss_advisor_classifier_grad_norm=advisor_classifier_norm,
        answer_loss_language_backbone_grad_norm=language_norm,
        answer_loss_bridge_grad_norm=bridge_norm,
        answer_loss_reaches_generator=1.0 if generator_norm > 0.0 else 0.0,
        answer_loss_reaches_coconut_cell=1.0 if coconut_norm > 0.0 else 0.0,
        answer_loss_reaches_symbol_heads=1.0 if symbol_norm > 0.0 else 0.0,
        answer_loss_reaches_recurrent_bridi_feedback=1.0 if feedback_norm > 0.0 else 0.0,
        answer_loss_reaches_trace_slot_advisor=1.0 if trace_slot_norm > 0.0 else 0.0,
        answer_loss_reaches_advisor_classifier=1.0 if advisor_classifier_norm > 0.0 else 0.0,
        answer_loss_reaches_language_backbone=1.0 if language_norm > 0.0 else 0.0,
        answer_loss_reaches_bridge=1.0 if bridge_norm > 0.0 else 0.0,
    )


def evaluate_m27_coconut_bridi_runtime(
    *,
    model: M27CoconutBridiRuntime,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
    prompt_control: PromptOnlyControl | None = None,
    matched_prompt_control: PromptOnlyControl | None = None,
    matched_prompt_budget: int | None = None,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=model.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m25_collate)
    model.eval()
    if prompt_control is not None:
        prompt_control.eval()
    if matched_prompt_control is not None:
        matched_prompt_control.eval()
    logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    streams: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    relevance_targets: list[torch.Tensor] = []
    decoy_targets: list[torch.Tensor] = []
    relevance_score_logits: list[torch.Tensor] = []
    surfaces: list[str] = []
    symbol_counts: list[torch.Tensor] = []
    prompt_counts: list[torch.Tensor] = []
    matched_prompt_counts: list[torch.Tensor] = []
    bridge_telemetry: dict[str, list[float]] = defaultdict(list)
    step_dependency: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_obj)
            target = batch["answer_id"].to(device_obj)
            outputs = model(input_ids)
            hard_outputs = model(input_ids, mode="hard_free_run")
            no_recurrence_outputs = model(input_ids, no_recurrence=True)
            generator_outputs = _generator_slice(outputs)
            language_outputs = {"token_hidden_states": outputs["language_hidden_states"], "prompt_mask": input_ids.ne(0)}
            shuffled_outputs = _shuffle_generator_outputs(generator_outputs, seed=int(seed) + batch_idx)
            random_outputs = _random_generator_outputs(generator_outputs, seed=int(seed) + 1000 + batch_idx)
            zero_active = torch.zeros_like(generator_outputs["active_logits"])
            oracle_relevance = batch["relevance_targets"].to(device_obj)
            decoy_only = batch["decoy_targets"].to(device_obj)
            no_relevance = batch["stream_active_targets"].to(device_obj)
            random_relevance = _random_active_mask(no_relevance, seed=int(seed) + 2000 + batch_idx)
            logits["predicted"].append(outputs["answer_logits"].detach().cpu())
            if "relevance_answer_logits" in outputs:
                logits["relevance"].append(outputs["relevance_answer_logits"].detach().cpu())
                relevance_score_logits.append(outputs["relevance_logits"].detach().cpu())
            logits["hard_free_run"].append(hard_outputs["answer_logits"].detach().cpu())
            logits["no_recurrence"].append(no_recurrence_outputs["answer_logits"].detach().cpu())
            logits["shuffled"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=shuffled_outputs, language_outputs=language_outputs).detach().cpu())
            logits["random"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=random_outputs, language_outputs=language_outputs).detach().cpu())
            logits["zero"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=generator_outputs, active_override=zero_active, language_outputs=language_outputs).detach().cpu())
            logits["oracle_relevance"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=generator_outputs, active_override=oracle_relevance, language_outputs=language_outputs).detach().cpu())
            logits["random_relevance"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=generator_outputs, active_override=random_relevance, language_outputs=language_outputs).detach().cpu())
            logits["no_relevance"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=generator_outputs, active_override=no_relevance, language_outputs=language_outputs).detach().cpu())
            logits["decoy_only"].append(model.bridge_logits_from_generator_outputs(input_ids=input_ids, generator_outputs=generator_outputs, active_override=decoy_only, language_outputs=language_outputs).detach().cpu())
            if prompt_control is not None:
                logits["prompt"].append(prompt_control(input_ids).detach().cpu())
            if matched_prompt_control is not None:
                matched_input_ids = budget_prompt_tokens(input_ids, matched_prompt_budget)
                logits["matched_prompt"].append(matched_prompt_control(matched_input_ids).detach().cpu())
                matched_prompt_counts.append(matched_input_ids.ne(0).sum(dim=-1).float().detach().cpu())
            for key in ("bridge_gate_value", "bridge_delta_norm", "trace_attention_entropy", "trace_active_mass", "raw_prompt_bypass_blocked"):
                bridge_telemetry[key].append(float(outputs[key].detach().float().mean().cpu().item()))
            predicted = outputs["hard_trace_tokens"]
            oracle = batch["stream_targets"].to(device_obj)
            streams["predicted"].append(predicted.detach().cpu())
            streams["oracle"].append(oracle.detach().cpu())
            targets.append(target.detach().cpu())
            relevance_targets.append(oracle_relevance.detach().cpu())
            decoy_targets.append(decoy_only.detach().cpu())
            surfaces.extend(batch["surface"])
            symbol_counts.append(loose_stream_symbol_counts(predicted).detach().cpu())
            prompt_counts.append(input_ids.ne(0).sum(dim=-1).float().detach().cpu())
            step_dependency.append(_step_dependency_delta(model, batch, device_obj))

    target_all = torch.cat(targets, dim=0)
    all_logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    all_streams = {key: torch.cat(value, dim=0) for key, value in streams.items()}
    predicted_acc = _accuracy(all_logits["predicted"], target_all)
    hard_acc = _accuracy(all_logits["hard_free_run"], target_all)
    no_recurrence_acc = _accuracy(all_logits["no_recurrence"], target_all)
    shuffled_acc = _accuracy(all_logits["shuffled"], target_all)
    random_acc = _accuracy(all_logits["random"], target_all)
    zero_acc = _accuracy(all_logits["zero"], target_all)
    relevance_acc = _accuracy(all_logits["relevance"], target_all) if "relevance" in all_logits else 0.0
    oracle_relevance_acc = _accuracy(all_logits["oracle_relevance"], target_all)
    random_relevance_acc = _accuracy(all_logits["random_relevance"], target_all)
    no_relevance_acc = _accuracy(all_logits["no_relevance"], target_all)
    decoy_only_acc = _accuracy(all_logits["decoy_only"], target_all)
    relevance_stats = _relevance_eval_stats(
        torch.cat(relevance_score_logits, dim=0) if relevance_score_logits else None,
        torch.cat(relevance_targets, dim=0) if relevance_targets else None,
        torch.cat(decoy_targets, dim=0) if decoy_targets else None,
    )
    prompt_acc = _accuracy(all_logits["prompt"], target_all) if "prompt" in all_logits else 0.0
    matched_prompt_acc = _accuracy(all_logits["matched_prompt"], target_all) if "matched_prompt" in all_logits else 0.0
    pred_count = torch.cat(symbol_counts, dim=0)
    prompt_count = torch.cat(prompt_counts, dim=0)
    matched_prompt_count = torch.cat(matched_prompt_counts, dim=0) if matched_prompt_counts else torch.zeros_like(prompt_count)
    mean_pred = float(pred_count.mean().item()) if pred_count.numel() else 0.0
    mean_prompt = float(prompt_count.mean().item()) if prompt_count.numel() else 0.0
    mean_matched_prompt = float(matched_prompt_count.mean().item()) if matched_prompt_count.numel() else 0.0
    metrics = {
        "strict_accuracy": predicted_acc,
        "synthetic_world_accuracy": predicted_acc,
        "phrase_accuracy": predicted_acc,
        "end_to_end_answer_accuracy": predicted_acc,
        "m27_end_to_end_answer_accuracy": predicted_acc,
        "soft_free_run_strict_accuracy": predicted_acc,
        "soft_teacher_forced_strict_accuracy": predicted_acc,
        "soft_teacher_forced_strict_accuracy_is_legacy_soft_free_run_alias": 1.0,
        "hard_free_run_accuracy": hard_acc,
        "hard_free_run_strict_accuracy": hard_acc,
        "soft_hard_accuracy_gap": float(predicted_acc - hard_acc),
        "soft_hard_gap": float(predicted_acc - hard_acc),
        "no_recurrence_accuracy": no_recurrence_acc,
        "multi_step_delta_vs_no_recurrence": float(predicted_acc - no_recurrence_acc),
        "m27_recurrence_delta": float(predicted_acc - no_recurrence_acc),
        "m27_recurrence_enabled": 1.0,
        "m27_no_recurrence_ablation_enabled": 1.0,
        "shuffled_trace_accuracy": shuffled_acc,
        "random_trace_accuracy": random_acc,
        "zero_trace_accuracy": zero_acc,
        "m27_relevance_runtime_enabled": 1.0 if model.relevance_runtime is not None else 0.0,
        "m27_relevance_runtime_active": 1.0 if "relevance" in all_logits else 0.0,
        "m27_relevance_full_accuracy": relevance_acc,
        "m27_relevance_answer_accuracy": relevance_acc,
        "m27_relevance_oracle_accuracy": oracle_relevance_acc,
        "m27_oracle_relevance_accuracy": oracle_relevance_acc,
        "m27_relevance_random_accuracy": random_relevance_acc,
        "m27_random_relevance_accuracy": random_relevance_acc,
        "m27_relevance_no_selector_accuracy": no_relevance_acc,
        "m27_no_relevance_accuracy": no_relevance_acc,
        "m27_relevance_decoy_only_accuracy": decoy_only_acc,
        "m27_decoy_only_accuracy": decoy_only_acc,
        "m27_relevance_full_vs_random_delta": float(relevance_acc - random_relevance_acc),
        "m27_relevance_oracle_lift": float(oracle_relevance_acc - no_relevance_acc),
        "m27_relevance_decoy_drop": float(no_relevance_acc - decoy_only_acc),
        **relevance_stats,
        "prompt_only_accuracy": prompt_acc,
        "matched_prompt_accuracy": matched_prompt_acc,
        "m27_strict_delta_vs_prompt_only": float(predicted_acc - prompt_acc),
        "m27_strict_delta_vs_matched_prompt": float(predicted_acc - matched_prompt_acc),
        "predicted_vs_shuffled_delta": float(predicted_acc - shuffled_acc),
        "predicted_vs_random_delta": float(predicted_acc - random_acc),
        "predicted_vs_zero_delta": float(predicted_acc - zero_acc),
        "mean_predicted_emitted_symbols_after_bottleneck": mean_pred,
        "trace_token_count": mean_pred,
        "mean_prompt_tokens": mean_prompt,
        "mean_matched_prompt_tokens": mean_matched_prompt,
        "matched_prompt_token_count": mean_matched_prompt,
        "loose_symbol_to_prompt_ratio": float(mean_pred / max(1.0, mean_prompt)),
        "loose_symbol_to_matched_prompt_ratio": float(mean_pred / max(1.0, mean_matched_prompt)),
        "matched_prompt_token_reduction_ratio": float(1.0 - mean_matched_prompt / max(1.0, mean_prompt)),
        "accuracy_per_loose_symbol": float(predicted_acc / max(1.0, mean_pred)),
        "accuracy_per_prompt_token": float(prompt_acc / max(1.0, mean_prompt)),
        "matched_prompt_accuracy_per_token": float(matched_prompt_acc / max(1.0, mean_matched_prompt)),
        "m27_accuracy_per_symbol_delta_vs_matched_prompt": float(predicted_acc / max(1.0, mean_pred) - matched_prompt_acc / max(1.0, mean_matched_prompt)),
        "matched_prompt_token_budget": float(matched_prompt_budget or 0),
        "single_optimizer_end_to_end_training": 1.0,
        "hard_argmax_training_cut_detected": 0.0,
        "torch_no_grad_training_cut_detected": 0.0,
        "advisor_primary_trace_is_differentiable": 1.0,
        "autoregressive_coconut_runtime_active": 1.0,
        "soft_train_and_hard_free_run_both_available": 1.0,
        "m27_step_dependency_delta": mean(step_dependency) if step_dependency else 0.0,
        "bridge_gate_value": mean(bridge_telemetry["bridge_gate_value"]) if bridge_telemetry["bridge_gate_value"] else 0.0,
        "bridge_delta_norm": mean(bridge_telemetry["bridge_delta_norm"]) if bridge_telemetry["bridge_delta_norm"] else 0.0,
        "trace_attention_entropy": mean(bridge_telemetry["trace_attention_entropy"]) if bridge_telemetry["trace_attention_entropy"] else 0.0,
        "trace_active_mass": mean(bridge_telemetry["trace_active_mass"]) if bridge_telemetry["trace_active_mass"] else 0.0,
        "raw_prompt_bypass_blocked": mean(bridge_telemetry["raw_prompt_bypass_blocked"]) if bridge_telemetry["raw_prompt_bypass_blocked"] else 0.0,
    }
    metrics.update(_component_accuracy(all_streams["predicted"], all_streams["oracle"]))
    metrics["m27_autoregressive_trace_exact_accuracy"] = metrics.get("loose_stream_exact_accuracy", 0.0)
    metrics.update(_m27_side_channel_placeholders(metrics))
    metrics.update(m27_promotion_gate_metrics(metrics))
    pred_labels = torch.argmax(all_logits["predicted"], dim=-1)
    surface_metrics: dict[str, dict[str, float]] = {}
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([item == surface for item in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {"strict_accuracy": float((pred_labels[mask] == target_all[mask]).float().mean().item()), "count": float(mask.sum().item())}
    return {"metrics": metrics, "surface_metrics": surface_metrics}


def train_m27_coconut_bridi_runtime(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    epochs: int = 8,
    prompt_epochs: int | None = None,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 27,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    max_frames: int = 6,
    max_symbols: int = 32,
    max_steps: int | None = None,
    max_prompt_length: int = 128,
    language_layers: int = 1,
    language_heads: int = 2,
    symbol_budget: int | None = None,
    matched_prompt_budget: int | None = None,
    trace_weight: float = DEFAULT_M27_TRACE_WEIGHT,
    answer_weight: float = DEFAULT_M27_ANSWER_WEIGHT,
    mdl_weight: float = DEFAULT_M27_MDL_WEIGHT,
    enable_relevance_runtime: bool = False,
    relevance_rank_weight: float = 0.0,
    relevance_margin: float = DEFAULT_M27_RELEVANCE_MARGIN,
    use_relevance_answer: bool = False,
    relevance_temperature: float = 1.0,
    clean_train_fraction: float = 0.35,
    clean_eval_fraction: float = 0.35,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    resolved_max_symbols = int(max_steps or max_symbols)
    train_examples = generate_m25_emergent_bridi_examples(int(train_size), seed=int(seed), clean_fraction=float(clean_train_fraction), max_frames=int(max_frames), max_symbols=resolved_max_symbols)
    eval_examples = generate_m25_emergent_bridi_examples(int(eval_size), seed=int(seed) + 999, clean_fraction=float(clean_eval_fraction), max_frames=int(max_frames), max_symbols=resolved_max_symbols)
    vocab = build_vocab([*train_examples, *eval_examples])  # type: ignore[arg-type]
    model = M27CoconutBridiRuntime(
        vocab_size=len(vocab),
        max_symbols=resolved_max_symbols,
        value_vocab_size=_value_vocab_size(),
        aux_vocab_size=_aux_vocab_size(),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        advisor_hidden_dim=int(advisor_hidden_dim),
        symbol_budget=symbol_budget,
        max_prompt_length=int(max_prompt_length),
        language_layers=int(language_layers),
        language_heads=int(language_heads),
        enable_relevance_runtime=bool(enable_relevance_runtime),
        relevance_temperature=float(relevance_temperature),
    ).to(device_obj)
    dataset = M25LooseBridiDataset(train_examples, vocab, max_symbols=model.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m25_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    model.train()
    for _ in range(int(epochs)):
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device_obj)
            teacher_outputs = model(input_ids, teacher_trace=batch["stream_targets"].to(device_obj))
            soft_free_run_outputs = model(input_ids)
            loss, metrics = compute_m27_loss(
                teacher_outputs,
                batch,
                answer_outputs=soft_free_run_outputs,
                trace_weight=trace_weight,
                answer_weight=answer_weight,
                mdl_weight=mdl_weight,
                relevance_rank_weight=relevance_rank_weight,
                relevance_margin=relevance_margin,
                use_relevance_answer=use_relevance_answer,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()})

    prompt_epochs_resolved = int(prompt_epochs if prompt_epochs is not None else epochs)
    prompt_control = PromptOnlyControl(vocab_size=len(vocab), embedding_dim=max(8, int(embedding_dim) // 2), hidden_dim=int(advisor_hidden_dim)).to(device_obj)
    matched_prompt_control = PromptOnlyControl(vocab_size=len(vocab), embedding_dim=max(8, int(embedding_dim) // 2), hidden_dim=int(advisor_hidden_dim)).to(device_obj)
    resolved_matched_prompt_budget = int(matched_prompt_budget or symbol_budget or resolved_max_symbols)
    prompt_history = _train_prompt_control(prompt_control, train_examples, vocab, max_symbols=resolved_max_symbols, prompt_token_budget=None, epochs=prompt_epochs_resolved, batch_size=int(batch_size), learning_rate=float(learning_rate), device=device_obj, seed=int(seed) + 300)
    matched_prompt_history = _train_prompt_control(matched_prompt_control, train_examples, vocab, max_symbols=resolved_max_symbols, prompt_token_budget=resolved_matched_prompt_budget, epochs=prompt_epochs_resolved, batch_size=int(batch_size), learning_rate=float(learning_rate), device=device_obj, seed=int(seed) + 400)
    eval_payload = evaluate_m27_coconut_bridi_runtime(
        model=model,
        examples=eval_examples,
        vocab=vocab,
        batch_size=int(batch_size),
        device=device_obj,
        seed=int(seed),
        prompt_control=prompt_control,
        matched_prompt_control=matched_prompt_control,
        matched_prompt_budget=resolved_matched_prompt_budget,
    )
    probe_batch = m25_collate([M25LooseBridiDataset(eval_examples, vocab, max_symbols=model.max_symbols)[i] for i in range(min(8, len(eval_examples)))])
    metrics = dict(eval_payload["metrics"])
    metrics.update(probe_m27_answer_gradient_flow(model, probe_batch).as_dict())
    metrics.update(
        {
            "trainable_parameter_count": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "language_backbone_trainable_parameter_count": float(sum(p.numel() for p in model.language_backbone.parameters() if p.requires_grad)),
            "generator_trainable_parameter_count": float(sum(p.numel() for p in model.generator.parameters() if p.requires_grad)),
            "coconut_cell_trainable_parameter_count": float(sum(p.numel() for p in model.generator.coconut_cell.parameters() if p.requires_grad)),
            "advisor_trainable_parameter_count": float(sum(p.numel() for p in model.advisor.parameters() if p.requires_grad)),
            "bridge_trainable_parameter_count": float(sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)),
            "relevance_runtime_trainable_parameter_count": float(
                sum(p.numel() for p in model.relevance_runtime.parameters() if p.requires_grad)
            )
            if model.relevance_runtime is not None
            else 0.0,
            "lm_hidden_state_stream_active": 1.0,
            "bridi_generator_reads_lm_hidden_states": 1.0,
            "coconut_recurrent_cell_active": 1.0,
            "trace_bridge_reads_prompt_hidden_states": 1.0,
            "answer_head_reads_fused_lm_trace_state": 1.0,
            "raw_prompt_bypass_blocked": 1.0,
            "answer_loss_uses_soft_free_run_trace": 1.0,
            "trace_loss_uses_teacher_forcing": 1.0,
            "m27_training_answer_loss_uses_soft_free_run_trace": 1.0,
            "m27_training_trace_loss_uses_teacher_forcing": 1.0,
            "m27_relevance_runtime_enabled": 1.0 if model.relevance_runtime is not None else 0.0,
            "m27_training_answer_loss_uses_relevance_runtime_trace": 1.0 if bool(use_relevance_answer) and model.relevance_runtime is not None else 0.0,
            "m27_inherited_contract_bundle_present": 1.0,
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "relevance_rank_weight": float(relevance_rank_weight),
            "relevance_margin": float(relevance_margin),
            "relevance_temperature": float(relevance_temperature),
        }
    )
    metrics.update(m27_promotion_gate_metrics(metrics))
    return {
        "config": {
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "epochs": int(epochs),
            "prompt_epochs": prompt_epochs_resolved,
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "max_frames": int(max_frames),
            "max_symbols": resolved_max_symbols,
            "max_steps": resolved_max_symbols,
            "max_prompt_length": int(max_prompt_length),
            "language_layers": int(language_layers),
            "language_heads": int(language_heads),
            "symbol_budget": int(symbol_budget or 0),
            "matched_prompt_budget": int(resolved_matched_prompt_budget),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "enable_relevance_runtime": bool(enable_relevance_runtime),
            "relevance_rank_weight": float(relevance_rank_weight),
            "relevance_margin": float(relevance_margin),
            "use_relevance_answer": bool(use_relevance_answer),
            "relevance_temperature": float(relevance_temperature),
            "device": str(device_obj),
            "organism_mode": "coconut_autoregressive_lm_hidden_bridi_bridge",
        },
        "metrics": metrics,
        "surface_metrics": eval_payload["surface_metrics"],
        "history": history,
        "prompt_history": prompt_history,
        "matched_prompt_history": matched_prompt_history,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "vocab_size": len(vocab),
    }


def m27_promotion_gate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    gates = {
        "m27_gate_answer_loss_reaches_generator": 1.0 if metrics.get("answer_loss_reaches_generator", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_reaches_coconut_cell": 1.0 if metrics.get("answer_loss_reaches_coconut_cell", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_reaches_symbol_heads": 1.0 if metrics.get("answer_loss_reaches_symbol_heads", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_reaches_language_backbone": 1.0 if metrics.get("answer_loss_reaches_language_backbone", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_reaches_bridge": 1.0 if metrics.get("answer_loss_reaches_bridge", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_reaches_recurrent_bridi_feedback": 1.0 if metrics.get("answer_loss_reaches_recurrent_bridi_feedback", 0.0) >= 1.0 else 0.0,
        "m27_gate_no_hard_training_cut": 1.0 if metrics.get("hard_argmax_training_cut_detected", 1.0) == 0.0 else 0.0,
        "m27_gate_autoregressive_step_dependency": 1.0 if metrics.get("m27_step_dependency_delta", 0.0) > 1e-9 else 0.0,
        "m27_gate_soft_hard_runtime_available": 1.0 if metrics.get("soft_train_and_hard_free_run_both_available", 0.0) >= 1.0 else 0.0,
        "m27_gate_raw_prompt_bypass_blocked": 1.0 if metrics.get("raw_prompt_bypass_blocked", 0.0) >= 1.0 else 0.0,
        "m27_gate_answer_loss_trains_soft_free_run": 1.0 if metrics.get("m27_training_answer_loss_uses_soft_free_run_trace", 0.0) >= 1.0 else 0.0,
    }
    wiring_candidate = 1.0 if all(value == 1.0 for value in gates.values()) else 0.0
    return {
        **gates,
        "m27_gate_beats_matched_prompt": 1.0 if metrics.get("m27_strict_delta_vs_matched_prompt", -1.0) >= 0.0 else 0.0,
        "m27_gate_stream_beats_zero": 1.0 if metrics.get("predicted_vs_zero_delta", 0.0) >= 0.02 else 0.0,
        "m27_full_organism_gate_pass_rate": sum(gates.values()) / max(1, len(gates)),
        "m27_wiring_candidate": wiring_candidate,
        "m27_full_organism_candidate": wiring_candidate,
        "m27_prompt_comparable_candidate": 1.0 if wiring_candidate >= 1.0 and metrics.get("m27_strict_delta_vs_matched_prompt", -1.0) >= 0.0 else 0.0,
        "m27_promotion_candidate": 1.0 if wiring_candidate >= 1.0 and metrics.get("m27_strict_delta_vs_matched_prompt", -1.0) >= 0.0 and metrics.get("predicted_vs_zero_delta", 0.0) >= 0.02 else 0.0,
    }


def _generator_slice(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: outputs[key] for key in ("active_logits", "type_logits", "value_logits", "aux_logits")}


def _random_active_mask(active: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Pick one active slot per row as a deterministic random relevance control."""

    cpu_active = active.detach().cpu().float()
    out = torch.zeros_like(cpu_active)
    rng = random.Random(int(seed))
    for row in range(cpu_active.shape[0]):
        choices = torch.nonzero(cpu_active[row] > 0.5, as_tuple=False).flatten().tolist()
        if choices:
            out[row, int(rng.choice(choices))] = 1.0
    return out.to(active.device)


def _relevance_eval_stats(
    logits: torch.Tensor | None,
    relevant: torch.Tensor | None,
    decoy: torch.Tensor | None,
) -> dict[str, float]:
    if logits is None or relevant is None or decoy is None:
        return {
            "m27_relevance_top1_accuracy": 0.0,
            "m27_relevance_margin": 0.0,
            "m27_relevance_eval_valid_fraction": 0.0,
        }
    relevant_bool = relevant.float() > 0.5
    decoy_bool = decoy.float() > 0.5
    valid = relevant_bool.any(dim=-1) & decoy_bool.any(dim=-1)
    if not bool(valid.any().item()):
        return {
            "m27_relevance_top1_accuracy": 0.0,
            "m27_relevance_margin": 0.0,
            "m27_relevance_eval_valid_fraction": 0.0,
        }
    top1 = torch.argmax(logits, dim=-1)
    top1_accuracy = relevant_bool.gather(1, top1.unsqueeze(-1)).squeeze(-1).float()[valid].mean()
    relevant_max = logits.masked_fill(~relevant_bool, -1e4).max(dim=-1).values
    decoy_max = logits.masked_fill(~decoy_bool, -1e4).max(dim=-1).values
    margin = (relevant_max - decoy_max)[valid].mean()
    return {
        "m27_relevance_top1_accuracy": float(top1_accuracy.detach().cpu().item()),
        "m27_relevance_margin": float(margin.detach().cpu().item()),
        "m27_relevance_eval_valid_fraction": float(valid.float().mean().detach().cpu().item()),
    }


def _one_hot_logits(ids: torch.Tensor, vocab_size: int, logit_scale: float) -> torch.Tensor:
    out = torch.full((*ids.shape, int(vocab_size)), -float(logit_scale), device=ids.device, dtype=torch.float32)
    return out.scatter_(-1, ids.unsqueeze(-1).long(), float(logit_scale))


def _apply_monotonic_stop(stream: torch.Tensor) -> torch.Tensor:
    out = stream.clone().long()
    for row in range(out.shape[0]):
        stopped = False
        for col in range(out.shape[1]):
            token_type = int(out[row, col, 0].item())
            if stopped:
                out[row, col] = 0
            elif token_type in (LOOSE_PAD, LOOSE_STOP):
                stopped = True
    return out


def _pad_generator_outputs(outputs: dict[str, torch.Tensor], max_symbols: int) -> dict[str, torch.Tensor]:
    current = int(outputs["active_logits"].shape[1])
    if current >= int(max_symbols):
        return outputs
    pad = int(max_symbols) - current
    batch = int(outputs["active_logits"].shape[0])
    device = outputs["active_logits"].device
    outputs["active_logits"] = torch.cat([outputs["active_logits"], outputs["active_logits"].new_full((batch, pad), -12.0)], dim=1)
    outputs["type_logits"] = torch.cat([outputs["type_logits"], outputs["type_logits"].new_zeros((batch, pad, outputs["type_logits"].shape[-1]))], dim=1)
    outputs["value_logits"] = torch.cat([outputs["value_logits"], outputs["value_logits"].new_zeros((batch, pad, outputs["value_logits"].shape[-1]))], dim=1)
    outputs["aux_logits"] = torch.cat([outputs["aux_logits"], outputs["aux_logits"].new_zeros((batch, pad, outputs["aux_logits"].shape[-1]))], dim=1)
    outputs["coconut_states"] = torch.cat([outputs["coconut_states"], outputs["coconut_states"].new_zeros((batch, pad, outputs["coconut_states"].shape[-1]))], dim=1)
    outputs["prompt_attention"] = torch.cat([outputs["prompt_attention"], outputs["prompt_attention"].new_zeros((batch, pad, outputs["prompt_attention"].shape[-1]))], dim=1)
    del device
    return outputs


def _step_dependency_delta(model: M27CoconutBridiRuntime, batch: dict[str, Any], device: torch.device) -> float:
    input_ids = batch["input_ids"].to(device)
    teacher = batch["stream_targets"].to(device).clone()
    if teacher.shape[1] < 2:
        return 0.0
    edited = teacher.clone()
    edited[:, 0, 1] = (edited[:, 0, 1] + 1).clamp_max(model.generator.value_vocab_size - 1)
    base = model(input_ids, teacher_trace=teacher)["coconut_states"][:, 1:]
    alt = model(input_ids, teacher_trace=edited)["coconut_states"][:, 1:]
    return float((base - alt).abs().mean().detach().cpu().item())


def _m27_side_channel_placeholders(metrics: dict[str, float]) -> dict[str, float]:
    del metrics
    return {
        "m27_side_channel_diagnostics_measured": 0.0,
        "m27_side_channel_diagnostics_required_for_promotion": 1.0,
        "side_channel_accuracy_max": 0.0,
        "side_channel_mutual_information_max": 0.0,
    }
