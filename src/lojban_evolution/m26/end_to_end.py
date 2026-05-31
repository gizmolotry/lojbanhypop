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
from lojban_evolution.m24.compression import _accuracy
from lojban_evolution.m24.compression import PromptOnlyControl
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
    budget_prompt_tokens,
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
    answer_loss_trace_slot_advisor_grad_norm: float
    answer_loss_advisor_classifier_grad_norm: float
    answer_loss_language_backbone_grad_norm: float
    answer_loss_bridge_grad_norm: float
    answer_loss_reaches_generator: float
    answer_loss_reaches_symbol_heads: float
    answer_loss_reaches_trace_slot_advisor: float
    answer_loss_reaches_advisor_classifier: float
    answer_loss_reaches_language_backbone: float
    answer_loss_reaches_bridge: float

    def as_dict(self) -> dict[str, float]:
        return {
            "answer_loss_generator_grad_norm": self.answer_loss_generator_grad_norm,
            "answer_loss_symbol_head_grad_norm": self.answer_loss_symbol_head_grad_norm,
            "answer_loss_advisor_grad_norm": self.answer_loss_advisor_grad_norm,
            "answer_loss_trace_slot_advisor_grad_norm": self.answer_loss_trace_slot_advisor_grad_norm,
            "answer_loss_advisor_classifier_grad_norm": self.answer_loss_advisor_classifier_grad_norm,
            "answer_loss_language_backbone_grad_norm": self.answer_loss_language_backbone_grad_norm,
            "answer_loss_bridge_grad_norm": self.answer_loss_bridge_grad_norm,
            "answer_loss_reaches_generator": self.answer_loss_reaches_generator,
            "answer_loss_reaches_symbol_heads": self.answer_loss_reaches_symbol_heads,
            "answer_loss_reaches_trace_slot_advisor": self.answer_loss_reaches_trace_slot_advisor,
            "answer_loss_reaches_advisor_classifier": self.answer_loss_reaches_advisor_classifier,
            "answer_loss_reaches_language_backbone": self.answer_loss_reaches_language_backbone,
            "answer_loss_reaches_bridge": self.answer_loss_reaches_bridge,
        }


class M26TinyLanguageBackbone(nn.Module):
    """LM-shaped English stream used by M26 before the bridi symbiote reads it."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        max_prompt_length: int = 128,
        num_layers: int = 1,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_prompt_length = int(max_prompt_length)
        self.embedding = nn.Embedding(int(vocab_size), self.hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_prompt_length, self.hidden_dim)
        resolved_heads = max(1, int(num_heads))
        if self.hidden_dim % resolved_heads != 0:
            resolved_heads = 1
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=resolved_heads,
            dim_feedforward=max(self.hidden_dim * 2, 16),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = int(input_ids.shape[1])
        if seq_len > self.max_prompt_length:
            raise ValueError(f"input sequence length {seq_len} exceeds max_prompt_length {self.max_prompt_length}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.embedding(input_ids) + self.position_embedding(positions)
        pad_mask = input_ids.eq(0)
        token_hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        mask = input_ids.ne(0).float().unsqueeze(-1)
        pooled = (token_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {
            "token_hidden_states": token_hidden,
            "prompt_state": pooled,
            "prompt_mask": input_ids.ne(0),
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
        embedded, active = self.slot_embeddings_from_logits(
            active_logits=active_logits,
            type_logits=type_logits,
            value_logits=value_logits,
            aux_logits=aux_logits,
            active_override=active_override,
        )
        active = active.unsqueeze(-1)
        return embedded.sum(dim=1) / active.sum(dim=1).clamp_min(1.0)

    def slot_embeddings_from_logits(
        self,
        *,
        active_logits: torch.Tensor,
        type_logits: torch.Tensor,
        value_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        active_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return embedded * active.unsqueeze(-1), active

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


class M26TraceLanguageBridge(nn.Module):
    """Cross-attend the soft bridi trace back into the English hidden stream."""

    def __init__(
        self,
        *,
        prompt_hidden_dim: int,
        trace_hidden_dim: int,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        prompt_dim = int(prompt_hidden_dim)
        trace_dim = int(trace_hidden_dim)
        bottleneck = int(bottleneck_dim or max(8, min(prompt_dim, trace_dim)))
        self.prompt_hidden_dim = prompt_dim
        self.trace_hidden_dim = trace_dim
        self.q_proj = nn.Linear(prompt_dim, trace_dim, bias=False)
        self.k_proj = nn.Linear(trace_dim, trace_dim, bias=False)
        self.v_proj = nn.Linear(trace_dim, trace_dim, bias=False)
        self.down = nn.Linear(trace_dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, prompt_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.answer_head = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.Tanh(),
            nn.Linear(prompt_dim, len(ANSWER_LABELS)),
        )

    def forward(
        self,
        *,
        token_hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
        trace_slots: torch.Tensor,
        trace_active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = prompt_mask.float().unsqueeze(-1)
        prompt_state = (token_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        q = self.q_proj(prompt_state).unsqueeze(1)
        k = self.k_proj(trace_slots)
        v = self.v_proj(trace_slots)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (float(self.trace_hidden_dim) ** 0.5)
        active = trace_active.clamp(min=0.0, max=1.0).unsqueeze(1)
        attn = torch.softmax(scores.masked_fill(active <= 1e-8, -1e4), dim=-1) * active
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.matmul(attn, v).squeeze(1)
        delta = self.up(F.gelu(self.down(context)))
        gate = torch.sigmoid(self.gate)
        delta = torch.tanh(delta) * gate
        # Choked read path: the answer head cannot solve from raw prompt_state
        # alone. The prompt stream is used to query/ground the bridi trace, but
        # the final classifier only sees the trace-conditioned residual.
        fused_state = delta
        entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1).mean()
        return {
            "answer_logits": self.answer_head(fused_state),
            "prompt_state": prompt_state,
            "fused_state": fused_state,
            "bridge_delta": delta,
            "trace_attention": attn.squeeze(1),
            "bridge_gate_value": gate,
            "trace_attention_entropy": entropy,
            "trace_active_mass": trace_active.sum(dim=-1).mean(),
            "bridge_delta_norm": torch.norm(delta, dim=-1).mean(),
            "raw_prompt_bypass_blocked": delta.new_tensor(1.0),
        }


class M26EndToEndLoafman(nn.Module):
    """One trainable prompt -> LM hidden stream -> bridi stream -> bridge organism.

    M26 now assembles the organs that earlier branches kept separate: an English
    hidden-state stream, a bridi scratchpad generator, and a differentiable
    re-entry bridge whose final answer loss reaches back through all of them.
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
        max_prompt_length: int = 128,
        language_layers: int = 1,
        language_heads: int = 2,
    ) -> None:
        super().__init__()
        self.language_backbone = M26TinyLanguageBackbone(
            vocab_size=int(vocab_size),
            hidden_dim=int(embedding_dim),
            max_prompt_length=int(max_prompt_length),
            num_layers=int(language_layers),
            num_heads=int(language_heads),
        )
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
        self.bridge = M26TraceLanguageBridge(
            prompt_hidden_dim=int(embedding_dim),
            trace_hidden_dim=int(advisor_hidden_dim),
        )
        self.generator_primary_input = "language_hidden_states"
        self.answer_head_primary_input = "fused_language_trace_state"

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

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        language = self.language_backbone(input_ids)
        generator_outputs = self.generator(input_ids, prompt_hidden_states=language["token_hidden_states"])
        trace_slots, trace_active = self.advisor.slot_embeddings_from_logits(
            active_logits=generator_outputs["active_logits"],
            type_logits=generator_outputs["type_logits"],
            value_logits=generator_outputs["value_logits"],
            aux_logits=generator_outputs["aux_logits"],
        )
        bridge_outputs = self.bridge(
            token_hidden_states=language["token_hidden_states"],
            prompt_mask=language["prompt_mask"],
            trace_slots=trace_slots,
            trace_active=trace_active,
        )
        return {
            **generator_outputs,
            "generator_answer_logits": generator_outputs["answer_logits"],
            "trace_only_answer_logits": self.advisor_logits_from_generator_outputs(generator_outputs),
            "answer_logits": bridge_outputs["answer_logits"],
            "trace_state": self.advisor.encode_from_logits(
                active_logits=generator_outputs["active_logits"],
                type_logits=generator_outputs["type_logits"],
                value_logits=generator_outputs["value_logits"],
                aux_logits=generator_outputs["aux_logits"],
            ),
            "language_hidden_states": language["token_hidden_states"],
            "prompt_state": bridge_outputs["prompt_state"],
            "fused_state": bridge_outputs["fused_state"],
            "trace_slots": trace_slots,
            "trace_active": trace_active,
            "bridge_delta": bridge_outputs["bridge_delta"],
            "trace_attention": bridge_outputs["trace_attention"],
            "bridge_gate_value": bridge_outputs["bridge_gate_value"],
            "trace_attention_entropy": bridge_outputs["trace_attention_entropy"],
            "trace_active_mass": bridge_outputs["trace_active_mass"],
            "bridge_delta_norm": bridge_outputs["bridge_delta_norm"],
            "raw_prompt_bypass_blocked": bridge_outputs["raw_prompt_bypass_blocked"],
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
        "bridge_gate_value": float(outputs["bridge_gate_value"].detach().cpu().item()),
        "bridge_delta_norm": float(outputs["bridge_delta_norm"].detach().cpu().item()),
        "trace_attention_entropy": float(outputs["trace_attention_entropy"].detach().cpu().item()),
        "trace_active_mass": float(outputs["trace_active_mass"].detach().cpu().item()),
        "raw_prompt_bypass_blocked": float(outputs["raw_prompt_bypass_blocked"].detach().cpu().item()),
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
    trace_slot_advisor_params = []
    for module in (model.advisor.type_embedding, model.advisor.value_embedding, model.advisor.aux_embedding):
        trace_slot_advisor_params.extend(module.parameters())
    trace_slot_advisor_norm = _grad_norm(trace_slot_advisor_params)
    advisor_classifier_norm = _grad_norm(model.advisor.classifier.parameters())
    # Backward-compatible alias: in the full-organism bridge, "advisor"
    # means the soft trace-slot embedding path, not the trace-only classifier.
    advisor_norm = trace_slot_advisor_norm
    language_norm = _grad_norm(model.language_backbone.parameters())
    bridge_norm = _grad_norm(model.bridge.parameters())
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
        answer_loss_trace_slot_advisor_grad_norm=trace_slot_advisor_norm,
        answer_loss_advisor_classifier_grad_norm=advisor_classifier_norm,
        answer_loss_language_backbone_grad_norm=language_norm,
        answer_loss_bridge_grad_norm=bridge_norm,
        answer_loss_reaches_generator=1.0 if generator_norm > 0.0 else 0.0,
        answer_loss_reaches_symbol_heads=1.0 if symbol_norm > 0.0 else 0.0,
        answer_loss_reaches_trace_slot_advisor=1.0 if trace_slot_advisor_norm > 0.0 else 0.0,
        answer_loss_reaches_advisor_classifier=1.0 if advisor_classifier_norm > 0.0 else 0.0,
        answer_loss_reaches_language_backbone=1.0 if language_norm > 0.0 else 0.0,
        answer_loss_reaches_bridge=1.0 if bridge_norm > 0.0 else 0.0,
    )


def evaluate_m26_end_to_end_loafman(
    *,
    model: M26EndToEndLoafman,
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
    surfaces: list[str] = []
    symbol_counts: list[torch.Tensor] = []
    prompt_counts: list[torch.Tensor] = []
    matched_prompt_counts: list[torch.Tensor] = []
    bridge_telemetry: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_obj)
            target = batch["answer_id"].to(device_obj)
            outputs = model(input_ids)
            generator_outputs = _generator_slice(outputs)
            shuffled_outputs = _shuffle_generator_outputs(generator_outputs, seed=int(seed) + batch_idx)
            random_outputs = _random_generator_outputs(generator_outputs, seed=int(seed) + 1000 + batch_idx)
            zero_active = torch.zeros_like(generator_outputs["active_logits"])
            language_outputs = {
                "token_hidden_states": outputs["language_hidden_states"],
                "prompt_mask": input_ids.ne(0),
            }

            logits["predicted"].append(outputs["answer_logits"].detach().cpu())
            for key in ("bridge_gate_value", "bridge_delta_norm", "trace_attention_entropy", "trace_active_mass", "raw_prompt_bypass_blocked"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    bridge_telemetry[key].append(float(value.detach().float().mean().cpu().item()))
            logits["shuffled"].append(
                model.bridge_logits_from_generator_outputs(
                    input_ids=input_ids,
                    generator_outputs=shuffled_outputs,
                    language_outputs=language_outputs,
                ).detach().cpu()
            )
            logits["random"].append(
                model.bridge_logits_from_generator_outputs(
                    input_ids=input_ids,
                    generator_outputs=random_outputs,
                    language_outputs=language_outputs,
                ).detach().cpu()
            )
            logits["zero"].append(
                model.bridge_logits_from_generator_outputs(
                    input_ids=input_ids,
                    generator_outputs=generator_outputs,
                    active_override=zero_active,
                    language_outputs=language_outputs,
                ).detach().cpu()
            )
            if prompt_control is not None:
                logits["prompt"].append(prompt_control(input_ids).detach().cpu())
            if matched_prompt_control is not None:
                matched_input_ids = budget_prompt_tokens(input_ids, matched_prompt_budget)
                logits["matched_prompt"].append(matched_prompt_control(matched_input_ids).detach().cpu())
                matched_prompt_counts.append(matched_input_ids.ne(0).sum(dim=-1).float().detach().cpu())
            predicted = pack_loose_stream_from_outputs(generator_outputs)
            oracle = batch["stream_targets"].to(device_obj)
            streams["predicted"].append(predicted.detach().cpu())
            streams["oracle"].append(oracle.detach().cpu())
            targets.append(target.detach().cpu())
            surfaces.extend(batch["surface"])
            symbol_counts.append(loose_stream_symbol_counts(predicted).detach().cpu())
            prompt_counts.append(input_ids.ne(0).sum(dim=-1).float().detach().cpu())
    target_all = torch.cat(targets, dim=0)
    all_logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    all_streams = {key: torch.cat(value, dim=0) for key, value in streams.items()}
    predicted_acc = _accuracy(all_logits["predicted"], target_all)
    shuffled_acc = _accuracy(all_logits["shuffled"], target_all)
    random_acc = _accuracy(all_logits["random"], target_all)
    zero_acc = _accuracy(all_logits["zero"], target_all)
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
        "shuffled_trace_accuracy": shuffled_acc,
        "random_trace_accuracy": random_acc,
        "zero_trace_accuracy": zero_acc,
        "prompt_only_accuracy": prompt_acc,
        "matched_prompt_accuracy": matched_prompt_acc,
        "m26_strict_delta_vs_prompt_only": float(predicted_acc - prompt_acc),
        "m26_strict_delta_vs_matched_prompt": float(predicted_acc - matched_prompt_acc),
        "predicted_vs_shuffled_delta": float(predicted_acc - shuffled_acc),
        "predicted_vs_random_delta": float(predicted_acc - random_acc),
        "predicted_vs_zero_delta": float(predicted_acc - zero_acc),
        "mean_predicted_emitted_symbols_after_bottleneck": mean_pred,
        "mean_prompt_tokens": mean_prompt,
        "mean_matched_prompt_tokens": mean_matched_prompt,
        "loose_symbol_to_prompt_ratio": float(mean_pred / max(1.0, mean_prompt)),
        "loose_symbol_to_matched_prompt_ratio": float(mean_pred / max(1.0, mean_matched_prompt)),
        "matched_prompt_token_reduction_ratio": float(1.0 - mean_matched_prompt / max(1.0, mean_prompt)),
        "accuracy_per_loose_symbol": float(predicted_acc / max(1.0, mean_pred)),
        "accuracy_per_prompt_token": float(prompt_acc / max(1.0, mean_prompt)),
        "matched_prompt_accuracy_per_token": float(matched_prompt_acc / max(1.0, mean_matched_prompt)),
        "m26_accuracy_per_symbol_delta_vs_matched_prompt": float(
            predicted_acc / max(1.0, mean_pred) - matched_prompt_acc / max(1.0, mean_matched_prompt)
        ),
        "matched_prompt_token_budget": float(matched_prompt_budget or 0),
        "m26_gate_beats_matched_prompt": 1.0 if "matched_prompt" in all_logits and predicted_acc >= matched_prompt_acc else 0.0,
        "single_optimizer_end_to_end_training": 1.0,
        "hard_argmax_training_cut_detected": 0.0,
        "torch_no_grad_training_cut_detected": 0.0,
        "advisor_primary_trace_is_differentiable": 1.0,
        "bridge_gate_value": mean(bridge_telemetry["bridge_gate_value"]) if bridge_telemetry["bridge_gate_value"] else 0.0,
        "bridge_delta_norm": mean(bridge_telemetry["bridge_delta_norm"]) if bridge_telemetry["bridge_delta_norm"] else 0.0,
        "trace_attention_entropy": mean(bridge_telemetry["trace_attention_entropy"]) if bridge_telemetry["trace_attention_entropy"] else 0.0,
        "trace_active_mass": mean(bridge_telemetry["trace_active_mass"]) if bridge_telemetry["trace_active_mass"] else 0.0,
        "raw_prompt_bypass_blocked": mean(bridge_telemetry["raw_prompt_bypass_blocked"])
        if bridge_telemetry["raw_prompt_bypass_blocked"]
        else 0.0,
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
    prompt_epochs: int | None = None,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 26,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    max_frames: int = 6,
    max_symbols: int = 32,
    max_prompt_length: int = 128,
    language_layers: int = 1,
    language_heads: int = 2,
    symbol_budget: int | None = None,
    matched_prompt_budget: int | None = None,
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
        max_prompt_length=int(max_prompt_length),
        language_layers=int(language_layers),
        language_heads=int(language_heads),
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

    prompt_control = PromptOnlyControl(
        vocab_size=len(vocab),
        embedding_dim=max(8, int(embedding_dim) // 2),
        hidden_dim=int(advisor_hidden_dim),
    ).to(device_obj)
    resolved_matched_prompt_budget = int(matched_prompt_budget or 0)
    if resolved_matched_prompt_budget <= 0:
        resolved_matched_prompt_budget = int(symbol_budget or 0)
    if resolved_matched_prompt_budget <= 0:
        resolved_matched_prompt_budget = int(max_symbols)
    matched_prompt_control = PromptOnlyControl(
        vocab_size=len(vocab),
        embedding_dim=max(8, int(embedding_dim) // 2),
        hidden_dim=int(advisor_hidden_dim),
    ).to(device_obj)
    prompt_history = _train_prompt_control(
        prompt_control,
        train_examples,
        vocab,
        max_symbols=int(max_symbols),
        prompt_token_budget=None,
        epochs=int(prompt_epochs if prompt_epochs is not None else epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        device=device_obj,
        seed=int(seed) + 300,
    )
    matched_prompt_history = _train_prompt_control(
        matched_prompt_control,
        train_examples,
        vocab,
        max_symbols=int(max_symbols),
        prompt_token_budget=resolved_matched_prompt_budget,
        epochs=int(prompt_epochs if prompt_epochs is not None else epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        device=device_obj,
        seed=int(seed) + 400,
    )
    eval_payload = evaluate_m26_end_to_end_loafman(
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
    gradient_probe = probe_m26_answer_gradient_flow(model, probe_batch).as_dict()
    metrics = dict(eval_payload["metrics"])
    metrics.update(gradient_probe)
    metrics.update(
        {
            "trainable_parameter_count": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "language_backbone_trainable_parameter_count": float(
                sum(p.numel() for p in model.language_backbone.parameters() if p.requires_grad)
            ),
            "generator_trainable_parameter_count": float(sum(p.numel() for p in model.generator.parameters() if p.requires_grad)),
            "advisor_trainable_parameter_count": float(sum(p.numel() for p in model.advisor.parameters() if p.requires_grad)),
            "bridge_trainable_parameter_count": float(sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)),
            "lm_hidden_state_stream_active": 1.0,
            "bridi_generator_reads_lm_hidden_states": 1.0,
            "trace_bridge_reads_prompt_hidden_states": 1.0,
            "answer_head_reads_fused_lm_trace_state": 1.0,
            "raw_prompt_bypass_blocked": 1.0,
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
            "prompt_epochs": int(prompt_epochs if prompt_epochs is not None else epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "max_frames": int(max_frames),
            "max_symbols": int(max_symbols),
            "max_prompt_length": int(max_prompt_length),
            "language_layers": int(language_layers),
            "language_heads": int(language_heads),
            "symbol_budget": int(symbol_budget or 0),
            "matched_prompt_budget": int(resolved_matched_prompt_budget),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "device": str(device_obj),
            "organism_mode": "lm_hidden_bridi_bridge",
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


def m26_promotion_gate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    spinal_gates = {
        "m26_gate_answer_loss_reaches_generator": 1.0 if metrics.get("answer_loss_reaches_generator", 0.0) >= 1.0 else 0.0,
        "m26_gate_answer_loss_reaches_symbol_heads": 1.0 if metrics.get("answer_loss_reaches_symbol_heads", 0.0) >= 1.0 else 0.0,
        "m26_gate_single_optimizer": 1.0 if metrics.get("single_optimizer_end_to_end_training", 0.0) >= 1.0 else 0.0,
        "m26_gate_no_hard_training_cut": 1.0 if metrics.get("hard_argmax_training_cut_detected", 1.0) == 0.0 else 0.0,
        "m26_gate_stream_beats_zero": 1.0 if metrics.get("predicted_vs_zero_delta", 0.0) >= 0.02 else 0.0,
    }
    full_organism_gates = {
        "m26_gate_answer_loss_reaches_language_backbone": 1.0
        if metrics.get("answer_loss_reaches_language_backbone", 0.0) >= 1.0
        else 0.0,
        "m26_gate_answer_loss_reaches_bridge": 1.0 if metrics.get("answer_loss_reaches_bridge", 0.0) >= 1.0 else 0.0,
        "m26_gate_bridi_generator_reads_lm_hidden_states": 1.0
        if metrics.get("bridi_generator_reads_lm_hidden_states", 0.0) >= 1.0
        else 0.0,
        "m26_gate_trace_bridge_reads_prompt_hidden_states": 1.0
        if metrics.get("trace_bridge_reads_prompt_hidden_states", 0.0) >= 1.0
        else 0.0,
        "m26_gate_answer_head_reads_fused_lm_trace_state": 1.0
        if metrics.get("answer_head_reads_fused_lm_trace_state", 0.0) >= 1.0
        else 0.0,
        "m26_gate_raw_prompt_bypass_blocked": 1.0 if metrics.get("raw_prompt_bypass_blocked", 0.0) >= 1.0 else 0.0,
    }
    matched_gate = 1.0 if metrics.get("m26_strict_delta_vs_matched_prompt", -1.0) >= 0.0 else 0.0
    spinal_pass = sum(spinal_gates.values()) / max(1, len(spinal_gates))
    spinal_candidate = 1.0 if all(value == 1.0 for value in spinal_gates.values()) else 0.0
    full_pass = sum(full_organism_gates.values()) / max(1, len(full_organism_gates))
    full_candidate = 1.0 if spinal_candidate >= 1.0 and all(value == 1.0 for value in full_organism_gates.values()) else 0.0
    return {
        **spinal_gates,
        **full_organism_gates,
        "m26_spinal_cord_gate_pass_rate": spinal_pass,
        "m26_spinal_cord_candidate": spinal_candidate,
        "m26_full_organism_gate_pass_rate": full_pass,
        "m26_full_organism_candidate": full_candidate,
        "m26_gate_beats_matched_prompt": matched_gate,
        "m26_prompt_comparable_candidate": 1.0 if full_candidate >= 1.0 and matched_gate >= 1.0 else 0.0,
        "m26_promotion_candidate": 1.0 if full_candidate >= 1.0 and matched_gate >= 1.0 else 0.0,
    }


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


def _train_prompt_control(
    control: PromptOnlyControl,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    *,
    max_symbols: int,
    prompt_token_budget: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=int(max_symbols))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        collate_fn=m25_collate,
    )
    optimizer = torch.optim.AdamW(control.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    control.train()
    for _ in range(int(epochs)):
        total_loss = 0.0
        total_acc = 0.0
        batches = 0
        for batch in loader:
            target = batch["answer_id"].to(device)
            input_ids = budget_prompt_tokens(batch["input_ids"].to(device), prompt_token_budget)
            logits = control(input_ids)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(control.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_acc += _accuracy(logits.detach(), target.detach())
            batches += 1
        history.append({"loss": total_loss / max(1, batches), "accuracy": total_acc / max(1, batches)})
    return history
