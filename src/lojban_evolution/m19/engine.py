from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, List
from contextlib import contextmanager

from .typed_physics import (
    FAMILY_TO_ID,
    apply_radius_bands,
    family_histogram_from_ids,
    hyperbolic_family_metrics,
    logmap0,
    mean_entropy_from_logits,
    parse_typed_slot_layout,
    slot_family_counts,
)


BRIDGE_CHANNEL_MODES = {
    "full",
    "zero_all",
    "gismu_only",
    "cmavo_only",
    "judri_only",
    "control_only",
    "op_only",
    "pointer_only",
    "no_gismu",
    "no_cmavo",
    "no_judri",
    "no_control",
}


def _off_diagonal_cosines(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D tensor for pairwise cosine stats, got shape {tuple(vectors.shape)}")
    count = int(vectors.shape[0])
    if count < 2:
        return torch.empty(0, device=vectors.device, dtype=vectors.dtype)
    normalized = F.normalize(vectors, dim=-1)
    cosine = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(count, device=vectors.device, dtype=torch.bool)
    return cosine.masked_select(mask)


def pairwise_cosine_stats(vectors: torch.Tensor) -> Dict[str, float]:
    off_diag = _off_diagonal_cosines(vectors)
    if off_diag.numel() == 0:
        return {
            "pairwise_cosine_mean": 0.0,
            "pairwise_cosine_max": 0.0,
            "pairwise_cosine_min": 0.0,
            "pairwise_cosine_std": 0.0,
            "anisotropy": 0.0,
        }
    return {
        "pairwise_cosine_mean": float(off_diag.mean().item()),
        "pairwise_cosine_max": float(off_diag.max().item()),
        "pairwise_cosine_min": float(off_diag.min().item()),
        "pairwise_cosine_std": float(off_diag.std(unbiased=False).item()),
        "anisotropy": float(off_diag.abs().mean().item()),
    }


def batched_pairwise_cosine_stats(trace: torch.Tensor, lengths: torch.Tensor | None = None) -> Dict[str, float]:
    if trace.ndim != 3:
        raise ValueError(f"Expected 3D trace tensor, got shape {tuple(trace.shape)}")
    if lengths is None:
        lengths = torch.full((trace.shape[0],), trace.shape[1], device=trace.device, dtype=torch.long)
    values: list[torch.Tensor] = []
    for batch_idx in range(trace.shape[0]):
        active = int(lengths[batch_idx].item())
        if active < 2:
            continue
        off_diag = _off_diagonal_cosines(trace[batch_idx, :active, :])
        if off_diag.numel() > 0:
            values.append(off_diag)
    if not values:
        return {
            "pairwise_cosine_mean": 0.0,
            "pairwise_cosine_max": 0.0,
            "pairwise_cosine_min": 0.0,
            "pairwise_cosine_std": 0.0,
            "anisotropy": 0.0,
        }
    merged = torch.cat(values, dim=0)
    return {
        "pairwise_cosine_mean": float(merged.mean().item()),
        "pairwise_cosine_max": float(merged.max().item()),
        "pairwise_cosine_min": float(merged.min().item()),
        "pairwise_cosine_std": float(merged.std(unbiased=False).item()),
        "anisotropy": float(merged.abs().mean().item()),
    }


def operator_distribution_stats(op_logits: torch.Tensor, lengths: torch.Tensor | None = None) -> Dict[str, float]:
    if op_logits.ndim != 3:
        raise ValueError(f"Expected 3D operator logits tensor, got shape {tuple(op_logits.shape)}")
    probs = F.softmax(op_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    top1 = probs.max(dim=-1).values
    if lengths is None:
        valid = torch.ones_like(entropy, dtype=torch.bool)
    else:
        valid = torch.arange(op_logits.shape[1], device=op_logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    entropy_values = entropy.masked_select(valid)
    top1_values = top1.masked_select(valid)
    if entropy_values.numel() == 0:
        return {
            "operator_entropy_mean": 0.0,
            "operator_entropy_ratio_mean": 0.0,
            "operator_top1_share_mean": 0.0,
        }
    max_entropy = math.log(op_logits.shape[-1])
    return {
        "operator_entropy_mean": float(entropy_values.mean().item()),
        "operator_entropy_ratio_mean": float((entropy_values.mean() / max_entropy).item()),
        "operator_top1_share_mean": float(top1_values.mean().item()),
    }


def compute_query_repulsion_loss(
    query_vectors: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    off_diag = _off_diagonal_cosines(query_vectors)
    if off_diag.numel() == 0:
        return torch.zeros((), device=query_vectors.device, dtype=query_vectors.dtype)
    penalty = torch.relu(off_diag.abs() - float(margin))
    return penalty.mean()


def dictionary_health_metrics(
    query_vectors: torch.Tensor,
    query_state: torch.Tensor,
    trace: torch.Tensor,
    delta: torch.Tensor,
    op_logits: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for prefix, stats in (
        ("query_embed", pairwise_cosine_stats(query_vectors)),
        ("query_state", batched_pairwise_cosine_stats(query_state)),
        ("scratch_trace", batched_pairwise_cosine_stats(trace, lengths=lengths)),
        ("scratch_delta", batched_pairwise_cosine_stats(delta, lengths=lengths)),
        ("operator", operator_distribution_stats(op_logits, lengths=lengths)),
    ):
        for key, value in stats.items():
            metrics[f"{prefix}_{key}"] = float(value)
    return metrics


class M19MagneticCollar(nn.Module):
    def __init__(self, bottleneck_dim: int, max_positions: int = 64):
        super().__init__()
        self.subject_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.relation_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.object_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.spatial_embeddings = nn.Parameter(torch.randn(max_positions, bottleneck_dim) * 0.02)

    def forward(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if trace.shape[1] < 3:
            return torch.tensor(0.0, device=trace.device, dtype=trace.dtype)
        active_positions = self.spatial_embeddings[: trace.shape[1], :].to(device=trace.device, dtype=trace.dtype)
        trace = trace + active_positions.unsqueeze(0)
        v_sub = self.subject_proj(trace[:, 0:-2, :])
        v_rel = self.relation_proj(trace[:, 1:-1, :])
        v_obj = self.object_proj(trace[:, 2:, :])
        residual = (v_sub + v_rel - v_obj) ** 2
        if lengths is None:
            return residual.mean()
        lengths = lengths.to(device=trace.device)
        triad_positions = torch.arange(trace.shape[1] - 2, device=trace.device).unsqueeze(0)
        valid = triad_positions < (lengths.unsqueeze(1) - 2).clamp_min(0)
        if not bool(valid.any()):
            return torch.tensor(0.0, device=trace.device, dtype=trace.dtype)
        weighted = residual.mean(dim=-1)
        return (weighted * valid.to(dtype=weighted.dtype)).sum() / valid.to(dtype=weighted.dtype).sum().clamp_min(1.0)

class M19SymbioteBridge(nn.Module):
    """
    M19.3 Capacity-Agnostic Bridge:
    Supports variable query counts, bottleneck dimensions, and scratchpad lengths.
    """
    def __init__(
        self,
        hidden_size: int = 896,
        bottleneck_dim: int = 64,
        scratchpad_len: int = 8,
        num_queries: int = 8,
        max_latent_steps: int | None = None,
        typed_slot_layout: list[str] | tuple[str, ...] | str | None = None,
        geometry_mode: str = "euclidean",
        arity_router_mode: str = "soft",
        gumbel_hard: bool = False,
        poincare_curvature: float = 1.0,
        radius_bands: dict[str, dict[str, float]] | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.scratchpad_len = scratchpad_len
        self.num_queries = num_queries
        self.max_latent_steps = max(int(max_latent_steps or scratchpad_len), int(scratchpad_len))
        self.typed_slot_layout = parse_typed_slot_layout(typed_slot_layout) if typed_slot_layout else []
        self.typed_physics_enabled = bool(self.typed_slot_layout)
        self.geometry_mode = str(geometry_mode).strip().lower() or "euclidean"
        self.arity_router_mode = str(arity_router_mode).strip().lower() or "soft"
        self.gumbel_hard = bool(gumbel_hard)
        self.poincare_curvature = max(float(poincare_curvature), 1e-5)
        self.radius_bands = radius_bands or {
            "gismu": {"min": 0.05, "max": 0.30},
            "cmavo": {"min": 0.22, "max": 0.55},
            "judri": {"min": 0.48, "max": 0.82},
            "control": {"min": 0.08, "max": 0.40},
        }
        self.slot_family_counts = slot_family_counts(self.typed_slot_layout) if self.typed_physics_enabled else {}
        self.family_indices = {
            family: [idx for idx, name in enumerate(self.typed_slot_layout) if name == family]
            for family in FAMILY_TO_ID
        } if self.typed_physics_enabled else {}
        self.register_buffer(
            "slot_family_ids",
            torch.tensor([FAMILY_TO_ID[name] for name in self.typed_slot_layout], dtype=torch.long),
            persistent=False,
        )

        # 1. Latent Queries (Extraction Heads)
        self.query_embeds = nn.Parameter(torch.randn(num_queries, bottleneck_dim) * 0.02)
        if self.typed_physics_enabled:
            self.family_query_banks = nn.ParameterDict()
            for family, count in self.slot_family_counts.items():
                if count > 0:
                    self.family_query_banks[family] = nn.Parameter(torch.randn(count, bottleneck_dim) * 0.02)
            self.typed_slot_position = nn.Parameter(torch.randn(len(self.typed_slot_layout), bottleneck_dim) * 0.02)
            self.family_head = nn.Linear(bottleneck_dim, len(FAMILY_TO_ID))
            self.arity_head = nn.Linear(bottleneck_dim, 3) if self.family_indices.get("gismu") else None
        else:
            self.family_query_banks = None
            self.typed_slot_position = None
            self.family_head = None
            self.arity_head = None
        
        # 2. Cross-Attention Bottleneck
        self.compress = nn.Linear(hidden_size, bottleneck_dim)
        self.cross_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=max(1, bottleneck_dim // 32), batch_first=True)
        
        # 3. Output Mapping Into A Dynamic Positional Reservoir
        self.output_map = nn.Linear(num_queries, self.max_latent_steps)
        
        # 4. Projections
        self.expand = nn.Linear(bottleneck_dim, hidden_size)
        self.collar = M19MagneticCollar(bottleneck_dim, max_positions=self.max_latent_steps)
        self.op_head = nn.Linear(bottleneck_dim, 2000)
        self.register_buffer("halt_centroid", torch.zeros(bottleneck_dim), persistent=True)
        self.register_buffer("halt_centroid_samples", torch.tensor(0.0), persistent=True)

    def query_dictionary(self) -> torch.Tensor:
        if not self.typed_physics_enabled or self.family_query_banks is None:
            return self.query_embeds
        counters = {family: 0 for family in FAMILY_TO_ID}
        slots: list[torch.Tensor] = []
        for idx, family in enumerate(self.typed_slot_layout):
            family_bank = self.family_query_banks[family]
            family_offset = counters[family]
            slots.append(family_bank[family_offset] + self.typed_slot_position[idx])
            counters[family] += 1
        return torch.stack(slots, dim=0)

    def _apply_typed_routing(
        self,
        query_state: torch.Tensor,
        gumbel_temperature: float,
        arity_override: int | None = None,
        disable_arity_mask: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.typed_physics_enabled or self.family_head is None:
            return query_state, {
                "slot_family_logits": None,
                "slot_family_entropy": 0.0,
                "judri_mask": None,
                "arity_logits": None,
                "arity_distribution": None,
                "active_predicate_slot": None,
                "active_arity_budget": None,
                "arity_override": None,
                "arity_mask_disabled": False,
                "masked_pointer_zero_rate": None,
                "hyperbolic_metrics": {},
            }

        routed_state = query_state
        telemetry: dict[str, Any] = {
            "slot_family_logits": self.family_head(query_state),
            "slot_family_entropy": 0.0,
            "judri_mask": None,
            "arity_logits": None,
            "arity_distribution": None,
            "active_predicate_slot": None,
            "active_arity_budget": None,
            "arity_override": None,
            "arity_mask_disabled": False,
            "masked_pointer_zero_rate": None,
            "hyperbolic_metrics": {},
        }
        telemetry["slot_family_entropy"] = float(mean_entropy_from_logits(telemetry["slot_family_logits"]).detach().item())

        if self.geometry_mode == "hyperbolic":
            routed_state, clip_mask = apply_radius_bands(
                query_state,
                self.typed_slot_layout,
                self.radius_bands,
                self.poincare_curvature,
            )
            telemetry["hyperbolic_metrics"] = hyperbolic_family_metrics(
                routed_state.detach(),
                self.typed_slot_layout,
                self.poincare_curvature,
                self.radius_bands,
            )
            telemetry["hyperbolic_metrics"]["hyperbolic_projection_clip_rate"] = float(clip_mask.mean().detach().item())

        gismu_indices = self.family_indices.get("gismu", [])
        judri_indices = self.family_indices.get("judri", [])
        if gismu_indices and judri_indices and self.arity_head is not None:
            family_logits = telemetry["slot_family_logits"]
            family_probs = F.softmax(family_logits[:, gismu_indices, :], dim=-1)
            gismu_scores = family_probs[..., FAMILY_TO_ID["gismu"]]
            active_local = gismu_scores.argmax(dim=1)
            batch_ids = torch.arange(routed_state.shape[0], device=routed_state.device)
            active_state = routed_state[:, gismu_indices, :][batch_ids, active_local, :]
            arity_logits = self.arity_head(active_state)
            if self.arity_router_mode == "gumbel_hard":
                arity_distribution = F.gumbel_softmax(
                    arity_logits,
                    tau=max(float(gumbel_temperature), 1e-4),
                    hard=bool(self.gumbel_hard),
                    dim=-1,
                )
            else:
                arity_distribution = F.softmax(arity_logits, dim=-1)
            pointer_budget = (arity_distribution * torch.tensor([1.0, 2.0, 3.0], device=routed_state.device, dtype=routed_state.dtype)).sum(dim=-1)
            if self.arity_router_mode == "gumbel_hard":
                active_budget = pointer_budget.round().to(dtype=torch.long)
            else:
                active_budget = pointer_budget.round().clamp_min(1.0).to(dtype=torch.long)
            if arity_override is not None:
                override_budget = max(1, min(len(judri_indices), int(arity_override)))
                active_budget = torch.full_like(active_budget, int(override_budget))
                pointer_budget = torch.full_like(pointer_budget, float(override_budget))
            if disable_arity_mask:
                active_budget = torch.full_like(active_budget, int(len(judri_indices)))
                pointer_budget = torch.full_like(pointer_budget, float(len(judri_indices)))
            hard_mask = torch.zeros(
                routed_state.shape[0],
                len(judri_indices),
                device=routed_state.device,
                dtype=routed_state.dtype,
            )
            for batch_idx in range(routed_state.shape[0]):
                budget = min(len(judri_indices), int(active_budget[batch_idx].item()))
                if budget > 0:
                    hard_mask[batch_idx, :budget] = 1.0
            if disable_arity_mask:
                hard_mask = torch.ones_like(hard_mask)
            positions = torch.arange(1, len(judri_indices) + 1, device=routed_state.device, dtype=routed_state.dtype).unsqueeze(0)
            soft_mask = torch.sigmoid((pointer_budget.unsqueeze(-1) - positions + 0.5) * 8.0)
            if disable_arity_mask:
                soft_mask = torch.ones_like(soft_mask)
            if self.arity_router_mode == "gumbel_hard":
                judri_mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                judri_mask = soft_mask
            routed_state = routed_state.clone()
            routed_state[:, judri_indices, :] = routed_state[:, judri_indices, :] * judri_mask.unsqueeze(-1)
            telemetry["judri_mask"] = hard_mask
            telemetry["arity_logits"] = arity_logits
            telemetry["arity_distribution"] = arity_distribution
            telemetry["active_predicate_slot"] = active_local
            telemetry["active_arity_budget"] = active_budget
            telemetry["arity_override"] = arity_override
            telemetry["arity_mask_disabled"] = bool(disable_arity_mask)
            if len(judri_indices) > 0:
                forbidden_mask = hard_mask < 0.5
                if bool(forbidden_mask.any().detach().item()):
                    forbidden_values = routed_state[:, judri_indices, :][forbidden_mask]
                    forbidden_zero = forbidden_values.detach().float().norm(dim=-1) < 1e-8
                    telemetry["masked_pointer_zero_rate"] = float(forbidden_zero.float().mean().item())
                else:
                    telemetry["masked_pointer_zero_rate"] = None

        return routed_state, telemetry

    def _family_energy(self, query_state: torch.Tensor) -> dict[str, float]:
        if not self.typed_physics_enabled:
            return {}
        energy: dict[str, float] = {}
        norms = query_state.detach().float().norm(dim=-1)
        for family, indices in self.family_indices.items():
            if indices:
                energy[family] = float(norms[:, indices].mean().item())
        return energy

    def _apply_bridge_channel_mask(
        self,
        query_state: torch.Tensor,
        bridge_channel_mode: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mode = str(bridge_channel_mode or "full").strip().lower()
        aliases = {
            "predicate_only": "gismu_only",
            "operator_only": "gismu_only",
            "op_only": "gismu_only",
            "pointer_only": "judri_only",
            "argument_only": "judri_only",
            "arg_only": "judri_only",
            "none": "zero_all",
            "zero": "zero_all",
        }
        mode = aliases.get(mode, mode)
        if mode not in BRIDGE_CHANNEL_MODES:
            raise ValueError(f"unsupported bridge_channel_mode: {bridge_channel_mode}")

        before_energy = self._family_energy(query_state)
        if not self.typed_physics_enabled or mode == "full":
            return query_state, {
                "bridge_channel_mode": mode,
                "bridge_channel_retained_slot_fraction": 1.0,
                "bridge_channel_family_energy_before": before_energy,
                "bridge_channel_family_energy_after": before_energy,
            }

        if mode == "zero_all":
            keep_families: set[str] = set()
            drop_families: set[str] = set(self.family_indices.keys())
        elif mode.endswith("_only"):
            keep_families = {mode.removesuffix("_only")}
            drop_families = set()
        elif mode.startswith("no_"):
            keep_families = set(self.family_indices.keys())
            drop_families = {mode.removeprefix("no_")}
        else:
            keep_families = set(self.family_indices.keys())
            drop_families = set()

        slot_mask = torch.ones(len(self.typed_slot_layout), device=query_state.device, dtype=query_state.dtype)
        for idx, family in enumerate(self.typed_slot_layout):
            if keep_families and family not in keep_families:
                slot_mask[idx] = 0.0
            if family in drop_families:
                slot_mask[idx] = 0.0

        masked = query_state * slot_mask.view(1, -1, 1)
        return masked, {
            "bridge_channel_mode": mode,
            "bridge_channel_retained_slot_fraction": float(slot_mask.detach().float().mean().item()) if slot_mask.numel() else 1.0,
            "bridge_channel_family_energy_before": before_energy,
            "bridge_channel_family_energy_after": self._family_energy(masked),
        }

    def forward(
        self,
        h_tap: torch.Tensor,
        active_steps: int | None = None,
        lengths: torch.Tensor | None = None,
        gumbel_temperature: float = 1.0,
        arity_override: int | None = None,
        disable_arity_mask: bool = False,
        bridge_channel_mode: str = "full",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b = h_tap.shape[0]
        target_steps = int(active_steps or self.scratchpad_len)
        target_steps = max(1, min(target_steps, self.max_latent_steps))
        h_bottleneck = self.compress(h_tap)
        
        queries = self.query_dictionary().unsqueeze(0).expand(b, -1, -1)
        h_query, _ = self.cross_attn(queries, h_bottleneck, h_bottleneck)
        h_query, typed_telemetry = self._apply_typed_routing(
            h_query,
            gumbel_temperature=gumbel_temperature,
            arity_override=arity_override,
            disable_arity_mask=disable_arity_mask,
        )
        h_query, channel_telemetry = self._apply_bridge_channel_mask(h_query, bridge_channel_mode)
        if self.geometry_mode == "hyperbolic" and self.typed_physics_enabled:
            h_reservoir = self.output_map(logmap0(h_query, self.poincare_curvature).transpose(1, 2)).transpose(1, 2)
            h_scratch = logmap0(
                apply_radius_bands(
                    torch.tanh(h_reservoir),
                    [self.typed_slot_layout[min(idx, len(self.typed_slot_layout) - 1)] for idx in range(h_reservoir.shape[1])]
                    if self.typed_slot_layout
                    else ["control"] * h_reservoir.shape[1],
                    self.radius_bands,
                    self.poincare_curvature,
                )[0][:, :target_steps, :],
                self.poincare_curvature,
            )
        else:
            h_reservoir = self.output_map(h_query.transpose(1, 2)).transpose(1, 2)
            h_scratch = h_reservoir[:, :target_steps, :]

        if lengths is None:
            lengths = torch.full((b,), target_steps, device=h_tap.device, dtype=torch.long)
        l_topo = self.collar(h_scratch, lengths=lengths)
        op_logits = self.op_head(h_scratch)
        delta = self.expand(h_scratch)
        telemetry = {
            "active_steps": target_steps,
            "halt_cosine_per_step": self.halt_similarity(h_scratch, lengths=lengths).detach(),
            "query_state": h_query.detach(),
            "trace": h_scratch.detach(),
            "delta": delta.detach(),
            "typed_slot_layout": list(self.typed_slot_layout),
            "slot_family_ids": self.slot_family_ids.detach().cpu().tolist() if self.typed_physics_enabled else [],
            "slot_family_logits": typed_telemetry.get("slot_family_logits").detach() if typed_telemetry.get("slot_family_logits") is not None else None,
            "slot_family_entropy": typed_telemetry.get("slot_family_entropy", 0.0),
            "judri_mask": typed_telemetry.get("judri_mask").detach() if typed_telemetry.get("judri_mask") is not None else None,
            "arity_logits": typed_telemetry.get("arity_logits").detach() if typed_telemetry.get("arity_logits") is not None else None,
            "arity_distribution": typed_telemetry.get("arity_distribution").detach() if typed_telemetry.get("arity_distribution") is not None else None,
            "active_predicate_slot": typed_telemetry.get("active_predicate_slot").detach() if typed_telemetry.get("active_predicate_slot") is not None else None,
            "active_arity_budget": typed_telemetry.get("active_arity_budget").detach() if typed_telemetry.get("active_arity_budget") is not None else None,
            "arity_override": typed_telemetry.get("arity_override"),
            "arity_mask_disabled": bool(typed_telemetry.get("arity_mask_disabled", False)),
            "masked_pointer_zero_rate": typed_telemetry.get("masked_pointer_zero_rate"),
            "geometry_mode": self.geometry_mode,
            "bridge_channel_mode": channel_telemetry.get("bridge_channel_mode", "full"),
            "bridge_channel_retained_slot_fraction": channel_telemetry.get("bridge_channel_retained_slot_fraction", 1.0),
            "bridge_channel_family_energy_before": channel_telemetry.get("bridge_channel_family_energy_before", {}),
            "bridge_channel_family_energy_after": channel_telemetry.get("bridge_channel_family_energy_after", {}),
            "hyperbolic_metrics": typed_telemetry.get("hyperbolic_metrics", {}),
            "dictionary_health": dictionary_health_metrics(
                self.query_dictionary().detach(),
                h_query.detach(),
                h_scratch.detach(),
                delta.detach(),
                op_logits.detach(),
                lengths=lengths.detach() if lengths is not None else None,
            ),
        }
        if self.typed_physics_enabled:
            dictionary_health = telemetry["dictionary_health"]
            masked_pointer_zero_rate = typed_telemetry.get("masked_pointer_zero_rate")
            dictionary_health["masked_pointer_zero_rate"] = (
                float(masked_pointer_zero_rate) if masked_pointer_zero_rate is not None else None
            )
            dictionary_health["family_slot_entropy"] = float(typed_telemetry.get("slot_family_entropy", 0.0))
            for key, value in typed_telemetry.get("hyperbolic_metrics", {}).items():
                dictionary_health[key] = float(value)

        return delta, l_topo, op_logits, telemetry

    def update_halt_centroid(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            if trace.ndim != 3 or trace.shape[1] == 0:
                return
            if lengths is None:
                lengths = torch.full((trace.shape[0],), trace.shape[1], device=trace.device, dtype=torch.long)
            lengths = lengths.to(device=trace.device)
            samples: list[torch.Tensor] = []
            for batch_idx in range(trace.shape[0]):
                final_idx = int(lengths[batch_idx].item()) - 1
                if final_idx < 0 or final_idx >= trace.shape[1]:
                    continue
                samples.append(trace[batch_idx, final_idx, :])
            if not samples:
                return
            batch_mean = torch.stack(samples, dim=0).mean(dim=0).to(device=self.halt_centroid.device, dtype=self.halt_centroid.dtype)
            current = float(self.halt_centroid_samples.item())
            batch_count = float(len(samples))
            if current <= 0:
                self.halt_centroid.copy_(batch_mean)
                self.halt_centroid_samples.fill_(batch_count)
                return
            mixed = ((self.halt_centroid * current) + (batch_mean * batch_count)) / max(current + batch_count, 1.0)
            self.halt_centroid.copy_(mixed)
            self.halt_centroid_samples.fill_(current + batch_count)

    def halt_similarity(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if trace.ndim != 3 or trace.shape[1] == 0:
            return torch.zeros(trace.shape[:2], device=trace.device, dtype=trace.dtype)
        centroid = self.halt_centroid.to(device=trace.device, dtype=trace.dtype)
        if float(self.halt_centroid_samples.item()) <= 0 or float(centroid.norm().item()) == 0.0:
            return torch.zeros(trace.shape[:2], device=trace.device, dtype=trace.dtype)
        sims = F.cosine_similarity(trace, centroid.view(1, 1, -1), dim=-1)
        if lengths is None:
            return sims
        lengths = lengths.to(device=trace.device)
        valid_steps = torch.arange(trace.shape[1], device=trace.device).unsqueeze(0) < lengths.unsqueeze(1)
        return sims * valid_steps.to(dtype=sims.dtype)

@contextmanager
def m19_injection_hook(model, layer_idx: int, scratchpad_mask: torch.Tensor, delta: torch.Tensor):
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        if hidden.shape[1] == scratchpad_mask.shape[1]:
            mask_indices = scratchpad_mask.to(hidden.device)
            for b in range(hidden.shape[0]):
                indices = torch.where(mask_indices[b])[0]
                # Allow partial injection if runway length differs slightly
                injection_len = min(len(indices), delta.shape[1])
                if injection_len > 0:
                    hidden[b, indices[:injection_len], :] += delta[b, :injection_len].to(hidden.dtype)
            
        return (hidden, *rest) if rest is not None else hidden

    layers = model.model.layers if hasattr(model, "model") else model.layers
    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()

def compute_m19_anti_collapse(logits: torch.Tensor, min_entropy: float = 0.85) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    max_ent = math.log(logits.shape[-1])
    entropy_ratio = entropy / max_ent
    return torch.relu(min_entropy - entropy_ratio) * 5.0


def ensure_special_tokens(model, tokenizer, tokens: List[str]) -> Dict[str, int]:
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [token for token in tokens if token not in existing_vocab]
    old_size = len(tokenizer)
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
    if tokens_to_add:
        input_emb = model.get_input_embeddings().weight
        with torch.no_grad():
            mean_in = input_emb[:old_size].mean(dim=0, keepdim=True)
            input_emb[old_size : len(tokenizer)] = mean_in.to(device=input_emb.device, dtype=input_emb.dtype)
            output_emb = model.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight") and output_emb.weight.shape[0] >= len(tokenizer):
                mean_out = output_emb.weight[:old_size].mean(dim=0, keepdim=True)
                output_emb.weight[old_size : len(tokenizer)] = mean_out.to(
                    device=output_emb.weight.device, dtype=output_emb.weight.dtype
                )
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}
