from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


FAMILY_TO_ID: dict[str, int] = {
    "gismu": 0,
    "cmavo": 1,
    "judri": 2,
    "control": 3,
}
ID_TO_FAMILY: dict[int, str] = {value: key for key, value in FAMILY_TO_ID.items()}

DEFAULT_SLOT_LAYOUT_SPEC = "gismu:2,cmavo:2,judri:4"
DEFAULT_POINCARE_EPS = 1e-5


@dataclass(frozen=True)
class TypedTargets:
    has_supervision: bool
    trace_tokens: tuple[str, ...]
    family_ids: tuple[int, ...]
    family_histogram: tuple[float, ...]
    primary_arity: int | None
    pointer_budget: int | None


def load_typed_physics_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("typed physics config must decode to a JSON object")
    return payload


def parse_typed_slot_layout(spec: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if spec is None:
        spec = DEFAULT_SLOT_LAYOUT_SPEC
    if isinstance(spec, (list, tuple)):
        values = [str(item).strip().lower() for item in spec if str(item).strip()]
        _validate_slot_families(values)
        return values
    text = str(spec).strip()
    if not text:
        text = DEFAULT_SLOT_LAYOUT_SPEC
    if "," not in text and ":" not in text and "|" in text:
        values = [part.strip().lower() for part in text.split("|") if part.strip()]
        _validate_slot_families(values)
        return values
    values: list[str] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" in token:
            family, count_text = token.split(":", 1)
            family_name = family.strip().lower()
            count = max(0, int(count_text.strip()))
            values.extend([family_name] * count)
        else:
            values.append(token.lower())
    _validate_slot_families(values)
    return values


def slot_family_ids(slot_layout: list[str]) -> list[int]:
    return [FAMILY_TO_ID[str(name)] for name in slot_layout]


def slot_family_counts(slot_layout: list[str]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILY_TO_ID}
    for family in slot_layout:
        counts[str(family)] = counts.get(str(family), 0) + 1
    return counts


def extract_trace_tokens_from_sample(raw_text: str, mode: str, row: dict[str, Any] | None = None) -> list[str]:
    payload = row or {}
    if isinstance(payload.get("logic_trace"), list):
        return [str(token).strip() for token in payload["logic_trace"] if str(token).strip()]
    if isinstance(payload.get("trace_tokens"), list):
        return [str(token).strip() for token in payload["trace_tokens"] if str(token).strip()]
    text = str(raw_text or "")
    if str(mode).lower() == "crystal":
        match = re.search(r"TRACE:\s*(.*?)\s*ANSWER:", text, flags=re.DOTALL)
        if match:
            return [part.strip() for part in match.group(1).split() if part.strip()]
    return []


def token_family_record(token: str, config: dict[str, Any]) -> dict[str, Any]:
    exact = config.get("exact_tokens", {})
    prefix_rules = config.get("prefix_rules", [])
    token_key = str(token).strip()
    if token_key in exact and isinstance(exact[token_key], dict):
        return dict(exact[token_key])
    for rule in prefix_rules:
        if not isinstance(rule, dict):
            continue
        prefix = str(rule.get("prefix", "")).strip()
        if prefix and token_key.startswith(prefix):
            return dict(rule)
    return dict(config.get("fallback", {"family": "control", "arity": None}))


def build_typed_targets(
    *,
    raw_text: str,
    mode: str,
    config: dict[str, Any],
    row: dict[str, Any] | None = None,
) -> TypedTargets:
    tokens = extract_trace_tokens_from_sample(raw_text, mode, row=row)
    if not tokens:
        return TypedTargets(
            has_supervision=False,
            trace_tokens=tuple(),
            family_ids=tuple(),
            family_histogram=tuple(0.0 for _ in FAMILY_TO_ID),
            primary_arity=None,
            pointer_budget=None,
        )
    family_ids: list[int] = []
    primary_arity: int | None = None
    for token in tokens:
        record = token_family_record(token, config)
        family = str(record.get("family", "control")).strip().lower()
        family_ids.append(FAMILY_TO_ID.get(family, FAMILY_TO_ID["control"]))
        arity = _coerce_arity(record.get("arity"))
        if primary_arity is None and family == "gismu" and arity is not None:
            primary_arity = arity
    histogram = family_histogram_from_ids(family_ids)
    pointer_budget = primary_arity if primary_arity is not None else None
    return TypedTargets(
        has_supervision=True,
        trace_tokens=tuple(tokens),
        family_ids=tuple(family_ids),
        family_histogram=tuple(histogram),
        primary_arity=primary_arity,
        pointer_budget=pointer_budget,
    )


def family_histogram_from_ids(family_ids: list[int] | tuple[int, ...]) -> list[float]:
    counts = [0.0 for _ in FAMILY_TO_ID]
    for family_id in family_ids:
        if 0 <= int(family_id) < len(counts):
            counts[int(family_id)] += 1.0
    total = sum(counts)
    if total <= 0.0:
        return counts
    return [value / total for value in counts]


def mean_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return entropy.mean()


def symbolic_trace_alignment_score(
    slot_family_logits: torch.Tensor,
    target_histogram: torch.Tensor | None,
) -> torch.Tensor:
    if target_histogram is None or target_histogram.numel() == 0:
        return torch.zeros((), device=slot_family_logits.device, dtype=slot_family_logits.dtype)
    probs = F.softmax(slot_family_logits, dim=-1).mean(dim=0)
    target = target_histogram.to(device=slot_family_logits.device, dtype=slot_family_logits.dtype)
    if float(target.sum().item()) <= 0.0:
        return torch.zeros((), device=slot_family_logits.device, dtype=slot_family_logits.dtype)
    return F.cosine_similarity(
        probs.unsqueeze(0),
        target.unsqueeze(0),
        dim=-1,
    ).mean()


def family_separation_loss(query_state: torch.Tensor, slot_layout: list[str]) -> torch.Tensor:
    if query_state.ndim != 3 or query_state.shape[1] < 2:
        return torch.zeros((), device=query_state.device, dtype=query_state.dtype)
    family_means: list[torch.Tensor] = []
    for family in FAMILY_TO_ID:
        indices = [idx for idx, name in enumerate(slot_layout) if name == family]
        if not indices:
            continue
        family_means.append(query_state[:, indices, :].mean(dim=1).mean(dim=0))
    if len(family_means) < 2:
        return torch.zeros((), device=query_state.device, dtype=query_state.dtype)
    normalized = F.normalize(torch.stack(family_means, dim=0), dim=-1)
    cosine = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)
    return torch.relu(cosine.masked_select(mask)).mean()


def slot_usage_balance_loss(judri_mask: torch.Tensor | None) -> torch.Tensor:
    if judri_mask is None or judri_mask.numel() == 0:
        if judri_mask is not None:
            return torch.zeros((), device=judri_mask.device, dtype=judri_mask.dtype)
        return torch.tensor(0.0)
    mean_use = judri_mask.float().mean(dim=0)
    target = torch.full_like(mean_use, mean_use.mean())
    return ((mean_use - target) ** 2).mean()


def project_poincare_ball(x: torch.Tensor, curvature: float, eps: float = DEFAULT_POINCARE_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    c = max(float(curvature), eps)
    max_norm = (1.0 - eps) / math.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    projected = x * scale
    clip_mask = (scale < 0.999999).to(dtype=x.dtype)
    return projected, clip_mask.squeeze(-1)


def expmap0(u: torch.Tensor, curvature: float, eps: float = DEFAULT_POINCARE_EPS) -> torch.Tensor:
    c = max(float(curvature), eps)
    sqrt_c = math.sqrt(c)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scaled = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    projected, _ = project_poincare_ball(scaled, c, eps=eps)
    return projected


def logmap0(y: torch.Tensor, curvature: float, eps: float = DEFAULT_POINCARE_EPS) -> torch.Tensor:
    c = max(float(curvature), eps)
    sqrt_c = math.sqrt(c)
    norm = y.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1.0 - eps) / sqrt_c
    clamped = torch.clamp(norm, max=max_norm)
    return torch.atanh(sqrt_c * clamped) * y / (sqrt_c * clamped)


def mobius_matvec(weight: torch.Tensor, x: torch.Tensor, curvature: float, eps: float = DEFAULT_POINCARE_EPS) -> torch.Tensor:
    tangent = logmap0(x, curvature, eps=eps)
    projected = tangent @ weight.transpose(0, 1)
    return expmap0(projected, curvature, eps=eps)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float, eps: float = DEFAULT_POINCARE_EPS) -> torch.Tensor:
    c = max(float(curvature), eps)
    sqrt_c = math.sqrt(c)
    x_norm_sq = (x * x).sum(dim=-1).clamp_max((1.0 - eps) / c)
    y_norm_sq = (y * y).sum(dim=-1).clamp_max((1.0 - eps) / c)
    diff_sq = ((x - y) ** 2).sum(dim=-1)
    denom = (1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq)
    argument = 1.0 + 2.0 * c * diff_sq / denom.clamp_min(eps)
    return torch.acosh(argument.clamp_min(1.0 + eps)) / sqrt_c


def apply_radius_bands(
    x: torch.Tensor,
    slot_layout: list[str],
    bands: dict[str, dict[str, float]],
    curvature: float,
    eps: float = DEFAULT_POINCARE_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected, clip_mask = project_poincare_ball(x, curvature, eps=eps)
    if projected.ndim != 3:
        return projected, clip_mask
    max_radius = (1.0 - eps) / math.sqrt(max(float(curvature), eps))
    adjusted_slices: list[torch.Tensor] = []
    band_clips: list[torch.Tensor] = []
    for idx in range(projected.shape[1]):
        family = slot_layout[idx] if idx < len(slot_layout) else "control"
        band = bands.get(str(family), {})
        min_radius = max(0.0, float(band.get("min", 0.0)))
        max_band_radius = min(max_radius, float(band.get("max", max_radius)))
        slice_x = projected[:, idx, :]
        radius = slice_x.norm(dim=-1, keepdim=True).clamp_min(eps)
        clamped_radius = radius.clamp(min=min_radius, max=max_band_radius)
        adjusted_slices.append(slice_x * (clamped_radius / radius))
        band_clips.append((torch.abs(clamped_radius - radius) > 1e-6).squeeze(-1).to(dtype=projected.dtype))
    adjusted = torch.stack(adjusted_slices, dim=1)
    band_clip = torch.stack(band_clips, dim=1)
    return adjusted, torch.maximum(clip_mask, band_clip)


def hyperbolic_family_metrics(
    query_state: torch.Tensor,
    slot_layout: list[str],
    curvature: float,
    bands: dict[str, dict[str, float]],
) -> dict[str, float]:
    if query_state.ndim != 3 or query_state.shape[1] == 0:
        return {
            "predicate_pointer_radial_gap": 0.0,
            "family_radius_violation_rate": 0.0,
            "hyperbolic_geodesic_margin": 0.0,
            "hyperbolic_projection_clip_rate": 0.0,
        }
    projected, clip_mask = apply_radius_bands(query_state, slot_layout, bands, curvature)
    norms = projected.norm(dim=-1)
    family_radius_means: dict[str, torch.Tensor] = {}
    violation = torch.zeros_like(norms)
    max_radius = (1.0 - DEFAULT_POINCARE_EPS) / math.sqrt(max(float(curvature), DEFAULT_POINCARE_EPS))
    for idx, family in enumerate(slot_layout[: projected.shape[1]]):
        family_radius_means.setdefault(str(family), []).append(norms[:, idx])
        band = bands.get(str(family), {})
        min_radius = float(band.get("min", 0.0))
        max_band = min(max_radius, float(band.get("max", max_radius)))
        violation[:, idx] = ((norms[:, idx] < min_radius) | (norms[:, idx] > max_band)).to(dtype=norms.dtype)
    reduced_radius_means = {
        family: torch.stack(values, dim=0).mean()
        for family, values in family_radius_means.items()
        if values
    }
    predicate_gap = 0.0
    if "gismu" in reduced_radius_means and "judri" in reduced_radius_means:
        predicate_gap = float((reduced_radius_means["judri"] - reduced_radius_means["gismu"]).item())
    geodesic_margin = 0.0
    gismu_indices = [idx for idx, family in enumerate(slot_layout[: projected.shape[1]]) if family == "gismu"]
    judri_indices = [idx for idx, family in enumerate(slot_layout[: projected.shape[1]]) if family == "judri"]
    if gismu_indices and judri_indices:
        distances: list[torch.Tensor] = []
        for g_idx in gismu_indices:
            for j_idx in judri_indices:
                distances.append(poincare_distance(projected[:, g_idx, :], projected[:, j_idx, :], curvature))
        if distances:
            geodesic_margin = float(torch.stack(distances, dim=0).mean().item())
    return {
        "predicate_pointer_radial_gap": float(predicate_gap),
        "family_radius_violation_rate": float(violation.mean().item()),
        "hyperbolic_geodesic_margin": float(geodesic_margin),
        "hyperbolic_projection_clip_rate": float(clip_mask.mean().item()),
    }


def _coerce_arity(value: Any) -> int | None:
    if value is None:
        return None
    try:
        arity = int(value)
    except (TypeError, ValueError):
        return None
    if arity <= 0:
        return None
    return arity


def _validate_slot_families(values: list[str]) -> None:
    if not values:
        raise ValueError("typed slot layout must include at least one slot family")
    invalid = [value for value in values if value not in FAMILY_TO_ID]
    if invalid:
        raise ValueError(f"unsupported typed slot families: {invalid}")
