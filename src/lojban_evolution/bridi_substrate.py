from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lojban_evolution.m21.bridi import CMAVO, DEFAULT_MAX_PLACES, GISMU


@dataclass(frozen=True)
class PackedBridiTraceSpec:
    """Integer-only bridi trace layout shared by substrate experiments."""

    max_frames: int = 6
    max_places: int = DEFAULT_MAX_PLACES
    cmavo_count: int = len(CMAVO)
    gismu_count: int = len(GISMU)

    @property
    def width(self) -> int:
        return 3 + int(self.cmavo_count) + int(self.max_places)

    @property
    def active_col(self) -> int:
        return 0

    @property
    def stop_col(self) -> int:
        return 1

    @property
    def gismu_col(self) -> int:
        return 2

    @property
    def cmavo_start(self) -> int:
        return 3

    @property
    def judri_start(self) -> int:
        return 3 + int(self.cmavo_count)


def packed_trace_spec(*, max_frames: int = 6, max_places: int = DEFAULT_MAX_PLACES) -> PackedBridiTraceSpec:
    return PackedBridiTraceSpec(max_frames=int(max_frames), max_places=int(max_places))


def pack_symbolic_trace_from_batch(batch: dict[str, Any], *, max_frames: int | None = None, max_places: int = DEFAULT_MAX_PLACES) -> torch.Tensor:
    """Pack gold bridi targets into integer symbolic traces.

    Layout per frame: active, stop, gismu_id, one binary column per cmavo,
    then judri/place binding ids. The result is `long`; it contains no hidden
    states, prompt states, or continuous frame representations.
    """

    active = batch["active_targets"].detach()
    stop = batch["stop_targets"].detach()
    gismu = batch["gismu_targets"].detach()
    cmavo = batch["cmavo_targets"].detach()
    judri = batch["judri_targets"].detach()
    frames = int(max_frames or active.shape[1])
    spec = packed_trace_spec(max_frames=frames, max_places=int(max_places))
    device = active.device
    out = torch.zeros(active.shape[0], frames, spec.width, dtype=torch.long, device=device)
    frame_slice = slice(0, min(frames, active.shape[1]))
    cmavo_width = min(spec.cmavo_count, cmavo.shape[-1])
    judri_width = min(spec.max_places, judri.shape[-1])
    out[:, frame_slice, spec.active_col] = (active[:, frame_slice] > 0.5).long()
    out[:, frame_slice, spec.stop_col] = (stop[:, frame_slice] > 0.5).long()
    out[:, frame_slice, spec.gismu_col] = gismu[:, frame_slice].long().clamp(0, spec.gismu_count - 1)
    out[:, frame_slice, spec.cmavo_start : spec.cmavo_start + cmavo_width] = (cmavo[:, frame_slice, :cmavo_width] > 0.5).long()
    out[:, frame_slice, spec.judri_start : spec.judri_start + judri_width] = judri[:, frame_slice, :judri_width].long().clamp_min(0)
    return out


@torch.no_grad()
def pack_symbolic_trace_from_outputs(outputs: dict[str, torch.Tensor], *, max_frames: int | None = None, max_places: int = DEFAULT_MAX_PLACES) -> torch.Tensor:
    """Pack predicted bridi logits into integer symbolic traces.

    This is the M24 advisor contract: use discrete predictions only. Callers
    should not pass `frame_repr`, `trace_state`, or `prompt_state` downstream.
    """

    active_logits = outputs["active_logits"].detach()
    stop_logits = outputs["stop_logits"].detach()
    gismu_logits = outputs["gismu_logits"].detach()
    cmavo_logits = outputs["cmavo_logits"].detach()
    judri_logits = outputs["judri_logits"].detach()
    frames = int(max_frames or active_logits.shape[1])
    spec = packed_trace_spec(max_frames=frames, max_places=int(max_places))
    device = active_logits.device
    out = torch.zeros(active_logits.shape[0], frames, spec.width, dtype=torch.long, device=device)
    frame_slice = slice(0, min(frames, active_logits.shape[1]))
    cmavo_width = min(spec.cmavo_count, cmavo_logits.shape[-1])
    judri_width = min(spec.max_places, judri_logits.shape[-2])
    out[:, frame_slice, spec.active_col] = (torch.sigmoid(active_logits[:, frame_slice]) > 0.5).long()
    out[:, frame_slice, spec.stop_col] = (torch.sigmoid(stop_logits[:, frame_slice]) > 0.5).long()
    out[:, frame_slice, spec.gismu_col] = torch.argmax(gismu_logits[:, frame_slice], dim=-1).long()
    out[:, frame_slice, spec.cmavo_start : spec.cmavo_start + cmavo_width] = (torch.sigmoid(cmavo_logits[:, frame_slice, :cmavo_width]) > 0.5).long()
    out[:, frame_slice, spec.judri_start : spec.judri_start + judri_width] = torch.argmax(judri_logits[:, frame_slice, :judri_width], dim=-1).long()
    return out


def zero_packed_trace_like(packed: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(packed, dtype=torch.long)


def random_packed_trace_like(packed: torch.Tensor, *, seed: int = 0, max_entities: int = 8) -> torch.Tensor:
    """Return a random integer trace with the same symbolic layout."""

    spec = packed_trace_spec(max_frames=int(packed.shape[1]), max_places=max(0, int(packed.shape[2]) - 3 - len(CMAVO)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    out = torch.zeros_like(packed, dtype=torch.long)
    shape2 = (packed.shape[0], packed.shape[1])
    out[..., spec.active_col] = torch.randint(0, 2, shape2, generator=generator, device="cpu").to(packed.device)
    out[..., spec.stop_col] = torch.randint(0, 2, shape2, generator=generator, device="cpu").to(packed.device)
    out[..., spec.gismu_col] = torch.randint(0, len(GISMU), shape2, generator=generator, device="cpu").to(packed.device)
    cmavo_shape = (packed.shape[0], packed.shape[1], len(CMAVO))
    out[..., spec.cmavo_start : spec.cmavo_start + len(CMAVO)] = torch.randint(0, 2, cmavo_shape, generator=generator, device="cpu").to(packed.device)
    judri_width = packed.shape[2] - spec.judri_start
    if judri_width > 0:
        out[..., spec.judri_start :] = torch.randint(0, int(max_entities) + 1, (packed.shape[0], packed.shape[1], judri_width), generator=generator, device="cpu").to(packed.device)
    return out


def packed_trace_exact_accuracy(predicted: torch.Tensor, oracle: torch.Tensor) -> float:
    if predicted.numel() == 0 or oracle.numel() == 0:
        return 0.0
    width = min(predicted.shape[-1], oracle.shape[-1])
    frames = min(predicted.shape[-2], oracle.shape[-2])
    exact = predicted[:, :frames, :width].long().eq(oracle[:, :frames, :width].long()).all(dim=(-1, -2))
    return float(exact.float().mean().detach().cpu().item())


def packed_trace_component_accuracy(predicted: torch.Tensor, oracle: torch.Tensor) -> dict[str, float]:
    spec = packed_trace_spec(max_frames=min(predicted.shape[1], oracle.shape[1]))
    pred = predicted[:, : spec.max_frames].long()
    gold = oracle[:, : spec.max_frames].long()
    active = gold[..., spec.active_col] > 0
    any_active = bool(active.any().detach().cpu().item())
    out = {
        "bridi_trace_exact_accuracy": packed_trace_exact_accuracy(pred, gold),
        "active_accuracy": float(pred[..., spec.active_col].eq(gold[..., spec.active_col]).float().mean().detach().cpu().item()),
        "stop_accuracy": float(pred[..., spec.stop_col].eq(gold[..., spec.stop_col]).float().mean().detach().cpu().item()),
    }
    if any_active:
        out["gismu_accuracy"] = float(pred[..., spec.gismu_col][active].eq(gold[..., spec.gismu_col][active]).float().mean().detach().cpu().item())
        cmavo_pred = pred[..., spec.cmavo_start : spec.cmavo_start + len(CMAVO)]
        cmavo_gold = gold[..., spec.cmavo_start : spec.cmavo_start + len(CMAVO)]
        out["cmavo_accuracy"] = float(cmavo_pred[active].eq(cmavo_gold[active]).all(dim=-1).float().mean().detach().cpu().item())
        judri_pred = pred[..., spec.judri_start :]
        judri_gold = gold[..., spec.judri_start :]
        out["judri_accuracy"] = float(judri_pred[active].eq(judri_gold[active]).all(dim=-1).float().mean().detach().cpu().item())
        out["judri_binding_accuracy"] = out["judri_accuracy"]
    else:
        out.update({"gismu_accuracy": 0.0, "cmavo_accuracy": 0.0, "judri_accuracy": 0.0, "judri_binding_accuracy": 0.0})
    return out


def trace_exact_surrogate_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    """Differentiable pressure for whole-trace exactness across bridi fields."""

    device = outputs["active_logits"].device
    active_targets = batch["active_targets"].to(device)
    active_mask = active_targets > 0.5
    stop_targets = batch["stop_targets"].to(device)
    gismu_targets = batch["gismu_targets"].to(device)
    cmavo_targets = batch["cmavo_targets"].to(device)
    judri_targets = batch["judri_targets"].to(device)
    active_loss = F.binary_cross_entropy_with_logits(outputs["active_logits"], active_targets, reduction="none").sum(dim=-1)
    stop_loss = F.binary_cross_entropy_with_logits(outputs["stop_logits"], stop_targets, reduction="none").sum(dim=-1)
    gismu_loss = F.cross_entropy(
        outputs["gismu_logits"].reshape(-1, outputs["gismu_logits"].shape[-1]),
        gismu_targets.reshape(-1),
        reduction="none",
    ).view_as(gismu_targets)
    cmavo_loss = F.binary_cross_entropy_with_logits(outputs["cmavo_logits"], cmavo_targets, reduction="none").sum(dim=-1)
    judri_loss = F.cross_entropy(
        outputs["judri_logits"].reshape(-1, outputs["judri_logits"].shape[-1]),
        judri_targets.reshape(-1),
        reduction="none",
    ).view(judri_targets.shape).sum(dim=-1)
    frame_loss = ((gismu_loss + cmavo_loss + judri_loss) * active_mask.float()).sum(dim=-1)
    return (active_loss + stop_loss + frame_loss).mean()


def assert_symbolic_trace_contract(value: torch.Tensor, *, name: str = "packed_trace") -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype not in (torch.long, torch.int64, torch.int32, torch.int16, torch.uint8):
        raise TypeError(f"{name} must be integer packed symbols, got {value.dtype}.")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, frames, symbolic_width], got {tuple(value.shape)}.")
