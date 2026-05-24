from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import (
    ANSWER_LABELS,
    ANSWER_TO_ID,
    CMAVO,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PLACES,
    DEFAULT_POINCARE_MAX_NORM,
    GISMU,
    BridiFrame,
    DynamicBridiExample,
    M21BridiDataset,
    M21DynamicBridiQFormer,
    build_vocab,
    compute_m21_loss,
    generate_dynamic_bridi_adversarial_examples,
    generate_dynamic_bridi_examples,
    poincare_tangent_handoff,
    tokenize,
)


M23_RELEVANCE_SURFACES = ("clean", "decoy_relation_ood")
M23_LOCKS: dict[str, str] = {
    "scale_control": "M22-style dynamic bridi substrate trained with more decoy-balanced data and no new router.",
    "relevance_router": "lightweight frame scoring head ranks active bridi frames before the answer bridge reads them.",
    "oracle_relevance": "eval-only gold relevant-frame read path checks whether relevance is usable at all.",
    "random_relevance": "eval-only randomized relevance read path should hurt decoy OOD if relevance matters.",
    "decoy_only": "eval-only decoy-frame read path exposes frame-selection bypasses.",
}


@dataclass(frozen=True)
class M23RelevanceExample:
    prompt: str
    frames: tuple[BridiFrame, ...]
    entities: tuple[str, ...]
    answer_id: int
    answer_label: str
    surface: str
    counterfactual_group: str
    entity_signature: str
    relevant_frame_indices: tuple[int, ...]
    decoy_frame_indices: tuple[int, ...]
    relevance_surface: str
    is_floating: bool = False
    template_id: str = ""
    template_family: str = ""
    is_relation_ood: bool = False

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_json() for frame in self.frames]
        return payload


class M23RelevanceDataset(M21BridiDataset):
    examples: list[M23RelevanceExample]

    def __init__(self, examples: Sequence[M23RelevanceExample], vocab: dict[str, int], *, max_length: int = 64, max_frames: int = DEFAULT_MAX_FRAMES):
        super().__init__(examples, vocab, max_length=max_length, max_frames=max_frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        item = super().__getitem__(idx)
        relevant = torch.zeros(self.max_frames, dtype=torch.float32)
        decoy = torch.zeros(self.max_frames, dtype=torch.float32)
        for frame_idx in row.relevant_frame_indices:
            if 0 <= int(frame_idx) < self.max_frames:
                relevant[int(frame_idx)] = 1.0
        for frame_idx in row.decoy_frame_indices:
            if 0 <= int(frame_idx) < self.max_frames:
                decoy[int(frame_idx)] = 1.0
        item["relevance_targets"] = relevant
        item["decoy_targets"] = decoy
        item["relevance_surface"] = row.relevance_surface
        return item


def m23_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "input_ids",
        "active_targets",
        "stop_targets",
        "gismu_targets",
        "cmavo_targets",
        "judri_targets",
        "answer_id",
        "relevance_targets",
        "decoy_targets",
    )
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ("surface", "counterfactual_group", "entity_signature", "prompt", "relevance_surface"):
        out[key] = [str(item[key]) for item in batch]
    return out


def _as_m23_clean(row: DynamicBridiExample) -> M23RelevanceExample:
    active_indices = tuple(idx for idx, frame in enumerate(row.frames[:DEFAULT_MAX_FRAMES]) if frame.active)
    if not active_indices:
        active_indices = (0,)
    return M23RelevanceExample(
        prompt=row.prompt,
        frames=row.frames,
        entities=row.entities,
        answer_id=row.answer_id,
        answer_label=row.answer_label,
        surface=row.surface,
        counterfactual_group=row.counterfactual_group,
        entity_signature=row.entity_signature,
        relevant_frame_indices=active_indices,
        decoy_frame_indices=tuple(),
        relevance_surface="clean",
        is_floating=row.is_floating,
        template_id=row.template_id,
        template_family=row.template_family,
        is_relation_ood=row.is_relation_ood,
    )


def _active_nonstop_frames(row: DynamicBridiExample) -> tuple[BridiFrame, ...]:
    return tuple(frame for frame in row.frames if frame.active)


def _decoy_pool(size: int, *, seed: int) -> list[DynamicBridiExample]:
    return generate_dynamic_bridi_adversarial_examples(
        max(int(size), len(ANSWER_LABELS) * 2),
        seed=int(seed),
        surfaces=("decoy_relation_ood",),
    )


def generate_m23_relevance_examples(
    size: int,
    *,
    seed: int = 0,
    clean_fraction: float = 0.35,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[M23RelevanceExample]:
    """Generate decoy-balanced M23 rows with explicit relevant and decoy frame masks."""

    rng = random.Random(int(seed))
    clean_count = int(round(max(0, int(size)) * min(1.0, max(0.0, float(clean_fraction)))))
    decoy_count = max(0, int(size) - clean_count)
    rows: list[M23RelevanceExample] = []
    if clean_count:
        rows.extend(_as_m23_clean(row) for row in generate_dynamic_bridi_examples(clean_count, seed=int(seed), floating_fraction=0.0))
    if decoy_count:
        relevant_rows = generate_dynamic_bridi_adversarial_examples(
            decoy_count,
            seed=int(seed) + 10_000,
            surfaces=("decoy_relation_ood",),
        )
        pool = _decoy_pool(decoy_count * 4 + len(ANSWER_LABELS), seed=int(seed) + 20_000)
        pool_by_label: dict[str, list[DynamicBridiExample]] = defaultdict(list)
        for candidate in pool:
            pool_by_label[candidate.answer_label].append(candidate)
        labels = list(ANSWER_LABELS)
        for idx, row in enumerate(relevant_rows):
            decoy_labels = [label for label in labels if label != row.answer_label]
            rng.shuffle(decoy_labels)
            decoy_source: DynamicBridiExample | None = None
            for label in decoy_labels:
                if pool_by_label[label]:
                    decoy_source = pool_by_label[label].pop()
                    break
            if decoy_source is None:
                decoy_source = rng.choice([candidate for candidate in pool if candidate.answer_label != row.answer_label])
            decoy_frames = _active_nonstop_frames(decoy_source)
            relevant_frames = _active_nonstop_frames(row)
            raw_decoy_frame = decoy_frames[-1] if decoy_frames else row.frames[0]
            decoy_frame = BridiFrame(
                active=True,
                gismu_id=raw_decoy_frame.gismu_id,
                cmavo_ids=raw_decoy_frame.cmavo_ids,
                judri_place_bindings=raw_decoy_frame.judri_place_bindings,
                stop=False,
            )
            insert_first = bool((idx + int(seed)) % 2)
            if insert_first:
                frames = (decoy_frame, *relevant_frames)
                decoy_indices = (0,)
                relevant_indices = tuple(range(1, min(len(frames), int(max_frames))))
            else:
                frames = (*relevant_frames, decoy_frame)
                relevant_indices = tuple(range(0, min(len(relevant_frames), int(max_frames))))
                decoy_indices = (min(len(relevant_frames), int(max_frames) - 1),)
            frames = frames[: int(max_frames)]
            relevant_indices = tuple(idx for idx in relevant_indices if idx < len(frames))
            decoy_indices = tuple(idx for idx in decoy_indices if idx < len(frames))
            if not relevant_indices:
                relevant_indices = (0,)
            if not decoy_indices:
                decoy_indices = (len(frames) - 1,)
            rows.append(
                M23RelevanceExample(
                    prompt=row.prompt,
                    frames=tuple(frames),
                    entities=row.entities,
                    answer_id=row.answer_id,
                    answer_label=row.answer_label,
                    surface="decoy_relation_ood",
                    counterfactual_group=f"m23:{row.answer_label}",
                    entity_signature=row.entity_signature,
                    relevant_frame_indices=relevant_indices,
                    decoy_frame_indices=decoy_indices,
                    relevance_surface="decoy_relation_ood",
                    is_floating=False,
                    template_id=row.template_id,
                    template_family="m23_relevance_decoy",
                    is_relation_ood=True,
                )
            )
    rng.shuffle(rows)
    return rows


class M23CausalRelevanceQFormer(M21DynamicBridiQFormer):
    """M21/M22 dynamic bridi substrate plus a small answer-causal frame relevance head."""

    def __init__(self, *args: Any, relevance_temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.relevance_temperature = float(relevance_temperature)
        self.relevance_head = nn.Linear(self.answer_head.in_features, 1)

    def _answer_from_trace_state(self, state: torch.Tensor) -> torch.Tensor:
        answer_state = state
        if self.geometry_mode == "poincare":
            answer_state, _clip = poincare_tangent_handoff(
                state,
                curvature=self.poincare_curvature,
                max_norm=self.poincare_max_norm,
            )
        return self.answer_head(answer_state)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(input_ids)
        frame_state = outputs["frame_state"]
        frame_repr = outputs["frame_repr"]
        active_prob = outputs["active_prob"].clamp(0.0, 1.0)
        relevance_logits = self.relevance_head(frame_state).squeeze(-1)
        temperature = max(float(self.relevance_temperature), 1e-6)
        active_bias = (active_prob + 1e-6).log()
        relevance_weights = torch.softmax((relevance_logits + active_bias) / temperature, dim=-1)
        relevance_trace_state = (frame_repr * relevance_weights.unsqueeze(-1)).sum(dim=1)
        outputs["relevance_logits"] = relevance_logits
        outputs["relevance_weights"] = relevance_weights
        outputs["relevance_trace_state"] = relevance_trace_state
        outputs["relevance_answer_logits"] = self._answer_from_trace_state(relevance_trace_state)
        return outputs


def relevance_rank_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], *, margin: float = 0.15) -> torch.Tensor:
    logits = outputs["relevance_logits"]
    device = logits.device
    relevant = batch["relevance_targets"].to(device) > 0.5
    decoy = batch["decoy_targets"].to(device) > 0.5
    valid = relevant.any(dim=-1) & decoy.any(dim=-1)
    if not bool(valid.any().detach().cpu().item()):
        return logits.new_zeros(())
    low = torch.finfo(logits.dtype).min / 4.0
    relevant_score = logits.masked_fill(~relevant, low).max(dim=-1).values
    decoy_score = logits.masked_fill(~decoy, low).max(dim=-1).values
    return torch.relu(float(margin) + decoy_score[valid] - relevant_score[valid]).mean()


def trace_exact_surrogate_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    """Differentiable pressure for whole-trace exactness, not just averaged component quality."""

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


def compute_m23_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    use_relevance_answer: bool,
    relevance_rank_weight: float = 1.0,
    relevance_margin: float = 0.15,
    trace_exact_surrogate_weight: float = 0.0,
    **m21_loss_kwargs: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_outputs = dict(outputs)
    if bool(use_relevance_answer):
        loss_outputs["answer_logits"] = outputs["relevance_answer_logits"]
    total, pieces = compute_m21_loss(loss_outputs, batch, **m21_loss_kwargs)
    rank = relevance_rank_loss(outputs, batch, margin=float(relevance_margin))
    total = total + float(relevance_rank_weight) * rank
    trace_exact = trace_exact_surrogate_loss(outputs, batch)
    total = total + float(trace_exact_surrogate_weight) * trace_exact
    pieces["loss_relevance_rank"] = float(rank.detach().cpu().item())
    pieces["loss_trace_exact_surrogate"] = float(trace_exact.detach().cpu().item())
    pieces["relevance_rank_weight"] = float(relevance_rank_weight)
    pieces["relevance_margin"] = float(relevance_margin)
    pieces["trace_exact_surrogate_weight"] = float(trace_exact_surrogate_weight)
    return total, pieces


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    return float((torch.argmax(logits, dim=-1) == target).float().mean().detach().cpu().item())


def _masked_state(frame_repr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    return (frame_repr * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def _random_relevance_mask(relevant: torch.Tensor, decoy: torch.Tensor, active: torch.Tensor, *, seed: int = 17) -> torch.Tensor:
    eligible = active.bool() & (~relevant.bool())
    fallback = active.bool()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scores = torch.rand(eligible.shape, generator=generator).to(active.device)
    scores = scores.masked_fill(~eligible, -1.0)
    no_eligible = ~eligible.any(dim=-1)
    if bool(no_eligible.any().detach().cpu().item()):
        fallback_scores = torch.rand(fallback.shape, generator=generator).to(active.device).masked_fill(~fallback, -1.0)
        scores[no_eligible] = fallback_scores[no_eligible]
    picked = torch.argmax(scores, dim=-1)
    out = torch.zeros_like(active, dtype=torch.float32)
    out.scatter_(1, picked.unsqueeze(-1), 1.0)
    out = torch.where(decoy.bool(), decoy.float(), out)
    return out


@torch.no_grad()
def evaluate_m23_model(
    model: M23CausalRelevanceQFormer,
    examples: Sequence[M23RelevanceExample],
    vocab: dict[str, int],
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    use_relevance_router: bool = True,
) -> dict[str, Any]:
    model.eval()
    dataset = M23RelevanceDataset(examples, vocab, max_frames=model.max_frames)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m23_collate)
    merged: dict[str, list[torch.Tensor]] = defaultdict(list)
    surfaces: list[str] = []
    prompts: list[str] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids)
        primary_logits = outputs["relevance_answer_logits"] if use_relevance_router else outputs["answer_logits"]
        frame_repr = outputs["frame_repr"]
        relevant = batch["relevance_targets"].to(device) > 0.5
        decoy = batch["decoy_targets"].to(device) > 0.5
        active = batch["active_targets"].to(device) > 0.5
        oracle_logits = model._answer_from_trace_state(_masked_state(frame_repr, relevant))
        no_relevance_logits = model._answer_from_trace_state(_masked_state(frame_repr, active))
        decoy_logits = model._answer_from_trace_state(_masked_state(frame_repr, decoy))
        random_mask = _random_relevance_mask(relevant, decoy, active)
        random_logits = model._answer_from_trace_state(_masked_state(frame_repr, random_mask > 0.5))
        for key, tensor in {
            "primary_answer_logits": primary_logits,
            "answer_logits": outputs["answer_logits"],
            "relevance_answer_logits": outputs["relevance_answer_logits"],
            "oracle_relevance_answer_logits": oracle_logits,
            "random_relevance_answer_logits": random_logits,
            "no_relevance_answer_logits": no_relevance_logits,
            "decoy_only_answer_logits": decoy_logits,
            "relevance_logits": outputs["relevance_logits"],
            "active_logits": outputs["active_logits"],
            "stop_logits": outputs["stop_logits"],
            "gismu_logits": outputs["gismu_logits"],
            "cmavo_logits": outputs["cmavo_logits"],
            "judri_logits": outputs["judri_logits"],
        }.items():
            merged[key].append(tensor.detach().cpu())
        for key in ("active_targets", "stop_targets", "gismu_targets", "cmavo_targets", "judri_targets", "answer_id", "relevance_targets", "decoy_targets"):
            merged[key].append(batch[key].detach().cpu())
        surfaces.extend(batch["surface"])
        prompts.extend(batch["prompt"])
    tensors = {key: torch.cat(values, dim=0) for key, values in merged.items()}
    target = tensors["answer_id"]
    active_target = tensors["active_targets"] > 0.5
    active_pred = torch.sigmoid(tensors["active_logits"]) > 0.5
    gismu_pred = torch.argmax(tensors["gismu_logits"], dim=-1)
    cmavo_pred = torch.sigmoid(tensors["cmavo_logits"]) > 0.5
    cmavo_target = tensors["cmavo_targets"] > 0.5
    judri_pred = torch.argmax(tensors["judri_logits"], dim=-1)
    judri_target = tensors["judri_targets"]
    stop_pred_idx = torch.argmax(tensors["stop_logits"], dim=-1)
    stop_target_idx = torch.argmax(tensors["stop_targets"], dim=-1)
    frame_exact = (~active_target) | ((gismu_pred == tensors["gismu_targets"]) & ((cmavo_pred == cmavo_target).all(dim=-1)) & ((judri_pred == judri_target).all(dim=-1)))
    trace_exact = (active_pred == active_target).all(dim=-1) & frame_exact.all(dim=-1) & (stop_pred_idx == stop_target_idx)
    relevant = tensors["relevance_targets"] > 0.5
    decoy = tensors["decoy_targets"] > 0.5
    valid_rel = relevant.any(dim=-1)
    top = torch.argmax(tensors["relevance_logits"], dim=-1)
    rel_top1 = float(relevant[torch.arange(relevant.shape[0]), top].float().mean().item()) if relevant.numel() else 0.0
    low = torch.finfo(tensors["relevance_logits"].dtype).min / 4.0
    rel_score = tensors["relevance_logits"].masked_fill(~relevant, low).max(dim=-1).values
    decoy_score = tensors["relevance_logits"].masked_fill(~decoy, low).max(dim=-1).values
    valid_margin = valid_rel & decoy.any(dim=-1)
    relevance_margin = float((rel_score[valid_margin] - decoy_score[valid_margin]).mean().item()) if bool(valid_margin.any().item()) else 0.0
    surface_metrics: dict[str, dict[str, float]] = {}
    primary_pred = torch.argmax(tensors["primary_answer_logits"], dim=-1)
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([value == surface for value in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {
            "strict_accuracy": float((primary_pred[mask] == target[mask]).float().mean().item()) if bool(mask.any().item()) else 0.0,
            "bridi_trace_exact_accuracy": float(trace_exact[mask].float().mean().item()) if bool(mask.any().item()) else 0.0,
            "count": float(mask.sum().item()),
        }
    avg_tokens = float(sum(len(tokenize(prompt)) for prompt in prompts) / max(1, len(prompts)))
    frame_count_pred = active_pred.sum(dim=-1).float()
    trace_tokens = float(frame_count_pred.mean().item() * (1 + DEFAULT_MAX_PLACES + len(CMAVO) / 4.0))
    full_accuracy = _accuracy(tensors["primary_answer_logits"], target)
    oracle_accuracy = _accuracy(tensors["oracle_relevance_answer_logits"], target)
    random_accuracy = _accuracy(tensors["random_relevance_answer_logits"], target)
    no_relevance_accuracy = _accuracy(tensors["no_relevance_answer_logits"], target)
    decoy_accuracy = _accuracy(tensors["decoy_only_answer_logits"], target)
    return {
        "strict_accuracy": full_accuracy,
        "synthetic_world_accuracy": full_accuracy,
        "phrase_accuracy": full_accuracy,
        "bridi_trace_exact_accuracy": float(trace_exact.float().mean().item()),
        "gismu_accuracy": float((gismu_pred[active_target] == tensors["gismu_targets"][active_target]).float().mean().item()) if bool(active_target.any().item()) else 0.0,
        "cmavo_accuracy": float(((cmavo_pred == cmavo_target).all(dim=-1))[active_target].float().mean().item()) if bool(active_target.any().item()) else 0.0,
        "judri_binding_accuracy": float(((judri_pred == judri_target).all(dim=-1))[active_target].float().mean().item()) if bool(active_target.any().item()) else 0.0,
        "relevance_top1_accuracy": rel_top1,
        "relevance_margin": relevance_margin,
        "oracle_relevance_accuracy": oracle_accuracy,
        "random_relevance_accuracy": random_accuracy,
        "no_relevance_accuracy": no_relevance_accuracy,
        "decoy_only_accuracy": decoy_accuracy,
        "oracle_relevance_delta": float(oracle_accuracy - full_accuracy),
        "random_relevance_delta": float(full_accuracy - random_accuracy),
        "decoy_only_delta": float(full_accuracy - decoy_accuracy),
        "frame_count_mae": float((frame_count_pred - active_target.sum(dim=-1).float()).abs().mean().item()),
        "mean_active_frames": float(frame_count_pred.mean().item()),
        "decoy_relation_ood_accuracy": float(surface_metrics.get("decoy_relation_ood", {}).get("strict_accuracy", 0.0)),
        "clean_accuracy": float(surface_metrics.get("clean", {}).get("strict_accuracy", surface_metrics.get("purged", {}).get("strict_accuracy", 0.0))),
        "worst_surface_accuracy": min((row["strict_accuracy"] for row in surface_metrics.values()), default=0.0),
        "surface_metrics": surface_metrics,
        "full_accuracy": full_accuracy,
        "avg_tokens": avg_tokens,
        "accuracy_per_token": float(full_accuracy / max(avg_tokens, 1e-8)),
        "trace_tokens": trace_tokens,
        "accuracy_per_trace_token": float(full_accuracy / max(trace_tokens, 1.0)),
    }


def train_m23_relevance_router(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    epochs: int = 16,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 23,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_places: int = DEFAULT_MAX_PLACES,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    trace_weight: float = 1.25,
    answer_weight: float = 1.25,
    counterfactual_weight: float = 1.25,
    brivi_lock_weight: float = 1.5,
    frame_necessity_weight: float = 1.0,
    mdl_weight: float = 0.01,
    necessity_margin: float = 0.04,
    pointer_necessity_weight: float = 0.0,
    pointer_necessity_margin: float = 0.05,
    relevance_rank_weight: float = 0.0,
    relevance_margin: float = 0.15,
    trace_exact_surrogate_weight: float = 0.0,
    use_relevance_router: bool = False,
    relevance_temperature: float = 1.0,
    clean_train_fraction: float = 0.35,
    clean_eval_fraction: float = 0.35,
    geometry_mode: str = "euclidean",
    poincare_curvature: float = 1.0,
    poincare_max_norm: float = DEFAULT_POINCARE_MAX_NORM,
    riemannian_gradient_scale: bool = True,
    judri_bridge_gate: bool = True,
    judri_bridge_gate_temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    train_examples = generate_m23_relevance_examples(
        int(train_size),
        seed=int(seed),
        clean_fraction=float(clean_train_fraction),
        max_frames=int(max_frames),
    )
    eval_examples = generate_m23_relevance_examples(
        int(eval_size),
        seed=int(seed) + 100_000,
        clean_fraction=float(clean_eval_fraction),
        max_frames=int(max_frames),
    )
    vocab = build_vocab(train_examples)  # type: ignore[arg-type]
    dataset = M23RelevanceDataset(train_examples, vocab, max_frames=int(max_frames))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m23_collate)
    model = M23CausalRelevanceQFormer(
        vocab_size=len(vocab),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        max_frames=int(max_frames),
        max_places=int(max_places),
        max_entities=int(max_entities),
        geometry_mode=str(geometry_mode),
        poincare_curvature=float(poincare_curvature),
        poincare_max_norm=float(poincare_max_norm),
        riemannian_gradient_scale=bool(riemannian_gradient_scale),
        judri_bridge_gate=bool(judri_bridge_gate),
        judri_bridge_gate_temperature=float(judri_bridge_gate_temperature),
        relevance_temperature=float(relevance_temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"].to(device))
            loss, pieces = compute_m23_loss(
                outputs,
                batch,
                use_relevance_answer=bool(use_relevance_router),
                relevance_rank_weight=float(relevance_rank_weight),
                relevance_margin=float(relevance_margin),
                trace_exact_surrogate_weight=float(trace_exact_surrogate_weight),
                trace_weight=float(trace_weight),
                answer_weight=float(answer_weight),
                counterfactual_weight=float(counterfactual_weight),
                brivi_lock_weight=float(brivi_lock_weight),
                frame_necessity_weight=float(frame_necessity_weight),
                mdl_weight=float(mdl_weight),
                necessity_margin=float(necessity_margin),
                pointer_necessity_weight=float(pointer_necessity_weight),
                pointer_necessity_margin=float(pointer_necessity_margin),
                hyperbolic_topology_weight=0.0,
            )
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError(f"M23 loss became non-finite at epoch {epoch + 1}, batch {batches + 1}.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm).detach().cpu().item()):
                raise FloatingPointError(f"M23 gradient norm became non-finite at epoch {epoch + 1}, batch {batches + 1}.")
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            totals["grad_norm"] += float(grad_norm.detach().cpu().item())
            for key, value in pieces.items():
                totals[key] += value
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    metrics = evaluate_m23_model(
        model,
        eval_examples,
        vocab,
        batch_size=int(batch_size),
        device=device,
        use_relevance_router=bool(use_relevance_router),
    )
    metrics["use_relevance_router"] = float(1.0 if use_relevance_router else 0.0)
    metrics["relevance_rank_weight"] = float(relevance_rank_weight)
    metrics["trace_exact_surrogate_weight"] = float(trace_exact_surrogate_weight)
    metrics["clean_train_fraction"] = float(clean_train_fraction)
    return {
        "model": model,
        "vocab": vocab,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "history": history,
        "metrics": metrics,
        "config": {
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "max_frames": int(max_frames),
            "max_places": int(max_places),
            "max_entities": int(max_entities),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "counterfactual_weight": float(counterfactual_weight),
            "brivi_lock_weight": float(brivi_lock_weight),
            "frame_necessity_weight": float(frame_necessity_weight),
            "mdl_weight": float(mdl_weight),
            "necessity_margin": float(necessity_margin),
            "pointer_necessity_weight": float(pointer_necessity_weight),
            "pointer_necessity_margin": float(pointer_necessity_margin),
            "relevance_rank_weight": float(relevance_rank_weight),
            "relevance_margin": float(relevance_margin),
            "trace_exact_surrogate_weight": float(trace_exact_surrogate_weight),
            "use_relevance_router": bool(use_relevance_router),
            "relevance_temperature": float(relevance_temperature),
            "clean_train_fraction": float(clean_train_fraction),
            "clean_eval_fraction": float(clean_eval_fraction),
            "geometry_mode": str(geometry_mode),
            "poincare_curvature": float(poincare_curvature),
            "poincare_max_norm": float(poincare_max_norm),
            "riemannian_gradient_scale": bool(riemannian_gradient_scale),
            "judri_bridge_gate": bool(judri_bridge_gate),
            "judri_bridge_gate_temperature": float(judri_bridge_gate_temperature),
        },
    }
