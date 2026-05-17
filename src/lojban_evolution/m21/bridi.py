from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lojban_evolution.m19.typed_physics import DEFAULT_POINCARE_EPS, expmap0, logmap0, poincare_distance


M21_LOCKS: dict[str, str] = {
    "dynamic_frame_count": "learn how many bridi frames are needed instead of using a fixed typed-slot shell",
    "bridi_trace_reconstruction": "reconstruct gismu, cmavo, judri bindings, and STOP from controlled traces",
    "cmavo_causality": "cmavo modifiers must be causally useful rather than decorative",
    "judri_binding_causality": "judri/entity-place bindings must carry downstream answer information",
    "judri_gated_bridge": "downstream predicate energy is silenced unless judri bindings ground the frame",
    "brivi_lock": "a predicate frame is silent unless it binds at least one judri argument",
    "actual_bridge_transfer": "dynamic traces transfer through a minimal downstream bridge-style adapter",
}

GISMU = [
    "size",
    "weight",
    "transfer",
    "preference",
    "containment",
    "visibility",
    "quantity",
    "motion",
    "ownership",
    "comparison",
    "causal",
    "permission",
]
CMAVO = [
    "causal",
    "negation",
    "excess",
    "deficit",
    "positive",
    "negative",
    "quantifier",
    "comparison",
    "conditional",
    "temporal",
    "refusal",
    "permission",
]
ANSWER_LABELS = [
    "size_excess",
    "size_deficit",
    "weight_excess",
    "weight_deficit",
    "transfer_success",
    "transfer_refused",
    "preference_like",
    "preference_dislike",
    "containment_success",
    "containment_blocked",
    "visibility_clear",
    "visibility_blocked",
    "quantity_excess",
    "quantity_deficit",
    "motion_allowed",
    "motion_blocked",
    "permission_granted",
    "permission_denied",
]
GISMU_TO_ID = {name: idx for idx, name in enumerate(GISMU)}
CMAVO_TO_ID = {name: idx for idx, name in enumerate(CMAVO)}
ANSWER_TO_ID = {name: idx for idx, name in enumerate(ANSWER_LABELS)}

DEFAULT_MAX_FRAMES = 6
DEFAULT_MAX_CMAVO_PER_FRAME = 3
DEFAULT_MAX_PLACES = 5
DEFAULT_MAX_ENTITIES = 8
DEFAULT_TOTAL_DICTIONARY_SIZE = 2000
DEFAULT_POINCARE_MAX_NORM = 0.99

OBJECTS = ("stone", "apple", "coin", "book", "vase", "tool", "block", "toy", "package", "bottle", "tablet", "rope")
CONTAINERS = ("box", "drawer", "bag", "crate", "basket", "case", "shelf", "locker")
PEOPLE = ("Alex", "Riley", "Jordan", "Morgan", "Taylor", "Casey", "Sam", "Quinn")
ALT_NOUNS = ("lantern", "marble", "folder", "button", "ticket", "cup", "shell", "cable", "key", "map")
TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def pointer_necessity_contrast_loss(
    full_loss: torch.Tensor,
    ablated_loss: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """M19 hinge: full trace must beat the no-pointer ablation by a margin."""

    return torch.relu(full_loss + full_loss.new_tensor(float(margin)) - ablated_loss)


def judri_grounding_gate_from_logits(
    judri_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    pad_entity_id: int = 0,
) -> torch.Tensor:
    """Return per-frame non-PAD judri mass used to gate downstream predicate energy."""

    temp = max(float(temperature), 1e-6)
    probs = torch.softmax(judri_logits / temp, dim=-1)
    pad_mass = probs[..., int(pad_entity_id)].clamp(0.0, 1.0)
    non_pad_mass = 1.0 - pad_mass
    return non_pad_mass.mean(dim=-1).clamp(0.0, 1.0)


def clamp_poincare_norm(
    x: torch.Tensor,
    *,
    curvature: float = 1.0,
    max_norm: float = DEFAULT_POINCARE_MAX_NORM,
    eps: float = DEFAULT_POINCARE_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict edge guard for the unit Poincare ball used by the M21 hyperbolic branch."""

    c = max(float(curvature), eps)
    radius = float(max_norm) / (c**0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(radius / norm, max=1.0)
    clipped = (scale < 0.999999).to(dtype=x.dtype)
    return x * scale, clipped.squeeze(-1)


def poincare_tangent_handoff(
    y: torch.Tensor,
    *,
    curvature: float = 1.0,
    max_norm: float = DEFAULT_POINCARE_MAX_NORM,
    eps: float = DEFAULT_POINCARE_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate a Poincare point back to the Euclidean tangent plane at the origin."""

    guarded, clip_mask = clamp_poincare_norm(y, curvature=curvature, max_norm=max_norm, eps=eps)
    return logmap0(guarded, curvature=curvature, eps=eps), clip_mask


def _apply_riemannian_gradient_scale(
    x: torch.Tensor,
    *,
    curvature: float,
    eps: float = DEFAULT_POINCARE_EPS,
) -> torch.Tensor:
    if not x.requires_grad:
        return x
    c = max(float(curvature), eps)
    norm_sq = (x.detach() * x.detach()).sum(dim=-1, keepdim=True)
    factor = ((1.0 - c * norm_sq).clamp_min(eps) ** 2) / 4.0

    def _scale_grad(grad: torch.Tensor, scale: torch.Tensor = factor) -> torch.Tensor:
        return grad * scale

    x.register_hook(_scale_grad)
    return x


@dataclass(frozen=True)
class BridiFrame:
    active: bool
    gismu_id: int
    cmavo_ids: tuple[int, ...]
    judri_place_bindings: tuple[int, ...]
    stop: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicBridiExample:
    prompt: str
    frames: tuple[BridiFrame, ...]
    entities: tuple[str, ...]
    answer_id: int
    answer_label: str
    surface: str
    counterfactual_group: str
    entity_signature: str
    is_floating: bool = False

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_json() for frame in self.frames]
        return payload


def _frame(gismu: str, cmavo: Sequence[str], places: Sequence[int], *, stop: bool = False) -> BridiFrame:
    padded = tuple(int(value) for value in list(places)[:DEFAULT_MAX_PLACES])
    padded = padded + tuple(0 for _ in range(DEFAULT_MAX_PLACES - len(padded)))
    return BridiFrame(
        active=True,
        gismu_id=GISMU_TO_ID[gismu],
        cmavo_ids=tuple(CMAVO_TO_ID[item] for item in cmavo[:DEFAULT_MAX_CMAVO_PER_FRAME]),
        judri_place_bindings=padded,
        stop=bool(stop),
    )


def _inactive_stop() -> BridiFrame:
    return BridiFrame(False, 0, tuple(), tuple(0 for _ in range(DEFAULT_MAX_PLACES)), True)


def _values(rng: random.Random, surface: str) -> dict[str, str]:
    values = {
        "object": rng.choice(OBJECTS),
        "object2": rng.choice(OBJECTS),
        "container": rng.choice(CONTAINERS),
        "person": rng.choice(PEOPLE),
        "giver": rng.choice(PEOPLE),
        "receiver": rng.choice(PEOPLE),
        "observer": rng.choice(PEOPLE),
        "count": str(rng.choice((1, 2, 3, 8, 11, 17, 23))),
    }
    if values["receiver"] == values["giver"]:
        values["receiver"] = rng.choice([name for name in PEOPLE if name != values["giver"]])
    if surface == "renamed":
        values.update(
            {
                "object": rng.choice(ALT_NOUNS),
                "object2": rng.choice(ALT_NOUNS),
                "container": f"place_{rng.randrange(10, 99)}",
                "person": f"Person{rng.randrange(10, 99)}",
                "giver": f"Agent{rng.randrange(10, 99)}",
                "receiver": f"Agent{rng.randrange(100, 199)}",
                "observer": f"Viewer{rng.randrange(10, 99)}",
            }
        )
    elif surface == "anonymized":
        values.update(
            {
                "object": "entity_a",
                "object2": "entity_g",
                "container": "entity_b",
                "person": "entity_c",
                "giver": "entity_d",
                "receiver": "entity_e",
                "observer": "entity_f",
            }
        )
    elif surface == "numeric":
        values["object"] = f"object_{rng.randrange(100, 999)}"
        values["object2"] = f"object_{rng.randrange(1000, 1999)}"
        values["container"] = f"container_{rng.randrange(100, 999)}"
    return values


def _entity_tuple(values: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        [
            values.get("object", ""),
            values.get("container", ""),
            values.get("person", ""),
            values.get("giver", ""),
            values.get("receiver", ""),
            values.get("observer", ""),
            values.get("object2", ""),
            values.get("count", ""),
        ][:DEFAULT_MAX_ENTITIES]
    )


def _format(text: str, values: dict[str, str], surface: str) -> str:
    out = text.format(**values)
    if surface == "flattened":
        out = out.replace(" because ", " ; cause: ").replace(" after ", " ; after: ").replace(".", "")
    if surface == "numeric":
        out = f"case {values['count']}: {out}"
    return out


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "size_excess",
            "templates": (
                "The {object} did not fit in the {container} because the {object} was too big.",
                "{object} failed to enter {container}; the reason was excessive size.",
            ),
            "frames": lambda v: [_frame("containment", ["negation", "causal"], [1, 2]), _frame("size", ["excess", "causal"], [1], stop=True)],
        },
        {
            "name": "size_deficit",
            "templates": (
                "The {object} did not fill the {container} because the {object} was too small.",
                "{object} rattled inside {container}; the reason was insufficient size.",
            ),
            "frames": lambda v: [_frame("containment", ["negative"], [2, 1]), _frame("size", ["deficit", "causal"], [1], stop=True)],
        },
        {
            "name": "weight_excess",
            "templates": (
                "{person} could not lift the {object} because the {object} was too heavy.",
                "The {object} stayed on the floor; its weight exceeded {person}'s lift.",
            ),
            "frames": lambda v: [_frame("motion", ["negation", "causal"], [3, 1]), _frame("weight", ["excess"], [1], stop=True)],
        },
        {
            "name": "weight_deficit",
            "templates": (
                "The wind moved the {object} because the {object} was too light.",
                "{person} carried the {object} easily because it had little weight.",
            ),
            "frames": lambda v: [_frame("motion", ["positive", "causal"], [3, 1]), _frame("weight", ["deficit"], [1], stop=True)],
        },
        {
            "name": "transfer_success",
            "templates": (
                "{giver} gave the {object} to {receiver} after the meeting.",
                "The {object} moved from {giver} to {receiver} by a gift event.",
            ),
            "frames": lambda v: [_frame("transfer", ["positive", "temporal"], [4, 5, 1]), _frame("ownership", ["positive"], [5, 1], stop=True)],
        },
        {
            "name": "transfer_refused",
            "templates": (
                "{giver} offered the {object} to {receiver}, but {receiver} refused it.",
                "The proposed transfer of {object} from {giver} to {receiver} was rejected.",
            ),
            "frames": lambda v: [_frame("transfer", ["negation", "refusal"], [4, 5, 1]), _frame("preference", ["negative", "causal"], [5, 1], stop=True)],
        },
        {
            "name": "preference_like",
            "templates": (
                "{person} kept the {object} because {person} liked using it.",
                "{person}'s preference toward the {object} was favorable.",
            ),
            "frames": lambda v: [_frame("preference", ["positive", "causal"], [3, 1], stop=True)],
        },
        {
            "name": "preference_dislike",
            "templates": (
                "{person} avoided the {object} because {person} disliked using it.",
                "{person}'s preference toward the {object} was unfavorable.",
            ),
            "frames": lambda v: [_frame("preference", ["negative", "causal"], [3, 1], stop=True)],
        },
        {
            "name": "containment_success",
            "templates": (
                "The {container} held the {object} because the opening accepted it.",
                "{object} remained inside {container} after placement.",
            ),
            "frames": lambda v: [_frame("containment", ["positive", "causal"], [2, 1], stop=True)],
        },
        {
            "name": "containment_blocked",
            "templates": (
                "The {container} blocked the {object} because the opening was narrow.",
                "{object} could not pass through {container}'s narrow opening.",
            ),
            "frames": lambda v: [_frame("containment", ["negative", "causal"], [2, 1]), _frame("size", ["excess"], [1], stop=True)],
        },
        {
            "name": "visibility_clear",
            "templates": (
                "{observer} saw the {object} because the path was clear.",
                "Nothing occluded {object} from {observer}.",
            ),
            "frames": lambda v: [_frame("visibility", ["positive"], [6, 1], stop=True)],
        },
        {
            "name": "visibility_blocked",
            "templates": (
                "{observer} could not see the {object} because the screen hid it.",
                "The screen occluded {object} from {observer}.",
            ),
            "frames": lambda v: [_frame("visibility", ["negation", "causal"], [6, 1], stop=True)],
        },
        {
            "name": "quantity_excess",
            "templates": (
                "There were {count} {object}s, so the shelf had too many items.",
                "The count of {object}s exceeded the allowed amount.",
            ),
            "frames": lambda v: [_frame("quantity", ["quantifier", "excess"], [8, 1], stop=True)],
        },
        {
            "name": "quantity_deficit",
            "templates": (
                "There were only {count} {object}s, so the shelf had too few items.",
                "The count of {object}s fell below the required amount.",
            ),
            "frames": lambda v: [_frame("quantity", ["quantifier", "deficit"], [8, 1], stop=True)],
        },
        {
            "name": "motion_allowed",
            "templates": (
                "If {person} had permission, {person} moved the {object} into the {container}.",
                "Permission allowed {person} to move {object} toward {container}.",
            ),
            "frames": lambda v: [_frame("permission", ["permission", "positive"], [3]), _frame("motion", ["conditional", "positive"], [3, 1, 2], stop=True)],
        },
        {
            "name": "motion_blocked",
            "templates": (
                "If {person} lacked permission, {person} could not move the {object} into the {container}.",
                "Permission was denied, so {person} left {object} outside {container}.",
            ),
            "frames": lambda v: [_frame("permission", ["permission", "negation"], [3]), _frame("motion", ["conditional", "negation"], [3, 1, 2], stop=True)],
        },
        {
            "name": "permission_granted",
            "templates": ("The rule allowed {person} to use the {object}.", "{person} received permission for the {object}."),
            "frames": lambda v: [_frame("permission", ["permission", "positive"], [3, 1], stop=True)],
        },
        {
            "name": "permission_denied",
            "templates": ("The rule forbade {person} from using the {object}.", "{person} was denied permission for the {object}."),
            "frames": lambda v: [_frame("permission", ["permission", "negation"], [3, 1], stop=True)],
        },
    ]


def generate_dynamic_bridi_examples(
    size: int,
    *,
    seed: int = 0,
    floating_fraction: float = 0.12,
    surfaces: Sequence[str] = ("purged", "flattened", "renamed", "anonymized", "numeric"),
) -> list[DynamicBridiExample]:
    rng = random.Random(int(seed))
    specs = _variant_specs()
    rows: list[DynamicBridiExample] = []
    for idx in range(int(size)):
        spec = specs[idx % len(specs)] if idx < len(specs) * 2 else rng.choice(specs)
        is_floating = rng.random() < float(floating_fraction)
        surface = "floating" if is_floating else rng.choice(tuple(surfaces))
        values = _values(rng, surface)
        entities = _entity_tuple(values)
        if is_floating:
            prompt = f"floating predicate phrase: {rng.choice(GISMU)} with {rng.choice(CMAVO)}, no arguments supplied"
            frames = (_inactive_stop(),)
            answer_label = rng.choice(ANSWER_LABELS)
        else:
            prompt = _format(rng.choice(spec["templates"]), values, surface)
            frames = tuple(spec["frames"](values))
            answer_label = str(spec["name"])
        rows.append(
            DynamicBridiExample(
                prompt=prompt,
                frames=frames,
                entities=entities,
                answer_id=ANSWER_TO_ID[answer_label],
                answer_label=answer_label,
                surface=surface,
                counterfactual_group=answer_label,
                entity_signature="|".join(entities),
                is_floating=is_floating,
            )
        )
    rng.shuffle(rows)
    return rows


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_vocab(examples: Iterable[DynamicBridiExample], *, min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in examples:
        for token in tokenize(row.prompt):
            counts[token] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= int(min_count):
            vocab[token] = len(vocab)
    return vocab


class M21BridiDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[DynamicBridiExample], vocab: dict[str, int], *, max_length: int = 64, max_frames: int = DEFAULT_MAX_FRAMES):
        self.examples = list(examples)
        self.vocab = dict(vocab)
        self.max_length = int(max_length)
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, text: str) -> list[int]:
        ids = [self.vocab.get(token, 1) for token in tokenize(text)][: self.max_length]
        ids.extend([0] * max(0, self.max_length - len(ids)))
        return ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        active = torch.zeros(self.max_frames, dtype=torch.float32)
        stop = torch.zeros(self.max_frames, dtype=torch.float32)
        gismu = torch.zeros(self.max_frames, dtype=torch.long)
        cmavo = torch.zeros(self.max_frames, len(CMAVO), dtype=torch.float32)
        judri = torch.zeros(self.max_frames, DEFAULT_MAX_PLACES, dtype=torch.long)
        for frame_idx, frame in enumerate(row.frames[: self.max_frames]):
            active[frame_idx] = 1.0 if frame.active else 0.0
            stop[frame_idx] = 1.0 if frame.stop else 0.0
            gismu[frame_idx] = int(frame.gismu_id)
            for cmavo_id in frame.cmavo_ids:
                if 0 <= int(cmavo_id) < len(CMAVO):
                    cmavo[frame_idx, int(cmavo_id)] = 1.0
            for place_idx, binding in enumerate(frame.judri_place_bindings[:DEFAULT_MAX_PLACES]):
                judri[frame_idx, place_idx] = max(0, min(DEFAULT_MAX_ENTITIES, int(binding)))
        if float(stop.sum().item()) <= 0.0:
            stop[min(max(1, len(row.frames)), self.max_frames) - 1] = 1.0
        return {
            "input_ids": torch.tensor(self._encode(row.prompt), dtype=torch.long),
            "active_targets": active,
            "stop_targets": stop,
            "gismu_targets": gismu,
            "cmavo_targets": cmavo,
            "judri_targets": judri,
            "answer_id": torch.tensor(row.answer_id, dtype=torch.long),
            "surface": row.surface,
            "counterfactual_group": row.counterfactual_group,
            "entity_signature": row.entity_signature,
            "prompt": row.prompt,
        }


def m21_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("input_ids", "active_targets", "stop_targets", "gismu_targets", "cmavo_targets", "judri_targets", "answer_id")
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ("surface", "counterfactual_group", "entity_signature", "prompt"):
        out[key] = [str(item[key]) for item in batch]
    return out


class M21DynamicBridiQFormer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        max_frames: int = DEFAULT_MAX_FRAMES,
        max_places: int = DEFAULT_MAX_PLACES,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        geometry_mode: str = "euclidean",
        poincare_curvature: float = 1.0,
        poincare_max_norm: float = DEFAULT_POINCARE_MAX_NORM,
        riemannian_gradient_scale: bool = True,
        judri_bridge_gate: bool = False,
        judri_bridge_gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.max_frames = int(max_frames)
        self.max_places = int(max_places)
        self.max_entities = int(max_entities)
        self.geometry_mode = str(geometry_mode).strip().lower()
        self.poincare_curvature = float(poincare_curvature)
        self.poincare_max_norm = float(poincare_max_norm)
        self.riemannian_gradient_scale = bool(riemannian_gradient_scale)
        self.judri_bridge_gate = bool(judri_bridge_gate)
        self.judri_bridge_gate_temperature = float(judri_bridge_gate_temperature)
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(int(embedding_dim), int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh())
        self.frame_queries = nn.Parameter(torch.randn(self.max_frames, int(hidden_dim)) * 0.02)
        self.frame_mlp = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh())
        self.active_head = nn.Linear(int(hidden_dim), 1)
        self.stop_head = nn.Linear(int(hidden_dim), 1)
        self.gismu_head = nn.Linear(int(hidden_dim), len(GISMU))
        self.cmavo_head = nn.Linear(int(hidden_dim), len(CMAVO))
        self.judri_head = nn.Linear(int(hidden_dim), self.max_places * (self.max_entities + 1))
        self.gismu_embed = nn.Parameter(torch.randn(len(GISMU), int(hidden_dim)) * 0.02)
        self.cmavo_embed = nn.Parameter(torch.randn(len(CMAVO), int(hidden_dim)) * 0.02)
        self.entity_embed = nn.Parameter(torch.randn(self.max_entities + 1, int(hidden_dim)) * 0.02)
        self.answer_head = nn.Linear(int(hidden_dim), len(ANSWER_LABELS))
        self.scratchpad_answer_head = nn.Linear(int(hidden_dim), len(ANSWER_LABELS))

    def _to_poincare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = expmap0(x, curvature=self.poincare_curvature)
        guarded, clip_mask = clamp_poincare_norm(
            projected,
            curvature=self.poincare_curvature,
            max_norm=self.poincare_max_norm,
        )
        if self.riemannian_gradient_scale:
            guarded = _apply_riemannian_gradient_scale(guarded, curvature=self.poincare_curvature)
        return guarded, clip_mask

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(input_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        prompt_state = self.encoder(pooled)
        prompt_routing = prompt_state.unsqueeze(1)
        query_routing = self.frame_queries.unsqueeze(0).expand(prompt_state.shape[0], -1, -1)
        routing_state = prompt_routing + query_routing
        clip_masks: list[torch.Tensor] = []
        norm_tensors: list[torch.Tensor] = []
        if self.geometry_mode == "poincare":
            prompt_key, prompt_clip = self._to_poincare(prompt_routing.expand_as(query_routing))
            query_key, query_clip = self._to_poincare(query_routing)
            routing_state, routing_clip = clamp_poincare_norm(
                prompt_key + query_key,
                curvature=self.poincare_curvature,
                max_norm=self.poincare_max_norm,
            )
            if self.riemannian_gradient_scale:
                routing_state = _apply_riemannian_gradient_scale(routing_state, curvature=self.poincare_curvature)
            clip_masks.extend([prompt_clip.float(), query_clip.float(), routing_clip.float()])
            norm_tensors.extend([prompt_key.norm(dim=-1), query_key.norm(dim=-1), routing_state.norm(dim=-1)])
        frame_state = self.frame_mlp(routing_state)
        if self.geometry_mode == "poincare":
            frame_state, frame_clip = self._to_poincare(frame_state)
            clip_masks.append(frame_clip.float())
            norm_tensors.append(frame_state.norm(dim=-1))
        active_logits = self.active_head(frame_state).squeeze(-1)
        stop_logits = self.stop_head(frame_state).squeeze(-1)
        gismu_logits = self.gismu_head(frame_state)
        cmavo_logits = self.cmavo_head(frame_state)
        judri_logits = self.judri_head(frame_state).view(input_ids.shape[0], self.max_frames, self.max_places, self.max_entities + 1)
        active_prob = torch.sigmoid(active_logits).unsqueeze(-1)
        gismu_state = torch.softmax(gismu_logits, dim=-1) @ self.gismu_embed
        cmavo_state = torch.sigmoid(cmavo_logits) @ self.cmavo_embed
        judri_state = (torch.softmax(judri_logits, dim=-1) @ self.entity_embed).mean(dim=2)
        predicate_state = gismu_state + cmavo_state
        judri_gate = judri_grounding_gate_from_logits(
            judri_logits,
            temperature=self.judri_bridge_gate_temperature,
        )
        judri_gate_expanded = judri_gate.unsqueeze(-1)
        if self.judri_bridge_gate:
            gated_predicate_state = predicate_state * judri_gate_expanded
            full_component = gated_predicate_state + judri_state
            no_cmavo_component = (gismu_state * judri_gate_expanded) + judri_state
            no_judri_component = predicate_state * 0.0
            gismu_only_component = gismu_state * 0.0
            silenced_predicate_energy = (predicate_state.norm(dim=-1) * (1.0 - judri_gate) * active_prob.squeeze(-1)).sum() / active_prob.squeeze(-1).sum().clamp_min(1.0)
        else:
            full_component = predicate_state + judri_state
            no_cmavo_component = gismu_state + judri_state
            no_judri_component = predicate_state
            gismu_only_component = gismu_state
            silenced_predicate_energy = prompt_state.new_zeros(())
        if self.geometry_mode == "poincare":
            full_component, full_clip = self._to_poincare(full_component)
            no_cmavo_component, no_cmavo_clip = self._to_poincare(no_cmavo_component)
            no_judri_component, no_judri_clip = self._to_poincare(no_judri_component)
            gismu_only_component, gismu_clip = self._to_poincare(gismu_only_component)
            clip_masks.extend([full_clip.float(), no_cmavo_clip.float(), no_judri_clip.float(), gismu_clip.float()])
            norm_tensors.extend(
                [
                    full_component.norm(dim=-1),
                    no_cmavo_component.norm(dim=-1),
                    no_judri_component.norm(dim=-1),
                    gismu_only_component.norm(dim=-1),
                ]
            )
        frame_repr = full_component * active_prob
        trace_state = frame_repr.sum(dim=1)
        no_cmavo_state = (no_cmavo_component * active_prob).sum(dim=1)
        no_judri_state = (no_judri_component * active_prob).sum(dim=1)
        gismu_only_state = (gismu_only_component * active_prob).sum(dim=1)
        frame_drop_state = frame_repr[:, :1, :].sum(dim=1) if self.max_frames > 1 else trace_state * 0.0
        zero = prompt_state.new_zeros(())
        hyperbolic_topology_loss = zero
        hyperbolic_projection_clip_rate = zero
        hyperbolic_max_norm = zero
        hyperbolic_distance_mean = zero
        hyperbolic_tangent_handoff_norm_mean = zero
        hyperbolic_tangent_handoff_finite_rate = zero
        answer_trace_state = trace_state
        answer_no_cmavo_state = no_cmavo_state
        answer_no_judri_state = no_judri_state
        answer_gismu_only_state = gismu_only_state
        answer_frame_drop_state = frame_drop_state
        if self.geometry_mode == "poincare":
            prompt_hyp, prompt_clip = self._to_poincare(prompt_state)
            clip_masks.append(prompt_clip.float())
            norm_tensors.append(prompt_hyp.norm(dim=-1))
            prompt_frames = prompt_hyp.unsqueeze(1).expand_as(frame_state)
            distances = poincare_distance(frame_state, prompt_frames, curvature=self.poincare_curvature)
            active = active_prob.squeeze(-1)
            hyperbolic_topology_loss = (distances * active).sum() / active.sum().clamp_min(1.0)
            hyperbolic_distance_mean = distances.mean()
            hyperbolic_projection_clip_rate = torch.stack([item.float().mean() for item in clip_masks]).mean() if clip_masks else zero
            hyperbolic_max_norm = torch.stack([item.max() for item in norm_tensors]).max() if norm_tensors else zero
            handoff_states = [trace_state, no_cmavo_state, no_judri_state, gismu_only_state, frame_drop_state]
            tangent_states: list[torch.Tensor] = []
            handoff_clips: list[torch.Tensor] = []
            for state in handoff_states:
                tangent, handoff_clip = poincare_tangent_handoff(
                    state,
                    curvature=self.poincare_curvature,
                    max_norm=self.poincare_max_norm,
                )
                tangent_states.append(tangent)
                handoff_clips.append(handoff_clip.float())
            (
                answer_trace_state,
                answer_no_cmavo_state,
                answer_no_judri_state,
                answer_gismu_only_state,
                answer_frame_drop_state,
            ) = tangent_states
            tangent_norms = torch.stack([state.norm(dim=-1).mean() for state in tangent_states])
            tangent_finite = torch.stack([torch.isfinite(state).float().mean() for state in tangent_states])
            hyperbolic_tangent_handoff_norm_mean = tangent_norms.mean()
            hyperbolic_tangent_handoff_finite_rate = tangent_finite.mean()
            clip_masks.extend(handoff_clips)
        return {
            "prompt_state": prompt_state,
            "active_logits": active_logits,
            "stop_logits": stop_logits,
            "gismu_logits": gismu_logits,
            "cmavo_logits": cmavo_logits,
            "judri_logits": judri_logits,
            "active_prob": active_prob.squeeze(-1),
            "answer_logits": self.answer_head(answer_trace_state),
            "no_cmavo_answer_logits": self.answer_head(answer_no_cmavo_state),
            "no_judri_answer_logits": self.answer_head(answer_no_judri_state),
            "gismu_only_answer_logits": self.answer_head(answer_gismu_only_state),
            "frame_drop_answer_logits": self.answer_head(answer_frame_drop_state),
            "scratchpad_answer_logits": self.scratchpad_answer_head(prompt_state.detach()),
            "hyperbolic_topology_loss": hyperbolic_topology_loss,
            "hyperbolic_projection_clip_rate": hyperbolic_projection_clip_rate,
            "hyperbolic_max_norm": hyperbolic_max_norm,
            "hyperbolic_distance_mean": hyperbolic_distance_mean,
            "hyperbolic_tangent_handoff_norm_mean": hyperbolic_tangent_handoff_norm_mean,
            "hyperbolic_tangent_handoff_finite_rate": hyperbolic_tangent_handoff_finite_rate,
            "judri_bridge_gate_mean": judri_gate.mean(),
            "judri_bridge_gate_active_mean": (judri_gate * active_prob.squeeze(-1)).sum() / active_prob.squeeze(-1).sum().clamp_min(1.0),
            "judri_bridge_gate_silenced_predicate_energy_mean": silenced_predicate_energy,
            "judri_bridge_gate_enabled": prompt_state.new_tensor(1.0 if self.judri_bridge_gate else 0.0),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim > mask.ndim:
        mask = mask.unsqueeze(-1).expand_as(values)
    return (values * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _ce_per_position(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets).float()


def counterfactual_trace_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    active = batch["active_targets"].to(outputs["gismu_logits"].device) > 0.5
    trace_probs = torch.cat([torch.softmax(outputs["gismu_logits"], dim=-1), torch.sigmoid(outputs["cmavo_logits"])], dim=-1)
    losses: list[torch.Tensor] = []
    groups = list(batch.get("counterfactual_group", []))
    entities = list(batch.get("entity_signature", []))
    for group in sorted(set(groups)):
        indices = [idx for idx, value in enumerate(groups) if value == group]
        if len(indices) < 2 or len({entities[idx] for idx in indices}) < 2:
            continue
        rows = trace_probs[indices]
        masks = active[indices].unsqueeze(-1).float()
        center = (rows * masks).sum(dim=0, keepdim=True) / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
        losses.append(_masked_mean((rows - center) ** 2, masks.bool()))
    return torch.stack(losses).mean() if losses else outputs["gismu_logits"].new_zeros(())


def compute_m21_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    trace_weight: float = 1.0,
    answer_weight: float = 1.0,
    counterfactual_weight: float = 1.0,
    brivi_lock_weight: float = 1.0,
    frame_necessity_weight: float = 0.5,
    mdl_weight: float = 0.01,
    necessity_margin: float = 0.04,
    pointer_necessity_weight: float = 0.0,
    pointer_necessity_margin: float | None = None,
    hyperbolic_topology_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["active_logits"].device
    active_targets = batch["active_targets"].to(device)
    active_mask = active_targets > 0.5
    gismu_targets = batch["gismu_targets"].to(device)
    cmavo_targets = batch["cmavo_targets"].to(device)
    judri_targets = batch["judri_targets"].to(device)
    answer_id = batch["answer_id"].to(device)
    active_positive = active_targets.sum().clamp_min(1.0)
    active_negative = (1.0 - active_targets).sum().clamp_min(1.0)
    active_pos_weight = (active_negative / active_positive).detach()
    active_loss = F.binary_cross_entropy_with_logits(outputs["active_logits"], active_targets, pos_weight=active_pos_weight)
    stop_loss = F.binary_cross_entropy_with_logits(outputs["stop_logits"], batch["stop_targets"].to(device))
    gismu_loss = _masked_mean(_ce_per_position(outputs["gismu_logits"], gismu_targets), active_mask)
    cmavo_loss = _masked_mean(F.binary_cross_entropy_with_logits(outputs["cmavo_logits"], cmavo_targets, reduction="none").mean(dim=-1), active_mask)
    judri_loss = _masked_mean(_ce_per_position(outputs["judri_logits"], judri_targets).mean(dim=-1), active_mask)
    trace_loss = active_loss + stop_loss + gismu_loss + cmavo_loss + judri_loss
    answer_loss = F.cross_entropy(outputs["answer_logits"], answer_id)
    no_cmavo_loss = F.cross_entropy(outputs["no_cmavo_answer_logits"], answer_id)
    no_judri_loss = F.cross_entropy(outputs["no_judri_answer_logits"], answer_id)
    frame_drop_loss = F.cross_entropy(outputs["frame_drop_answer_logits"], answer_id)
    margin = torch.tensor(float(necessity_margin), device=device)
    frame_necessity = torch.relu(margin + answer_loss - no_cmavo_loss) + torch.relu(margin + answer_loss - no_judri_loss) + torch.relu(margin + answer_loss - frame_drop_loss)
    pointer_margin = float(necessity_margin if pointer_necessity_margin is None else pointer_necessity_margin)
    pointer_necessity = pointer_necessity_contrast_loss(answer_loss, no_judri_loss, pointer_margin)
    counter_loss = counterfactual_trace_loss(outputs, batch)
    has_any_binding = (judri_targets > 0).any(dim=-1)
    brivi_lock_loss = (outputs["active_prob"] * (~has_any_binding).float()).mean()
    mdl_loss = outputs["active_prob"].mean()
    hyperbolic_topology_loss = outputs.get("hyperbolic_topology_loss", answer_loss.new_zeros(()))
    total = (
        float(trace_weight) * trace_loss
        + float(answer_weight) * answer_loss
        + float(counterfactual_weight) * counter_loss
        + float(brivi_lock_weight) * brivi_lock_loss
        + float(frame_necessity_weight) * frame_necessity
        + float(pointer_necessity_weight) * pointer_necessity
        + float(hyperbolic_topology_weight) * hyperbolic_topology_loss
        + float(mdl_weight) * mdl_loss
    )
    pointer_gap = no_judri_loss.detach() - answer_loss.detach()
    return total, {
        "loss_trace": float(trace_loss.detach().cpu().item()),
        "loss_active": float(active_loss.detach().cpu().item()),
        "loss_stop": float(stop_loss.detach().cpu().item()),
        "loss_gismu": float(gismu_loss.detach().cpu().item()),
        "loss_cmavo": float(cmavo_loss.detach().cpu().item()),
        "loss_judri": float(judri_loss.detach().cpu().item()),
        "loss_answer": float(answer_loss.detach().cpu().item()),
        "loss_counterfactual": float(counter_loss.detach().cpu().item()),
        "loss_brivi_lock": float(brivi_lock_loss.detach().cpu().item()),
        "loss_frame_necessity": float(frame_necessity.detach().cpu().item()),
        "loss_pointer_necessity": float(pointer_necessity.detach().cpu().item()),
        "pointer_necessity_gap": float(pointer_gap.cpu().item()),
        "loss_hyperbolic_topology": float(hyperbolic_topology_loss.detach().cpu().item()),
        "hyperbolic_projection_clip_rate": float(outputs.get("hyperbolic_projection_clip_rate", answer_loss.new_zeros(())).detach().cpu().item()),
        "hyperbolic_max_norm": float(outputs.get("hyperbolic_max_norm", answer_loss.new_zeros(())).detach().cpu().item()),
        "hyperbolic_distance_mean": float(outputs.get("hyperbolic_distance_mean", answer_loss.new_zeros(())).detach().cpu().item()),
        "hyperbolic_tangent_handoff_norm_mean": float(outputs.get("hyperbolic_tangent_handoff_norm_mean", answer_loss.new_zeros(())).detach().cpu().item()),
        "hyperbolic_tangent_handoff_finite_rate": float(outputs.get("hyperbolic_tangent_handoff_finite_rate", answer_loss.new_zeros(())).detach().cpu().item()),
        "judri_bridge_gate_mean": float(outputs.get("judri_bridge_gate_mean", answer_loss.new_zeros(())).detach().cpu().item()),
        "judri_bridge_gate_active_mean": float(outputs.get("judri_bridge_gate_active_mean", answer_loss.new_zeros(())).detach().cpu().item()),
        "judri_bridge_gate_silenced_predicate_energy_mean": float(outputs.get("judri_bridge_gate_silenced_predicate_energy_mean", answer_loss.new_zeros(())).detach().cpu().item()),
        "judri_bridge_gate_enabled": float(outputs.get("judri_bridge_gate_enabled", answer_loss.new_zeros(())).detach().cpu().item()),
        "loss_mdl": float(mdl_loss.detach().cpu().item()),
    }


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((torch.argmax(logits, dim=-1) == target).float().mean().detach().cpu().item())


def _masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return 0.0
    return float((torch.argmax(logits[mask], dim=-1) == target[mask]).float().mean().detach().cpu().item())


def _counter_consistency_metric(gismu_probs: torch.Tensor, cmavo_probs: torch.Tensor, groups: Sequence[str], entities: Sequence[str], active: torch.Tensor) -> float:
    trace = torch.cat([gismu_probs, cmavo_probs], dim=-1)
    values: list[float] = []
    for group in sorted(set(groups)):
        indices = [idx for idx, value in enumerate(groups) if value == group]
        if len(indices) < 2 or len({entities[idx] for idx in indices}) < 2:
            continue
        rows = trace[indices]
        masks = active[indices].unsqueeze(-1).float()
        center = (rows * masks).sum(dim=0, keepdim=True) / masks.sum(dim=0, keepdim=True).clamp_min(1.0)
        values.append(float(_masked_mean((rows - center) ** 2, masks.bool()).detach().cpu().item()))
    return float(max(0.0, 1.0 - (sum(values) / max(1, len(values))) * 100.0))


def _categorical_entropy(values: torch.Tensor, *, max_value: int) -> float:
    counts = torch.bincount(values.round().clamp(0, int(max_value)).to(torch.long), minlength=int(max_value) + 1).float()
    probs = counts / counts.sum().clamp_min(1.0)
    probs = probs[probs > 0]
    return float((-(probs * probs.log()).sum()).item()) if probs.numel() else 0.0


@torch.no_grad()
def evaluate_model(
    model: M21DynamicBridiQFormer,
    examples: Sequence[DynamicBridiExample],
    vocab: dict[str, int],
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    total_dictionary_size: int = DEFAULT_TOTAL_DICTIONARY_SIZE,
) -> dict[str, Any]:
    model.eval()
    dataset = M21BridiDataset(examples, vocab, max_frames=model.max_frames)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m21_collate)
    merged: dict[str, list[torch.Tensor]] = defaultdict(list)
    surfaces: list[str] = []
    groups: list[str] = []
    entities: list[str] = []
    prompts: list[str] = []
    hyper_sums: dict[str, float] = defaultdict(float)
    hyper_batches = 0
    for batch in loader:
        outputs = model(batch["input_ids"].to(device))
        for key in ("active_logits", "stop_logits", "gismu_logits", "cmavo_logits", "judri_logits", "answer_logits", "no_cmavo_answer_logits", "no_judri_answer_logits", "gismu_only_answer_logits", "frame_drop_answer_logits", "scratchpad_answer_logits"):
            merged[key].append(outputs[key].detach().cpu())
        for key in (
            "hyperbolic_projection_clip_rate",
            "hyperbolic_max_norm",
            "hyperbolic_distance_mean",
            "hyperbolic_topology_loss",
            "hyperbolic_tangent_handoff_norm_mean",
            "hyperbolic_tangent_handoff_finite_rate",
            "judri_bridge_gate_mean",
            "judri_bridge_gate_active_mean",
            "judri_bridge_gate_silenced_predicate_energy_mean",
            "judri_bridge_gate_enabled",
        ):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                hyper_sums[key] += float(value.detach().cpu().item())
        hyper_batches += 1
        for key in ("active_targets", "stop_targets", "gismu_targets", "cmavo_targets", "judri_targets", "answer_id"):
            merged[key].append(batch[key].detach().cpu())
        surfaces.extend(batch["surface"])
        groups.extend(batch["counterfactual_group"])
        entities.extend(batch["entity_signature"])
        prompts.extend(batch["prompt"])
    tensors = {key: torch.cat(values, dim=0) for key, values in merged.items()}
    active_target = tensors["active_targets"] > 0.5
    active_pred = torch.sigmoid(tensors["active_logits"]) > 0.5
    stop_pred_idx = torch.argmax(tensors["stop_logits"], dim=-1)
    stop_target_idx = torch.argmax(tensors["stop_targets"], dim=-1)
    gismu_pred = torch.argmax(tensors["gismu_logits"], dim=-1)
    cmavo_pred = torch.sigmoid(tensors["cmavo_logits"]) > 0.5
    cmavo_target = tensors["cmavo_targets"] > 0.5
    judri_pred = torch.argmax(tensors["judri_logits"], dim=-1)
    judri_target = tensors["judri_targets"]
    frame_count_pred = active_pred.sum(dim=-1).float()
    frame_count_target = active_target.sum(dim=-1).float()
    active_exact = (active_pred == active_target).all(dim=-1)
    frame_exact = (~active_target) | ((gismu_pred == tensors["gismu_targets"]) & ((cmavo_pred == cmavo_target).all(dim=-1)) & ((judri_pred == judri_target).all(dim=-1)))
    trace_exact = active_exact & frame_exact.all(dim=-1) & (stop_pred_idx == stop_target_idx)
    has_any_binding_target = (judri_target > 0).any(dim=-1)
    predicted_ungrounded = active_pred & (~has_any_binding_target)
    active_gismu = sorted({int(value) for value in gismu_pred[active_pred].reshape(-1).tolist()}) if bool(active_pred.any().item()) else []
    active_cmavo_ids: set[int] = set()
    if bool(active_pred.any().item()):
        for item in torch.nonzero(cmavo_pred & active_pred.unsqueeze(-1), as_tuple=False):
            active_cmavo_ids.add(int(item[-1].item()))
    full_accuracy = _accuracy(tensors["answer_logits"], tensors["answer_id"])
    no_cmavo_accuracy = _accuracy(tensors["no_cmavo_answer_logits"], tensors["answer_id"])
    no_judri_accuracy = _accuracy(tensors["no_judri_answer_logits"], tensors["answer_id"])
    gismu_only_accuracy = _accuracy(tensors["gismu_only_answer_logits"], tensors["answer_id"])
    frame_drop_accuracy = _accuracy(tensors["frame_drop_answer_logits"], tensors["answer_id"])
    scratchpad_accuracy = _accuracy(tensors["scratchpad_answer_logits"], tensors["answer_id"])
    surface_metrics: dict[str, dict[str, float]] = {}
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([value == surface for value in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {
            "strict_accuracy": float((torch.argmax(tensors["answer_logits"], dim=-1)[mask] == tensors["answer_id"][mask]).float().mean().item()) if bool(mask.any().item()) else 0.0,
            "bridi_trace_exact_accuracy": float(trace_exact[mask].float().mean().item()) if bool(mask.any().item()) else 0.0,
            "count": float(mask.sum().item()),
        }
    avg_tokens = float(sum(len(tokenize(prompt)) for prompt in prompts) / max(1, len(prompts)))
    trace_tokens = float(frame_count_pred.mean().item() * (1 + DEFAULT_MAX_PLACES + len(CMAVO) / 4.0))
    active_code_count = len(active_gismu) + len(active_cmavo_ids)
    consistency = _counter_consistency_metric(torch.softmax(tensors["gismu_logits"], dim=-1), torch.sigmoid(tensors["cmavo_logits"]), groups, entities, active_target)
    return {
        "strict_accuracy": full_accuracy,
        "synthetic_world_accuracy": full_accuracy,
        "bridi_trace_exact_accuracy": float(trace_exact.float().mean().item()),
        "gismu_accuracy": _masked_accuracy(tensors["gismu_logits"], tensors["gismu_targets"], active_target),
        "cmavo_accuracy": float(((cmavo_pred == cmavo_target).all(dim=-1))[active_target].float().mean().item()) if bool(active_target.any().item()) else 0.0,
        "judri_binding_accuracy": float(((judri_pred == judri_target).all(dim=-1))[active_target].float().mean().item()) if bool(active_target.any().item()) else 0.0,
        "frame_count_mae": float((frame_count_pred - frame_count_target).abs().mean().item()),
        "stop_accuracy": float((stop_pred_idx == stop_target_idx).float().mean().item()),
        "brivi_lock_violation_rate": float(predicted_ungrounded.float().mean().item()),
        "brivi_gate_accuracy": float((predicted_ungrounded.sum(dim=-1) == 0).float().mean().item()),
        "counterfactual_quotient_consistency": consistency,
        "entity_leakage_proxy": float(max(0.0, 1.0 - consistency)),
        "mean_active_frames": float(frame_count_pred.mean().item()),
        "frame_count_entropy": _categorical_entropy(frame_count_pred, max_value=model.max_frames),
        "active_gismu_count": float(len(active_gismu)),
        "active_cmavo_count": float(len(active_cmavo_ids)),
        "active_code_fraction_reachable": float(active_code_count / max(1, len(GISMU) + len(CMAVO))),
        "active_code_fraction_total": float(active_code_count / max(1, int(total_dictionary_size))),
        "full_accuracy": full_accuracy,
        "no_cmavo_accuracy": no_cmavo_accuracy,
        "no_judri_accuracy": no_judri_accuracy,
        "gismu_only_accuracy": gismu_only_accuracy,
        "random_trace_accuracy": 1.0 / len(ANSWER_LABELS),
        "scratchpad_only_accuracy": scratchpad_accuracy,
        "frame_drop_accuracy": frame_drop_accuracy,
        "frame_drop_delta": float(full_accuracy - frame_drop_accuracy),
        "cmavo_causal_delta": float(full_accuracy - no_cmavo_accuracy),
        "judri_causal_delta": float(full_accuracy - no_judri_accuracy),
        "avg_tokens": avg_tokens,
        "accuracy_per_token": float(full_accuracy / max(avg_tokens, 1e-8)),
        "trace_tokens": trace_tokens,
        "accuracy_per_trace_token": float(full_accuracy / max(trace_tokens, 1.0)),
        "phrase_accuracy": full_accuracy,
        "surface_metrics": surface_metrics,
        "hyperbolic_projection_clip_rate": hyper_sums["hyperbolic_projection_clip_rate"] / max(1, hyper_batches),
        "hyperbolic_max_norm": hyper_sums["hyperbolic_max_norm"] / max(1, hyper_batches),
        "hyperbolic_distance_mean": hyper_sums["hyperbolic_distance_mean"] / max(1, hyper_batches),
        "hyperbolic_tangent_handoff_norm_mean": hyper_sums["hyperbolic_tangent_handoff_norm_mean"] / max(1, hyper_batches),
        "hyperbolic_tangent_handoff_finite_rate": hyper_sums["hyperbolic_tangent_handoff_finite_rate"] / max(1, hyper_batches),
        "judri_bridge_gate_mean": hyper_sums["judri_bridge_gate_mean"] / max(1, hyper_batches),
        "judri_bridge_gate_active_mean": hyper_sums["judri_bridge_gate_active_mean"] / max(1, hyper_batches),
        "judri_bridge_gate_silenced_predicate_energy_mean": hyper_sums["judri_bridge_gate_silenced_predicate_energy_mean"] / max(1, hyper_batches),
        "judri_bridge_gate_enabled": hyper_sums["judri_bridge_gate_enabled"] / max(1, hyper_batches),
        "loss_hyperbolic_topology": hyper_sums["hyperbolic_topology_loss"] / max(1, hyper_batches),
    }


def train_m21_dynamic_bridi(
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
    max_cmavo_per_frame: int = DEFAULT_MAX_CMAVO_PER_FRAME,
    max_places: int = DEFAULT_MAX_PLACES,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    trace_weight: float = 1.0,
    answer_weight: float = 1.0,
    counterfactual_weight: float = 1.0,
    brivi_lock_weight: float = 1.0,
    frame_necessity_weight: float = 0.5,
    mdl_weight: float = 0.01,
    necessity_margin: float = 0.04,
    pointer_necessity_weight: float = 0.0,
    pointer_necessity_margin: float = 0.05,
    hyperbolic_topology_weight: float = 0.0,
    geometry_mode: str = "euclidean",
    poincare_curvature: float = 1.0,
    poincare_max_norm: float = DEFAULT_POINCARE_MAX_NORM,
    riemannian_gradient_scale: bool = True,
    judri_bridge_gate: bool = False,
    judri_bridge_gate_temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    del max_cmavo_per_frame
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    train_examples = generate_dynamic_bridi_examples(int(train_size), seed=int(seed), floating_fraction=0.12)
    eval_examples = generate_dynamic_bridi_examples(int(eval_size), seed=int(seed) + 10_000, floating_fraction=0.16)
    vocab = build_vocab([*train_examples, *eval_examples])
    dataset = M21BridiDataset(train_examples, vocab, max_frames=int(max_frames))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m21_collate)
    model = M21DynamicBridiQFormer(
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
            loss, pieces = compute_m21_loss(
                outputs,
                batch,
                trace_weight=float(trace_weight),
                answer_weight=float(answer_weight),
                counterfactual_weight=float(counterfactual_weight),
                brivi_lock_weight=float(brivi_lock_weight),
                frame_necessity_weight=float(frame_necessity_weight),
                mdl_weight=float(mdl_weight),
                necessity_margin=float(necessity_margin),
                pointer_necessity_weight=float(pointer_necessity_weight),
                pointer_necessity_margin=float(pointer_necessity_margin),
                hyperbolic_topology_weight=float(hyperbolic_topology_weight),
            )
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError(f"M21 loss became non-finite at epoch {epoch + 1}, batch {batches + 1}.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm).detach().cpu().item()):
                raise FloatingPointError(f"M21 gradient norm became non-finite at epoch {epoch + 1}, batch {batches + 1}.")
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            totals["grad_norm"] += float(grad_norm.detach().cpu().item())
            totals["nan_batches"] += 0.0
            for key, value in pieces.items():
                totals[key] += value
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    metrics = evaluate_model(model, eval_examples, vocab, batch_size=int(batch_size), device=device)
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
            "max_cmavo_per_frame": int(DEFAULT_MAX_CMAVO_PER_FRAME),
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
            "hyperbolic_topology_weight": float(hyperbolic_topology_weight),
            "geometry_mode": str(geometry_mode),
            "poincare_curvature": float(poincare_curvature),
            "poincare_max_norm": float(poincare_max_norm),
            "riemannian_gradient_scale": bool(riemannian_gradient_scale),
            "judri_bridge_gate": bool(judri_bridge_gate),
            "judri_bridge_gate_temperature": float(judri_bridge_gate_temperature),
        },
    }
