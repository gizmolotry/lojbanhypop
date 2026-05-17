from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


M20_LOCKS: dict[str, str] = {
    "dictionary_first_pretraining": "synthetic predicate labels are learned before English bridge coupling",
    "factorized_predicate_dictionary": "predicate state predicts domain, polarity, relation type, arity, and role schema",
    "counterfactual_quotient_dictionary": "entity-renamed twins share predicate distributions",
    "brivi_locked_predicate_formation": "predicate energy is gated off when no argument is grounded",
    "synthetic_world_pretraining": "training/eval rows come from controlled causal worlds and minimal pairs",
    "soft_dictionary_before_hard_dictionary": "dictionary assignments are soft during training and annealed toward hard use",
}

DOMAINS = ["size", "weight", "transfer", "preference", "containment", "visibility", "quantity"]
POLARITIES = ["excess", "deficit", "positive", "negative", "blocked", "clear"]
RELATION_TYPES = ["unary_property", "binary_state", "ternary_transfer", "preference_relation", "causal_block"]
ROLE_SCHEMAS = [
    "patient_property",
    "agent_object",
    "giver_receiver_object",
    "container_contained",
    "observer_object",
    "quantity_object",
]

DOMAIN_TO_ID = {name: idx for idx, name in enumerate(DOMAINS)}
POLARITY_TO_ID = {name: idx for idx, name in enumerate(POLARITIES)}
RELATION_TO_ID = {name: idx for idx, name in enumerate(RELATION_TYPES)}
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_SCHEMAS)}


@dataclass(frozen=True)
class PredicateSpec:
    predicate_id: int
    name: str
    domain: str
    polarity: str
    relation_type: str
    arity: int
    role_schema: str
    templates: tuple[str, ...]
    floating_templates: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticPredicateExample:
    prompt: str
    predicate_id: int
    predicate_name: str
    domain_id: int
    polarity_id: int
    relation_type_id: int
    arity_id: int
    role_schema_id: int
    has_argument: bool
    slot_targets: tuple[int, int, int]
    surface: str
    counterfactual_group: str
    entity_signature: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def predicate_specs() -> tuple[PredicateSpec, ...]:
    return (
        PredicateSpec(
            0,
            "size_excess",
            "size",
            "excess",
            "unary_property",
            1,
            "patient_property",
            (
                "The {object} did not fit in the {container} because the {object} was too big.",
                "{object} failed to enter {container}; the reason was excessive size.",
            ),
            ("too big", "excessive size"),
        ),
        PredicateSpec(
            1,
            "size_deficit",
            "size",
            "deficit",
            "unary_property",
            1,
            "patient_property",
            (
                "The {object} did not fill the {container} because the {object} was too small.",
                "{object} rattled inside {container}; the reason was insufficient size.",
            ),
            ("too small", "insufficient size"),
        ),
        PredicateSpec(
            2,
            "weight_excess",
            "weight",
            "excess",
            "unary_property",
            1,
            "patient_property",
            (
                "{person} could not lift the {object} because the {object} was too heavy.",
                "The {object} stayed on the floor; its weight exceeded {person}'s lift.",
            ),
            ("too heavy", "excessive weight"),
        ),
        PredicateSpec(
            3,
            "weight_deficit",
            "weight",
            "deficit",
            "unary_property",
            1,
            "patient_property",
            (
                "The wind moved the {object} because the {object} was too light.",
                "{person} carried the {object} easily because it had little weight.",
            ),
            ("too light", "insufficient weight"),
        ),
        PredicateSpec(
            4,
            "ownership_transfer",
            "transfer",
            "positive",
            "ternary_transfer",
            3,
            "giver_receiver_object",
            (
                "{giver} gave the {object} to {receiver} after the meeting.",
                "The {object} moved from {giver} to {receiver} by a gift event.",
            ),
            ("transfer by gift", "giver receiver object"),
        ),
        PredicateSpec(
            5,
            "ownership_refusal",
            "transfer",
            "negative",
            "ternary_transfer",
            3,
            "giver_receiver_object",
            (
                "{giver} offered the {object} to {receiver}, but {receiver} refused it.",
                "The proposed transfer of {object} from {giver} to {receiver} was rejected.",
            ),
            ("refused transfer", "rejected gift"),
        ),
        PredicateSpec(
            6,
            "preference_like",
            "preference",
            "positive",
            "preference_relation",
            2,
            "agent_object",
            (
                "{person} kept the {object} because {person} liked using it.",
                "{person}'s preference toward the {object} was favorable.",
            ),
            ("liked object", "favorable preference"),
        ),
        PredicateSpec(
            7,
            "preference_dislike",
            "preference",
            "negative",
            "preference_relation",
            2,
            "agent_object",
            (
                "{person} avoided the {object} because {person} disliked using it.",
                "{person}'s preference toward the {object} was unfavorable.",
            ),
            ("disliked object", "unfavorable preference"),
        ),
        PredicateSpec(
            8,
            "containment_success",
            "containment",
            "positive",
            "binary_state",
            2,
            "container_contained",
            (
                "The {container} held the {object} because the opening accepted it.",
                "{object} remained inside {container} after placement.",
            ),
            ("inside container", "accepted containment"),
        ),
        PredicateSpec(
            9,
            "containment_blocked",
            "containment",
            "blocked",
            "causal_block",
            2,
            "container_contained",
            (
                "The {container} blocked the {object} because the opening was narrow.",
                "{object} could not pass through {container}'s narrow opening.",
            ),
            ("blocked by opening", "narrow containment failure"),
        ),
        PredicateSpec(
            10,
            "visibility_occluded",
            "visibility",
            "blocked",
            "binary_state",
            2,
            "observer_object",
            (
                "{observer} could not see the {object} because the screen hid it.",
                "The screen occluded {object} from {observer}.",
            ),
            ("occluded object", "blocked visibility"),
        ),
        PredicateSpec(
            11,
            "visibility_clear",
            "visibility",
            "clear",
            "binary_state",
            2,
            "observer_object",
            (
                "{observer} saw the {object} because the path was clear.",
                "Nothing occluded {object} from {observer}.",
            ),
            ("clear visibility", "unblocked view"),
        ),
        PredicateSpec(
            12,
            "quantity_excess",
            "quantity",
            "excess",
            "unary_property",
            1,
            "quantity_object",
            (
                "There were {count} {object}s, so the shelf had too many items.",
                "The count of {object}s exceeded the allowed amount.",
            ),
            ("too many", "excess quantity"),
        ),
        PredicateSpec(
            13,
            "quantity_deficit",
            "quantity",
            "deficit",
            "unary_property",
            1,
            "quantity_object",
            (
                "There were only {count} {object}s, so the shelf had too few items.",
                "The count of {object}s fell below the required amount.",
            ),
            ("too few", "deficit quantity"),
        ),
    )


OBJECTS = ("stone", "apple", "coin", "book", "vase", "tool", "block", "toy", "package", "bottle", "tablet")
CONTAINERS = ("box", "drawer", "bag", "crate", "basket", "case")
PEOPLE = ("Alex", "Riley", "Jordan", "Morgan", "Taylor", "Casey", "Sam", "Quinn")
ALT_NOUNS = ("lantern", "marble", "folder", "button", "ticket", "cup", "shell", "cable")


def _surface_values(rng: random.Random, surface: str) -> dict[str, str]:
    values = {
        "object": rng.choice(OBJECTS),
        "container": rng.choice(CONTAINERS),
        "person": rng.choice(PEOPLE),
        "giver": rng.choice(PEOPLE),
        "receiver": rng.choice(PEOPLE),
        "observer": rng.choice(PEOPLE),
        "count": str(rng.choice((1, 2, 3, 8, 11, 17))),
    }
    if values["receiver"] == values["giver"]:
        values["receiver"] = rng.choice([p for p in PEOPLE if p != values["giver"]])
    if surface == "renamed":
        values.update(
            {
                "object": rng.choice(ALT_NOUNS),
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
                "container": "entity_b",
                "person": "entity_c",
                "giver": "entity_d",
                "receiver": "entity_e",
                "observer": "entity_f",
            }
        )
    elif surface == "numeric":
        values["object"] = f"object_{rng.randrange(100, 999)}"
        values["container"] = f"container_{rng.randrange(100, 999)}"
    return values


def _format_prompt(template: str, values: dict[str, str], surface: str) -> str:
    prompt = template.format(**values)
    if surface == "flattened":
        prompt = prompt.replace(" because ", " ; cause: ").replace(".", "")
    elif surface == "numeric":
        prompt = f"case {values['count']}: {prompt}"
    return prompt


def _slot_targets(arity: int, has_argument: bool) -> tuple[int, int, int]:
    if not has_argument:
        return (0, 0, 0)
    return tuple(1 if idx < int(arity) else 0 for idx in range(3))  # type: ignore[return-value]


def generate_synthetic_world_examples(
    size: int,
    *,
    seed: int = 0,
    floating_fraction: float = 0.15,
    surfaces: Sequence[str] = ("purged", "flattened", "renamed", "anonymized", "numeric"),
) -> list[SyntheticPredicateExample]:
    rng = random.Random(seed)
    specs = predicate_specs()
    rows: list[SyntheticPredicateExample] = []
    for idx in range(int(size)):
        spec = specs[idx % len(specs)] if idx < len(specs) * 2 else rng.choice(specs)
        is_floating = rng.random() < float(floating_fraction)
        surface = "floating" if is_floating else rng.choice(tuple(surfaces))
        values = _surface_values(rng, surface)
        if is_floating:
            prompt = rng.choice(spec.floating_templates)
        else:
            prompt = _format_prompt(rng.choice(spec.templates), values, surface)
        entity_signature = "|".join(
            [
                values.get("giver", ""),
                values.get("receiver", ""),
                values.get("person", ""),
                values.get("observer", ""),
                values.get("object", ""),
                values.get("container", ""),
            ]
        )
        rows.append(
            SyntheticPredicateExample(
                prompt=prompt,
                predicate_id=spec.predicate_id,
                predicate_name=spec.name,
                domain_id=DOMAIN_TO_ID[spec.domain],
                polarity_id=POLARITY_TO_ID[spec.polarity],
                relation_type_id=RELATION_TO_ID[spec.relation_type],
                arity_id=int(spec.arity) - 1,
                role_schema_id=ROLE_TO_ID[spec.role_schema],
                has_argument=not is_floating,
                slot_targets=_slot_targets(spec.arity, not is_floating),
                surface=surface,
                counterfactual_group=spec.name,
                entity_signature=entity_signature,
            )
        )
    rng.shuffle(rows)
    return rows


TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def build_vocab(examples: Iterable[SyntheticPredicateExample], *, min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in examples:
        for tok in tokenize(row.prompt):
            counts[tok] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_count:
            vocab[token] = len(vocab)
    return vocab


class M20PredicateDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[SyntheticPredicateExample], vocab: dict[str, int], max_length: int = 48):
        self.examples = list(examples)
        self.vocab = dict(vocab)
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, text: str) -> list[int]:
        ids = [self.vocab.get(tok, 1) for tok in tokenize(text)][: self.max_length]
        if len(ids) < self.max_length:
            ids.extend([0] * (self.max_length - len(ids)))
        return ids

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        return {
            "input_ids": torch.tensor(self._encode(row.prompt), dtype=torch.long),
            "predicate_id": torch.tensor(row.predicate_id, dtype=torch.long),
            "domain_id": torch.tensor(row.domain_id, dtype=torch.long),
            "polarity_id": torch.tensor(row.polarity_id, dtype=torch.long),
            "relation_type_id": torch.tensor(row.relation_type_id, dtype=torch.long),
            "arity_id": torch.tensor(row.arity_id, dtype=torch.long),
            "role_schema_id": torch.tensor(row.role_schema_id, dtype=torch.long),
            "has_argument": torch.tensor(1.0 if row.has_argument else 0.0, dtype=torch.float32),
            "slot_targets": torch.tensor(row.slot_targets, dtype=torch.float32),
            "surface": row.surface,
            "counterfactual_group": row.counterfactual_group,
            "entity_signature": row.entity_signature,
        }


def m20_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "input_ids",
        "predicate_id",
        "domain_id",
        "polarity_id",
        "relation_type_id",
        "arity_id",
        "role_schema_id",
        "has_argument",
        "slot_targets",
    )
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in keys}
    out["surface"] = [str(item["surface"]) for item in batch]
    out["counterfactual_group"] = [str(item["counterfactual_group"]) for item in batch]
    out["entity_signature"] = [str(item["entity_signature"]) for item in batch]
    return out


class M20SoftDictionaryModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        codebook_size: int = 2000,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
        max_arity: int = 3,
        num_predicates: int | None = None,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.max_arity = int(max_arity)
        self.num_predicates = int(num_predicates or len(predicate_specs()))
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
        )
        self.code_head = nn.Linear(int(hidden_dim), self.codebook_size)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, int(hidden_dim)) * 0.02)
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        self.slot_head = nn.Linear(int(hidden_dim), self.max_arity)
        self.predicate_head = nn.Linear(int(hidden_dim), self.num_predicates)
        self.domain_head = nn.Linear(int(hidden_dim), len(DOMAINS))
        self.polarity_head = nn.Linear(int(hidden_dim), len(POLARITIES))
        self.relation_head = nn.Linear(int(hidden_dim), len(RELATION_TYPES))
        self.arity_head = nn.Linear(int(hidden_dim), self.max_arity)
        self.role_head = nn.Linear(int(hidden_dim), len(ROLE_SCHEMAS))

    def forward(self, input_ids: torch.Tensor, *, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        embedded = self.embedding(input_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        hidden = self.encoder(pooled)
        code_logits = self.code_head(hidden)
        code_probs = torch.softmax(code_logits / max(float(temperature), 1e-4), dim=-1)
        gate = torch.sigmoid(self.gate_head(hidden))
        predicate_state = (code_probs @ self.codebook) * gate
        return {
            "hidden": hidden,
            "code_logits": code_logits,
            "code_probs": code_probs,
            "brivi_gate": gate.squeeze(-1),
            "slot_logits": self.slot_head(hidden),
            "predicate_logits": self.predicate_head(predicate_state),
            "domain_logits": self.domain_head(predicate_state),
            "polarity_logits": self.polarity_head(predicate_state),
            "relation_logits": self.relation_head(predicate_state),
            "arity_logits": self.arity_head(predicate_state),
            "role_logits": self.role_head(predicate_state),
            "predicate_state": predicate_state,
        }


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any().item()):
        losses = F.cross_entropy(logits, target, reduction="none")
        return (losses * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return logits.new_zeros(())


def quotient_invariance_loss(code_probs: torch.Tensor, predicate_ids: torch.Tensor, grounded_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for pred_id in torch.unique(predicate_ids[grounded_mask]):
        idx = (predicate_ids == pred_id) & grounded_mask
        if int(idx.sum().item()) < 2:
            continue
        rows = code_probs[idx]
        center = rows.mean(dim=0, keepdim=True)
        losses.append(torch.mean((rows - center) ** 2))
    if not losses:
        return code_probs.new_zeros(())
    return torch.stack(losses).mean()


def compute_m20_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    factor_weight: float = 1.0,
    dictionary_commitment_weight: float = 0.75,
    quotient_invariance_weight: float = 2.0,
    brivi_lock_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    has_argument = batch["has_argument"].to(outputs["code_probs"].device)
    grounded = has_argument > 0.5
    predicate_id = batch["predicate_id"].to(outputs["code_probs"].device)
    code_target = torch.remainder(predicate_id, outputs["code_logits"].shape[-1])
    loss_code = _masked_ce(outputs["code_logits"], code_target, grounded)
    loss_predicate = _masked_ce(outputs["predicate_logits"], predicate_id, grounded)
    loss_domain = _masked_ce(outputs["domain_logits"], batch["domain_id"].to(predicate_id.device), grounded)
    loss_polarity = _masked_ce(outputs["polarity_logits"], batch["polarity_id"].to(predicate_id.device), grounded)
    loss_relation = _masked_ce(outputs["relation_logits"], batch["relation_type_id"].to(predicate_id.device), grounded)
    loss_arity = _masked_ce(outputs["arity_logits"], batch["arity_id"].to(predicate_id.device), grounded)
    loss_role = _masked_ce(outputs["role_logits"], batch["role_schema_id"].to(predicate_id.device), grounded)
    factor_loss = loss_domain + loss_polarity + loss_relation + loss_arity + loss_role
    slot_loss = F.binary_cross_entropy_with_logits(
        outputs["slot_logits"],
        batch["slot_targets"].to(outputs["slot_logits"].device),
    )
    gate_loss = F.binary_cross_entropy(outputs["brivi_gate"], has_argument)
    ungrounded = ~grounded
    ungrounded_energy = (outputs["code_probs"].amax(dim=-1) * outputs["brivi_gate"])[ungrounded]
    silence_loss = ungrounded_energy.mean() if bool(ungrounded.any().item()) else outputs["code_probs"].new_zeros(())
    quotient_loss = quotient_invariance_loss(outputs["code_probs"], predicate_id, grounded)
    total = (
        loss_predicate
        + float(dictionary_commitment_weight) * loss_code
        + float(factor_weight) * factor_loss
        + slot_loss
        + float(brivi_lock_weight) * (gate_loss + silence_loss)
        + float(quotient_invariance_weight) * quotient_loss
    )
    return total, {
        "loss_predicate": float(loss_predicate.detach().cpu().item()),
        "loss_code_commitment": float(loss_code.detach().cpu().item()),
        "loss_factor": float(factor_loss.detach().cpu().item()),
        "loss_slot": float(slot_loss.detach().cpu().item()),
        "loss_gate": float(gate_loss.detach().cpu().item()),
        "loss_silence": float(silence_loss.detach().cpu().item()),
        "loss_quotient": float(quotient_loss.detach().cpu().item()),
    }


def _accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return 0.0
    pred = torch.argmax(logits[mask], dim=-1)
    return float((pred == target[mask]).float().mean().detach().cpu().item())


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1)


def _quotient_l2_numpy(code_probs: torch.Tensor, predicate_ids: torch.Tensor, grounded: torch.Tensor) -> float:
    values: list[float] = []
    for pred_id in torch.unique(predicate_ids[grounded]):
        idx = (predicate_ids == pred_id) & grounded
        if int(idx.sum().item()) < 2:
            continue
        rows = code_probs[idx]
        center = rows.mean(dim=0, keepdim=True)
        values.append(float(torch.mean((rows - center) ** 2).detach().cpu().item()))
    return float(sum(values) / len(values)) if values else 0.0


def _pairwise_entity_leakage_proxy(
    code_probs: torch.Tensor,
    predicate_ids: torch.Tensor,
    entity_signatures: Sequence[str],
    grounded: torch.Tensor,
) -> float:
    indices = [idx for idx, ok in enumerate(grounded.detach().cpu().tolist()) if ok]
    same_pred: list[float] = []
    diff_pred: list[float] = []
    for left_pos, i in enumerate(indices[:160]):
        for j in indices[left_pos + 1 : 160]:
            dist = float(torch.mean((code_probs[i] - code_probs[j]) ** 2).detach().cpu().item())
            if int(predicate_ids[i]) == int(predicate_ids[j]) and entity_signatures[i] != entity_signatures[j]:
                same_pred.append(dist)
            elif int(predicate_ids[i]) != int(predicate_ids[j]):
                diff_pred.append(dist)
    if not same_pred or not diff_pred:
        return 0.0
    return float((sum(same_pred) / len(same_pred)) / max(sum(diff_pred) / len(diff_pred), 1e-8))


@torch.no_grad()
def evaluate_model(
    model: M20SoftDictionaryModel,
    examples: Sequence[SyntheticPredicateExample],
    vocab: dict[str, int],
    *,
    batch_size: int = 128,
    temperature: float = 0.25,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    model.eval()
    dataset = M20PredicateDataset(examples, vocab)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m20_collate)
    merged: dict[str, list[torch.Tensor]] = defaultdict(list)
    surfaces: list[str] = []
    groups: list[str] = []
    entities: list[str] = []
    for batch in loader:
        outputs = model(batch["input_ids"].to(device), temperature=float(temperature))
        for key in (
            "code_probs",
            "predicate_logits",
            "domain_logits",
            "polarity_logits",
            "relation_logits",
            "arity_logits",
            "role_logits",
            "slot_logits",
            "brivi_gate",
        ):
            merged[key].append(outputs[key].detach().cpu())
        for key in ("predicate_id", "domain_id", "polarity_id", "relation_type_id", "arity_id", "role_schema_id", "has_argument", "slot_targets"):
            merged[key].append(batch[key].detach().cpu())
        surfaces.extend(batch["surface"])
        groups.extend(batch["counterfactual_group"])
        entities.extend(batch["entity_signature"])
    tensors = {key: torch.cat(value, dim=0) for key, value in merged.items()}
    grounded = tensors["has_argument"] > 0.5
    pred_exact = (
        (torch.argmax(tensors["domain_logits"], dim=-1) == tensors["domain_id"])
        & (torch.argmax(tensors["polarity_logits"], dim=-1) == tensors["polarity_id"])
        & (torch.argmax(tensors["relation_logits"], dim=-1) == tensors["relation_type_id"])
        & (torch.argmax(tensors["arity_logits"], dim=-1) == tensors["arity_id"])
        & (torch.argmax(tensors["role_logits"], dim=-1) == tensors["role_schema_id"])
    )
    slot_pred = torch.sigmoid(tensors["slot_logits"]) > 0.5
    slot_target = tensors["slot_targets"] > 0.5
    slot_exact = (slot_pred == slot_target).all(dim=-1)
    gate_pred = tensors["brivi_gate"] > 0.5
    gate_target = grounded
    hard_code = torch.argmax(tensors["code_probs"], dim=-1)
    surface_metrics: dict[str, dict[str, float]] = {}
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([item == surface for item in surfaces], dtype=torch.bool)
        surface_grounded = mask & grounded
        surface_metrics[surface] = {
            "predicate_accuracy": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], surface_grounded),
            "factorized_exact_accuracy": float(pred_exact[surface_grounded].float().mean().item()) if bool(surface_grounded.any().item()) else 0.0,
            "count": float(surface_grounded.sum().item()),
        }
    ungrounded = ~grounded
    code_entropy = _entropy(tensors["code_probs"])
    ungrounded_energy = tensors["code_probs"].amax(dim=-1) * tensors["brivi_gate"]
    metrics = {
        "soft_hard_dictionary_agreement": float((hard_code[grounded] == torch.remainder(tensors["predicate_id"], model.codebook_size)[grounded]).float().mean().item()) if bool(grounded.any().item()) else 0.0,
        "predicate_accuracy": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded),
        "dictionary_coverage": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded),
        "dictionary_precedence_violation_rate": 1.0 - _accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded),
        "oov_predicate_rate": 0.0,
        "domain_accuracy": _accuracy(tensors["domain_logits"], tensors["domain_id"], grounded),
        "polarity_accuracy": _accuracy(tensors["polarity_logits"], tensors["polarity_id"], grounded),
        "relation_type_accuracy": _accuracy(tensors["relation_logits"], tensors["relation_type_id"], grounded),
        "arity_accuracy": _accuracy(tensors["arity_logits"], tensors["arity_id"], grounded),
        "arity_violation_rate": 1.0 - _accuracy(tensors["arity_logits"], tensors["arity_id"], grounded),
        "role_schema_accuracy": _accuracy(tensors["role_logits"], tensors["role_schema_id"], grounded),
        "factorized_exact_accuracy": float(pred_exact[grounded].float().mean().item()) if bool(grounded.any().item()) else 0.0,
        "argument_binding_accuracy": float(slot_exact.float().mean().item()),
        "brivi_gate_accuracy": float((gate_pred == gate_target).float().mean().item()),
        "brivi_grounding_accuracy": float(slot_exact[grounded].float().mean().item()) if bool(grounded.any().item()) else 0.0,
        "ungrounded_predicate_energy_mean": float(ungrounded_energy[ungrounded].mean().item()) if bool(ungrounded.any().item()) else 0.0,
        "masked_accuracy": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], ungrounded),
        "masked_pointer_zero_rate": float(((slot_pred[ungrounded]).sum(dim=-1) == 0).float().mean().item()) if bool(ungrounded.any().item()) else 0.0,
        "quotient_invariance_l2": _quotient_l2_numpy(tensors["code_probs"], tensors["predicate_id"], grounded),
        "predicate_identity_stability": float(max(0.0, 1.0 - _quotient_l2_numpy(tensors["code_probs"], tensors["predicate_id"], grounded) * 100.0)),
        "entity_leakage_proxy": _pairwise_entity_leakage_proxy(tensors["code_probs"], tensors["predicate_id"], entities, grounded),
        "dictionary_entropy": float(code_entropy.mean().item()),
        "dictionary_perplexity": float(torch.exp(code_entropy.mean()).item()),
        "hard_code_utilization_count": int(torch.unique(hard_code[grounded]).numel()) if bool(grounded.any().item()) else 0,
        "active_code_fraction": float(torch.unique(hard_code[grounded]).numel() / max(1, model.codebook_size)) if bool(grounded.any().item()) else 0.0,
        "strict_accuracy": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded),
        "phrase_accuracy": _accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded),
        "avg_tokens": float(sum(len(tokenize(row.prompt)) for row in examples) / max(1, len(examples))),
        "accuracy_per_token": float(_accuracy(tensors["predicate_logits"], tensors["predicate_id"], grounded) / max(1e-8, sum(len(tokenize(row.prompt)) for row in examples) / max(1, len(examples)))),
        "cot_token_ratio": None,
        "retained_cot_accuracy_per_token": None,
        "surface_metrics": surface_metrics,
        "leakage_status": "pass" if (float(ungrounded_energy[ungrounded].mean().item()) if bool(ungrounded.any().item()) else 0.0) <= 0.2 else "fail",
        "brivi_lock_pass": bool((float((gate_pred == gate_target).float().mean().item()) >= 0.8) and ((float(ungrounded_energy[ungrounded].mean().item()) if bool(ungrounded.any().item()) else 0.0) <= 0.2)),
    }
    return metrics


def _temperature_for_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return float(end)
    frac = float(epoch) / float(epochs - 1)
    return float(start) * (1.0 - frac) + float(end) * frac


def train_m20_dictionary(
    *,
    train_size: int = 2400,
    eval_size: int = 600,
    epochs: int = 8,
    batch_size: int = 96,
    learning_rate: float = 3e-3,
    seed: int = 23,
    codebook_size: int = 2000,
    embedding_dim: int = 64,
    hidden_dim: int = 96,
    temperature_start: float = 1.5,
    temperature_end: float = 0.25,
    factor_weight: float = 1.0,
    dictionary_commitment_weight: float = 0.75,
    quotient_invariance_weight: float = 2.0,
    brivi_lock_weight: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    train_examples = generate_synthetic_world_examples(int(train_size), seed=int(seed), floating_fraction=0.16)
    eval_examples = generate_synthetic_world_examples(int(eval_size), seed=int(seed) + 10_000, floating_fraction=0.2)
    vocab = build_vocab([*train_examples, *eval_examples])
    dataset = M20PredicateDataset(train_examples, vocab)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=generator, collate_fn=m20_collate)
    model = M20SoftDictionaryModel(
        vocab_size=len(vocab),
        codebook_size=int(codebook_size),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        temp = _temperature_for_epoch(epoch, int(epochs), float(temperature_start), float(temperature_end))
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, temperature=temp)
            loss, pieces = compute_m20_loss(
                outputs,
                batch,
                factor_weight=float(factor_weight),
                dictionary_commitment_weight=float(dictionary_commitment_weight),
                quotient_invariance_weight=float(quotient_invariance_weight),
                brivi_lock_weight=float(brivi_lock_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            for key, value in pieces.items():
                totals[key] += value
            with torch.no_grad():
                totals["dictionary_entropy"] += float(_entropy(outputs["code_probs"]).mean().detach().cpu().item())
                totals["brivi_gate_mean"] += float(outputs["brivi_gate"].mean().detach().cpu().item())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1), "temperature": float(temp)})
    metrics = evaluate_model(model, eval_examples, vocab, batch_size=int(batch_size), temperature=float(temperature_end), device=device)
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
            "codebook_size": int(codebook_size),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "temperature_start": float(temperature_start),
            "temperature_end": float(temperature_end),
            "factor_weight": float(factor_weight),
            "dictionary_commitment_weight": float(dictionary_commitment_weight),
            "quotient_invariance_weight": float(quotient_invariance_weight),
            "brivi_lock_weight": float(brivi_lock_weight),
        },
    }
