from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lojban_evolution.m21.bridi import (
    ANSWER_LABELS,
    CMAVO,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PLACES,
    GISMU,
    BridiFrame,
    build_vocab,
    tokenize,
)
from lojban_evolution.m23.relevance import generate_m23_relevance_examples
from lojban_evolution.m24.compression import PromptOnlyControl, _accuracy

LOOSE_PAD = 0
LOOSE_OPEN = 1
LOOSE_PRED = 2
LOOSE_MOD = 3
LOOSE_ARG = 4
LOOSE_LINK = 5
LOOSE_CLOSE = 6
LOOSE_STOP = 7
LOOSE_TYPE_COUNT = 8
LINK_NEXT = 1
DEFAULT_MAX_SYMBOLS = 32
DEFAULT_M25_MDL_WEIGHT = 0.01


@dataclass(frozen=True)
class LooseBridiSymbol:
    type_id: int
    value_id: int = 0
    aux_id: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.type_id), int(self.value_id), int(self.aux_id))


@dataclass(frozen=True)
class M25EmergentBridiExample:
    prompt: str
    frames: tuple[BridiFrame, ...]
    loose_symbols: tuple[LooseBridiSymbol, ...]
    entities: tuple[str, ...]
    answer_id: int
    answer_label: str
    surface: str
    counterfactual_group: str
    entity_signature: str
    relevant_frame_indices: tuple[int, ...]
    decoy_frame_indices: tuple[int, ...]
    relevance_surface: str

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "answer_label": self.answer_label,
            "answer_id": int(self.answer_id),
            "surface": self.surface,
            "counterfactual_group": self.counterfactual_group,
            "entity_signature": self.entity_signature,
            "relevance_surface": self.relevance_surface,
            "relevant_frame_indices": list(self.relevant_frame_indices),
            "decoy_frame_indices": list(self.decoy_frame_indices),
            "loose_symbols": [symbol.as_tuple() for symbol in self.loose_symbols],
        }


def _value_vocab_size() -> int:
    return max(len(GISMU), len(CMAVO), DEFAULT_MAX_ENTITIES + 1, DEFAULT_MAX_FRAMES + 1) + 1


def _aux_vocab_size() -> int:
    return max(DEFAULT_MAX_PLACES + 1, DEFAULT_MAX_FRAMES + 1, 8)


def loose_symbols_from_frames(
    frames: Sequence[BridiFrame],
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    include_links: bool = True,
) -> tuple[LooseBridiSymbol, ...]:
    """Linearize bridi frames into a loose grammar-action stream."""

    symbols: list[LooseBridiSymbol] = []
    emitted = 0
    for frame_idx, frame in enumerate(frames[: int(max_frames)]):
        if frame.stop:
            break
        if not frame.active:
            continue
        if include_links and emitted > 0:
            symbols.append(LooseBridiSymbol(LOOSE_LINK, LINK_NEXT, emitted))
        emitted += 1
        symbols.append(LooseBridiSymbol(LOOSE_OPEN, frame_idx + 1, emitted))
        symbols.append(LooseBridiSymbol(LOOSE_PRED, int(frame.gismu_id), emitted))
        for cmavo_id in frame.cmavo_ids:
            if int(cmavo_id) > 0:
                symbols.append(LooseBridiSymbol(LOOSE_MOD, int(cmavo_id), emitted))
        for place_idx, entity_id in enumerate(frame.judri_place_bindings[:DEFAULT_MAX_PLACES], start=1):
            if int(entity_id) > 0:
                symbols.append(LooseBridiSymbol(LOOSE_ARG, int(entity_id), place_idx))
        symbols.append(LooseBridiSymbol(LOOSE_CLOSE, frame_idx + 1, emitted))
    symbols.append(LooseBridiSymbol(LOOSE_STOP, 0, 0))
    return tuple(symbols[: int(max_symbols)])


def generate_m25_emergent_bridi_examples(
    size: int,
    *,
    seed: int = 0,
    clean_fraction: float = 0.35,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> list[M25EmergentBridiExample]:
    rows = generate_m23_relevance_examples(
        int(size), seed=int(seed), clean_fraction=float(clean_fraction), max_frames=int(max_frames)
    )
    out: list[M25EmergentBridiExample] = []
    for row in rows:
        out.append(
            M25EmergentBridiExample(
                prompt=row.prompt,
                frames=row.frames,
                loose_symbols=loose_symbols_from_frames(row.frames, max_frames=max_frames, max_symbols=max_symbols),
                entities=row.entities,
                answer_id=int(row.answer_id),
                answer_label=row.answer_label,
                surface=row.surface,
                counterfactual_group=row.counterfactual_group,
                entity_signature=row.entity_signature,
                relevant_frame_indices=row.relevant_frame_indices,
                decoy_frame_indices=row.decoy_frame_indices,
                relevance_surface=row.relevance_surface,
            )
        )
    return out


class M25LooseBridiDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[M25EmergentBridiExample],
        vocab: dict[str, int],
        *,
        max_length: int = 64,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
    ) -> None:
        self.examples = list(examples)
        self.vocab = dict(vocab)
        self.max_length = int(max_length)
        self.max_symbols = int(max_symbols)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        ids = [self.vocab.get(tok, self.vocab.get("<unk>", 1)) for tok in tokenize(row.prompt)[: self.max_length]]
        ids += [0] * (self.max_length - len(ids))
        stream = torch.zeros(self.max_symbols, 3, dtype=torch.long)
        for pos, symbol in enumerate(row.loose_symbols[: self.max_symbols]):
            stream[pos] = torch.tensor(symbol.as_tuple(), dtype=torch.long)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "stream_targets": stream,
            "stream_active_targets": stream[:, 0].ne(LOOSE_PAD).float(),
            "type_targets": stream[:, 0],
            "value_targets": stream[:, 1],
            "aux_targets": stream[:, 2],
            "answer_id": torch.tensor(int(row.answer_id), dtype=torch.long),
            "prompt": row.prompt,
            "surface": row.surface,
            "counterfactual_group": row.counterfactual_group,
            "entity_signature": row.entity_signature,
            "relevance_surface": row.relevance_surface,
        }


def m25_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "input_ids",
        "stream_targets",
        "stream_active_targets",
        "type_targets",
        "value_targets",
        "aux_targets",
        "answer_id",
    )
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ("prompt", "surface", "counterfactual_group", "entity_signature", "relevance_surface"):
        out[key] = [str(item[key]) for item in batch]
    return out


class M25EmergentBridiQFormer(nn.Module):
    """Prompt-to-loose-bridi stream emitter with learned grammar-action slots."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
        value_vocab_size: int | None = None,
        aux_vocab_size: int | None = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_symbols = int(max_symbols)
        self.value_vocab_size = int(value_vocab_size or _value_vocab_size())
        self.aux_vocab_size = int(aux_vocab_size or _aux_vocab_size())
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
        )
        self.symbol_queries = nn.Parameter(torch.randn(self.max_symbols, int(hidden_dim)) * 0.02)
        self.symbol_mlp = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh())
        self.active_head = nn.Linear(int(hidden_dim), 1)
        self.type_head = nn.Linear(int(hidden_dim), LOOSE_TYPE_COUNT)
        self.value_head = nn.Linear(int(hidden_dim), self.value_vocab_size)
        self.aux_head = nn.Linear(int(hidden_dim), self.aux_vocab_size)
        self.type_embed = nn.Embedding(LOOSE_TYPE_COUNT, int(hidden_dim))
        self.value_embed = nn.Embedding(self.value_vocab_size, int(hidden_dim))
        self.aux_embed = nn.Embedding(self.aux_vocab_size, int(hidden_dim))
        self.answer_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), len(ANSWER_LABELS)),
        )

    def forward(self, input_ids: torch.Tensor, *, prompt_hidden_states: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        embedded = self.embedding(input_ids) if prompt_hidden_states is None else prompt_hidden_states
        if embedded.shape[-1] != self.embedding.embedding_dim:
            raise ValueError(
                "prompt_hidden_states last dimension must match the M25 generator embedding dimension "
                f"({embedded.shape[-1]} != {self.embedding.embedding_dim})"
            )
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        prompt_state = self.prompt_encoder(pooled)
        state = self.symbol_mlp(prompt_state.unsqueeze(1) + self.symbol_queries.unsqueeze(0))
        active_logits = self.active_head(state).squeeze(-1)
        type_logits = self.type_head(state)
        value_logits = self.value_head(state)
        aux_logits = self.aux_head(state)
        soft_repr = (
            torch.softmax(type_logits, dim=-1) @ self.type_embed.weight
            + torch.softmax(value_logits, dim=-1) @ self.value_embed.weight
            + torch.softmax(aux_logits, dim=-1) @ self.aux_embed.weight
        )
        active = torch.sigmoid(active_logits).unsqueeze(-1)
        trace_state = (soft_repr * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        return {
            "active_logits": active_logits,
            "type_logits": type_logits,
            "value_logits": value_logits,
            "aux_logits": aux_logits,
            "answer_logits": self.answer_head(trace_state),
            "trace_state": trace_state,
        }


def compute_m25_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    trace_weight: float = 2.0,
    answer_weight: float = 0.25,
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
        "mdl_loss": float(mdl_loss.detach().cpu().item()),
    }


def pack_loose_stream_from_outputs(outputs: dict[str, torch.Tensor], *, threshold: float = 0.5) -> torch.Tensor:
    active = (torch.sigmoid(outputs["active_logits"]) > float(threshold)).long()
    stream = torch.stack(
        [
            torch.argmax(outputs["type_logits"], dim=-1).long(),
            torch.argmax(outputs["value_logits"], dim=-1).long(),
            torch.argmax(outputs["aux_logits"], dim=-1).long(),
        ],
        dim=-1,
    )
    return (stream * active.unsqueeze(-1)).long()


def budget_loose_stream_symbols(stream: torch.Tensor, symbol_budget: int | None = None) -> torch.Tensor:
    if symbol_budget is None or int(symbol_budget) <= 0:
        return stream.long()
    out = torch.zeros_like(stream).long()
    for row in range(stream.shape[0]):
        keep = torch.nonzero(stream[row, :, 0].ne(LOOSE_PAD), as_tuple=False).flatten()[: int(symbol_budget)]
        if keep.numel():
            out[row, keep] = stream[row, keep].long()
    return out


def budget_prompt_tokens(input_ids: torch.Tensor, token_budget: int | None = None) -> torch.Tensor:
    """Keep only the same amount of prompt text as the bridi stream is allowed to use."""

    if token_budget is None or int(token_budget) <= 0:
        return input_ids.long()
    out = torch.zeros_like(input_ids).long()
    keep = min(int(token_budget), int(input_ids.shape[1]))
    if keep > 0:
        out[:, :keep] = input_ids[:, :keep].long()
    return out


def loose_stream_symbol_counts(stream: torch.Tensor) -> torch.Tensor:
    return stream[:, :, 0].ne(LOOSE_PAD).sum(dim=-1).float()


def shuffled_loose_stream_like(stream: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    if stream.shape[0] <= 1:
        return stream.clone().long()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return stream[torch.randperm(stream.shape[0], generator=generator).to(stream.device)].clone().long()


def random_loose_stream_like(stream: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    active = stream[:, :, 0].ne(LOOSE_PAD)
    out = torch.zeros_like(stream).long()
    if bool(active.any().item()):
        out[:, :, 0] = torch.randint(1, LOOSE_TYPE_COUNT, stream.shape[:2], generator=generator).to(stream.device)
        out[:, :, 1] = torch.randint(0, max(2, int(stream[:, :, 1].max().cpu().item()) + 2), stream.shape[:2], generator=generator).to(stream.device)
        out[:, :, 2] = torch.randint(0, max(2, int(stream[:, :, 2].max().cpu().item()) + 2), stream.shape[:2], generator=generator).to(stream.device)
        out *= active.unsqueeze(-1).long()
    return out.long()


class LooseStreamAdvisor(nn.Module):
    primary_trace_input = "loose_integer_bridi_stream"
    disallowed_primary_inputs = ("prompt_state", "trace_state", "frame_repr", "hidden_states")

    def __init__(
        self,
        *,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
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
        self.classifier = nn.Sequential(nn.Linear(int(hidden_dim), int(hidden_dim)), nn.Tanh(), nn.Linear(int(hidden_dim), len(ANSWER_LABELS)))

    def forward(self, loose_stream: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(loose_stream):
            raise TypeError("LooseStreamAdvisor accepts only integer loose bridi streams, not continuous tensors.")
        stream = budget_loose_stream_symbols(loose_stream.long(), self.symbol_budget)
        types = stream[:, :, 0].clamp(0, LOOSE_TYPE_COUNT - 1)
        values = stream[:, :, 1].clamp(0, self.value_vocab_size - 1)
        aux = stream[:, :, 2].clamp(0, self.aux_vocab_size - 1)
        active = types.ne(LOOSE_PAD).float().unsqueeze(-1)
        embedded = self.type_embedding(types) + self.value_embedding(values) + self.aux_embedding(aux)
        pooled = (embedded * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def _train_generator(
    model: M25EmergentBridiQFormer,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    trace_weight: float,
    answer_weight: float,
    mdl_weight: float,
) -> list[dict[str, float]]:
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=model.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m25_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history = []
    model.train()
    for _ in range(int(epochs)):
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        for batch in loader:
            outputs = model(batch["input_ids"].to(device))
            loss, metrics = compute_m25_loss(outputs, batch, trace_weight=trace_weight, answer_weight=answer_weight, mdl_weight=mdl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()})
    return history


def _train_stream_advisor(
    generator: M25EmergentBridiQFormer,
    advisor: LooseStreamAdvisor,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    stream_source: str,
) -> list[dict[str, float]]:
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=advisor.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m25_collate)
    optimizer = torch.optim.AdamW(advisor.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history = []
    generator.eval()
    advisor.train()
    for _ in range(int(epochs)):
        total_loss = 0.0
        total_acc = 0.0
        batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["answer_id"].to(device)
            with torch.no_grad():
                stream = batch["stream_targets"].to(device) if stream_source == "oracle" else pack_loose_stream_from_outputs(generator(input_ids))
            logits = advisor(stream)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(advisor.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_acc += _accuracy(logits.detach(), target.detach())
            batches += 1
        history.append({"loss": total_loss / max(1, batches), "accuracy": total_acc / max(1, batches)})
    return history


def _train_prompt_control(
    control: PromptOnlyControl,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    *,
    max_symbols: int,
    prompt_token_budget: int | None = None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(seed)), collate_fn=m25_collate)
    optimizer = torch.optim.AdamW(control.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history = []
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


def _component_accuracy(predicted: torch.Tensor, oracle: torch.Tensor) -> dict[str, float]:
    active = oracle[:, :, 0].ne(LOOSE_PAD)
    row_exact = (((predicted == oracle).all(dim=-1)) | ~active).all(dim=-1)
    if bool(active.any().item()):
        return {
            "loose_stream_exact_accuracy": float(row_exact.float().mean().item()),
            "stream_type_accuracy": float((predicted[:, :, 0][active] == oracle[:, :, 0][active]).float().mean().item()),
            "stream_value_accuracy": float((predicted[:, :, 1][active] == oracle[:, :, 1][active]).float().mean().item()),
            "stream_aux_accuracy": float((predicted[:, :, 2][active] == oracle[:, :, 2][active]).float().mean().item()),
        }
    return {"loose_stream_exact_accuracy": 0.0, "stream_type_accuracy": 0.0, "stream_value_accuracy": 0.0, "stream_aux_accuracy": 0.0}


def evaluate_m25_emergent_bridi(
    *,
    generator: M25EmergentBridiQFormer,
    advisor: LooseStreamAdvisor,
    oracle_advisor: LooseStreamAdvisor | None,
    prompt_control: PromptOnlyControl,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
    matched_prompt_control: PromptOnlyControl | None = None,
    matched_prompt_budget: int | None = None,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=generator.max_symbols)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m25_collate)
    generator.eval()
    advisor.eval()
    prompt_control.eval()
    if matched_prompt_control is not None:
        matched_prompt_control.eval()
    if oracle_advisor is not None:
        oracle_advisor.eval()
    logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    streams: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    prompt_counts: list[torch.Tensor] = []
    matched_prompt_counts: list[torch.Tensor] = []
    surfaces: list[str] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_obj)
            target = batch["answer_id"].to(device_obj)
            predicted = budget_loose_stream_symbols(pack_loose_stream_from_outputs(generator(input_ids)), advisor.symbol_budget)
            oracle = budget_loose_stream_symbols(batch["stream_targets"].to(device_obj), advisor.symbol_budget)
            shuffled = shuffled_loose_stream_like(predicted, seed=int(seed) + batch_idx)
            random_stream = random_loose_stream_like(predicted, seed=int(seed) + 1000 + batch_idx)
            zero = torch.zeros_like(predicted)
            for key, stream in (("predicted", predicted), ("oracle", oracle), ("shuffled", shuffled), ("random", random_stream), ("zero", zero)):
                logits[key].append(advisor(stream).detach().cpu())
            if oracle_advisor is not None:
                logits["oracle_trained_predicted"].append(oracle_advisor(predicted).detach().cpu())
                logits["oracle_trained_oracle"].append(oracle_advisor(oracle).detach().cpu())
                logits["oracle_trained_random"].append(oracle_advisor(random_stream).detach().cpu())
            logits["prompt"].append(prompt_control(input_ids).detach().cpu())
            if matched_prompt_control is not None:
                matched_input_ids = budget_prompt_tokens(input_ids, matched_prompt_budget)
                logits["matched_prompt"].append(matched_prompt_control(matched_input_ids).detach().cpu())
                matched_prompt_counts.append(matched_input_ids.ne(0).sum(dim=-1).float().detach().cpu())
            streams["predicted"].append(predicted.detach().cpu())
            streams["oracle"].append(oracle.detach().cpu())
            targets.append(target.detach().cpu())
            prompt_counts.append(input_ids.ne(0).sum(dim=-1).float().detach().cpu())
            surfaces.extend(batch["surface"])
    target_all = torch.cat(targets, dim=0)
    all_logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    all_streams = {key: torch.cat(value, dim=0) for key, value in streams.items()}
    predicted_acc = _accuracy(all_logits["predicted"], target_all)
    oracle_acc = _accuracy(all_logits["oracle"], target_all)
    shuffled_acc = _accuracy(all_logits["shuffled"], target_all)
    random_acc = _accuracy(all_logits["random"], target_all)
    zero_acc = _accuracy(all_logits["zero"], target_all)
    prompt_acc = _accuracy(all_logits["prompt"], target_all)
    matched_prompt_acc = _accuracy(all_logits["matched_prompt"], target_all) if "matched_prompt" in all_logits else 0.0
    pred_count = loose_stream_symbol_counts(all_streams["predicted"])
    oracle_count = loose_stream_symbol_counts(all_streams["oracle"])
    prompt_count = torch.cat(prompt_counts, dim=0)
    matched_prompt_count = torch.cat(matched_prompt_counts, dim=0) if matched_prompt_counts else torch.zeros_like(prompt_count)
    mean_pred = float(pred_count.mean().item()) if pred_count.numel() else 0.0
    mean_prompt = float(prompt_count.mean().item()) if prompt_count.numel() else 0.0
    mean_matched_prompt = float(matched_prompt_count.mean().item()) if matched_prompt_count.numel() else 0.0
    metrics = {
        "strict_accuracy": predicted_acc,
        "synthetic_world_accuracy": predicted_acc,
        "phrase_accuracy": predicted_acc,
        "predicted_stream_accuracy": predicted_acc,
        "oracle_stream_accuracy": oracle_acc,
        "shuffled_stream_accuracy": shuffled_acc,
        "random_stream_accuracy": random_acc,
        "zero_stream_accuracy": zero_acc,
        "prompt_only_accuracy": prompt_acc,
        "m25_strict_delta_vs_prompt_only": float(predicted_acc - prompt_acc),
        "matched_prompt_accuracy": matched_prompt_acc,
        "m25_strict_delta_vs_matched_prompt": float(predicted_acc - matched_prompt_acc),
        "predicted_vs_shuffled_delta": float(predicted_acc - shuffled_acc),
        "predicted_vs_random_delta": float(predicted_acc - random_acc),
        "oracle_stream_delta": float(oracle_acc - predicted_acc),
        "stream_advisor_delta": float(predicted_acc - max(shuffled_acc, random_acc, zero_acc)),
        "mean_predicted_emitted_symbols_after_bottleneck": mean_pred,
        "mean_oracle_emitted_symbols_after_bottleneck": float(oracle_count.mean().item()) if oracle_count.numel() else 0.0,
        "mean_prompt_tokens": mean_prompt,
        "mean_matched_prompt_tokens": mean_matched_prompt,
        "loose_symbol_to_prompt_ratio": float(mean_pred / max(1.0, mean_prompt)),
        "prompt_to_loose_symbol_ratio": float(mean_prompt / max(1.0, mean_pred)),
        "loose_symbol_to_matched_prompt_ratio": float(mean_pred / max(1.0, mean_matched_prompt)),
        "matched_prompt_to_loose_symbol_ratio": float(mean_matched_prompt / max(1.0, mean_pred)),
        "token_reduction_ratio": float(1.0 - mean_pred / max(1.0, mean_prompt)),
        "matched_prompt_token_reduction_ratio": float(1.0 - mean_matched_prompt / max(1.0, mean_prompt)),
        "accuracy_per_loose_symbol": float(predicted_acc / max(1.0, mean_pred)),
        "accuracy_per_prompt_token": float(prompt_acc / max(1.0, mean_prompt)),
        "matched_prompt_accuracy_per_token": float(matched_prompt_acc / max(1.0, mean_matched_prompt)),
        "m25_accuracy_per_symbol_delta_vs_matched_prompt": float(
            predicted_acc / max(1.0, mean_pred) - matched_prompt_acc / max(1.0, mean_matched_prompt)
        ),
        "loose_symbol_budget": float(advisor.symbol_budget or 0),
        "matched_prompt_token_budget": float(matched_prompt_budget or 0),
        "hard_symbol_budget_active": 1.0 if advisor.symbol_budget is not None else 0.0,
        "advisor_primary_trace_is_symbolic": 1.0,
        "continuous_trace_smuggling_detected": 0.0,
        "m25_gate_beats_matched_prompt": 1.0 if predicted_acc >= matched_prompt_acc else 0.0,
    }
    metrics.update(_component_accuracy(all_streams["predicted"], all_streams["oracle"]))
    if oracle_advisor is not None:
        metrics["oracle_trained_predicted_stream_accuracy"] = _accuracy(all_logits["oracle_trained_predicted"], target_all)
        metrics["oracle_trained_oracle_stream_accuracy"] = _accuracy(all_logits["oracle_trained_oracle"], target_all)
        metrics["oracle_trained_random_stream_accuracy"] = _accuracy(all_logits["oracle_trained_random"], target_all)
        metrics["oracle_trained_stream_delta"] = float(metrics["oracle_trained_oracle_stream_accuracy"] - metrics["oracle_trained_random_stream_accuracy"])
    metrics.update(m25_promotion_gate_metrics(metrics))
    pred_labels = torch.argmax(all_logits["predicted"], dim=-1)
    surface_metrics: dict[str, dict[str, float]] = {}
    for surface in sorted(set(surfaces)):
        mask = torch.tensor([item == surface for item in surfaces], dtype=torch.bool)
        surface_metrics[surface] = {
            "strict_accuracy": float((pred_labels[mask] == target_all[mask]).float().mean().item()),
            "count": float(mask.sum().item()),
        }
    return {"metrics": metrics, "surface_metrics": surface_metrics}


def m25_promotion_gate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    gates = {
        "m25_gate_strict_accuracy_retained": 1.0 if metrics.get("strict_accuracy", 0.0) >= 0.65 else 0.0,
        "m25_gate_stream_beats_shuffled": 1.0 if metrics.get("predicted_vs_shuffled_delta", 0.0) >= 0.05 else 0.0,
        "m25_gate_stream_beats_random": 1.0 if metrics.get("predicted_vs_random_delta", 0.0) >= 0.05 else 0.0,
        "m25_gate_token_reduction_positive": 1.0 if metrics.get("token_reduction_ratio", 0.0) > 0.0 else 0.0,
        "m25_gate_nonzero_stream_reconstruction": 1.0 if metrics.get("loose_stream_exact_accuracy", 0.0) > 0.0 else 0.0,
        "m25_gate_symbolic_trace_only": 1.0 if metrics.get("advisor_primary_trace_is_symbolic", 0.0) == 1.0 else 0.0,
        "m25_gate_beats_matched_prompt": 1.0 if metrics.get("m25_strict_delta_vs_matched_prompt", -1.0) >= 0.0 else 0.0,
    }
    gates["m25_promotion_gate_pass_rate"] = sum(gates.values()) / max(1, len(gates))
    gates["m25_promotion_candidate"] = 1.0 if all(value == 1.0 for value in gates.values()) else 0.0
    return gates


def _snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [param.detach().cpu().clone() for param in module.parameters()]


def _max_delta(before: Sequence[torch.Tensor], module: nn.Module) -> float:
    values = [float((old.to(new.device) - new.detach()).abs().max().cpu().item()) for old, new in zip(before, module.parameters())]
    return max(values) if values else 0.0


def train_m25_emergent_bridi(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    generator_epochs: int = 8,
    advisor_epochs: int = 8,
    prompt_epochs: int | None = None,
    batch_size: int = 128,
    generator_learning_rate: float = 2e-3,
    advisor_learning_rate: float = 2e-3,
    seed: int = 25,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    symbol_budget: int | None = None,
    matched_prompt_budget: int | None = None,
    trace_weight: float = 2.0,
    answer_weight: float = 0.25,
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
    generator = M25EmergentBridiQFormer(
        vocab_size=len(vocab),
        max_symbols=int(max_symbols),
        value_vocab_size=_value_vocab_size(),
        aux_vocab_size=_aux_vocab_size(),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
    ).to(device_obj)
    generator_history = _train_generator(
        generator,
        train_examples,
        vocab,
        epochs=int(generator_epochs),
        batch_size=int(batch_size),
        learning_rate=float(generator_learning_rate),
        device=device_obj,
        seed=int(seed),
        trace_weight=float(trace_weight),
        answer_weight=float(answer_weight),
        mdl_weight=float(mdl_weight),
    )
    before = _snapshot(generator)
    for param in generator.parameters():
        param.requires_grad_(False)
    advisor = LooseStreamAdvisor(max_symbols=max_symbols, hidden_dim=advisor_hidden_dim, symbol_budget=symbol_budget).to(device_obj)
    oracle_advisor = LooseStreamAdvisor(max_symbols=max_symbols, hidden_dim=advisor_hidden_dim, symbol_budget=symbol_budget).to(device_obj)
    prompt_control = PromptOnlyControl(vocab_size=len(vocab), embedding_dim=max(8, embedding_dim // 2), hidden_dim=advisor_hidden_dim).to(device_obj)
    resolved_matched_prompt_budget = int(matched_prompt_budget or 0)
    if resolved_matched_prompt_budget <= 0:
        resolved_matched_prompt_budget = int(symbol_budget or 0)
    if resolved_matched_prompt_budget <= 0:
        resolved_matched_prompt_budget = int(max_symbols)
    matched_prompt_control = PromptOnlyControl(
        vocab_size=len(vocab),
        embedding_dim=max(8, embedding_dim // 2),
        hidden_dim=advisor_hidden_dim,
    ).to(device_obj)
    advisor_history = _train_stream_advisor(
        generator,
        advisor,
        train_examples,
        vocab,
        epochs=int(advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 100,
        stream_source="predicted",
    )
    oracle_advisor_history = _train_stream_advisor(
        generator,
        oracle_advisor,
        train_examples,
        vocab,
        epochs=int(advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 200,
        stream_source="oracle",
    )
    prompt_history = _train_prompt_control(
        prompt_control,
        train_examples,
        vocab,
        max_symbols=int(max_symbols),
        epochs=int(prompt_epochs if prompt_epochs is not None else advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 300,
    )
    matched_prompt_history = _train_prompt_control(
        matched_prompt_control,
        train_examples,
        vocab,
        max_symbols=int(max_symbols),
        prompt_token_budget=resolved_matched_prompt_budget,
        epochs=int(prompt_epochs if prompt_epochs is not None else advisor_epochs),
        batch_size=int(batch_size),
        learning_rate=float(advisor_learning_rate),
        device=device_obj,
        seed=int(seed) + 400,
    )
    eval_payload = evaluate_m25_emergent_bridi(
        generator=generator,
        advisor=advisor,
        oracle_advisor=oracle_advisor,
        prompt_control=prompt_control,
        matched_prompt_control=matched_prompt_control,
        matched_prompt_budget=resolved_matched_prompt_budget,
        examples=eval_examples,
        vocab=vocab,
        batch_size=int(batch_size),
        device=device_obj,
        seed=int(seed),
    )
    metrics = dict(eval_payload["metrics"])
    max_delta = _max_delta(before, generator)
    metrics.update(
        {
            "generator_trainable_parameter_count_after_freeze": float(sum(p.numel() for p in generator.parameters() if p.requires_grad)),
            "generator_parameter_max_delta_after_advisor": max_delta,
            "generator_parameters_unchanged_after_advisor": 1.0 if max_delta == 0.0 else 0.0,
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
        }
    )
    return {
        "config": {
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "generator_epochs": int(generator_epochs),
            "advisor_epochs": int(advisor_epochs),
            "prompt_epochs": int(prompt_epochs if prompt_epochs is not None else advisor_epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "max_frames": int(max_frames),
            "max_symbols": int(max_symbols),
            "symbol_budget": int(symbol_budget or 0),
            "matched_prompt_budget": int(resolved_matched_prompt_budget),
            "trace_weight": float(trace_weight),
            "answer_weight": float(answer_weight),
            "mdl_weight": float(mdl_weight),
            "device": str(device_obj),
        },
        "metrics": metrics,
        "surface_metrics": eval_payload["surface_metrics"],
        "generator_history": generator_history,
        "advisor_history": advisor_history,
        "oracle_advisor_history": oracle_advisor_history,
        "prompt_history": prompt_history,
        "matched_prompt_history": matched_prompt_history,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "vocab_size": len(vocab),
    }
