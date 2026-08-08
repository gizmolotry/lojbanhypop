from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lojban_evolution.m21.bridi import ANSWER_LABELS, CMAVO, GISMU, tokenize
from lojban_evolution.m24.compression import PromptOnlyControl, _accuracy
from lojban_evolution.m25.emergent_bridi import (
    LOOSE_ARG,
    LOOSE_CLOSE,
    LOOSE_LINK,
    LOOSE_MOD,
    LOOSE_OPEN,
    LOOSE_PAD,
    LOOSE_PRED,
    LOOSE_STOP,
    M25EmergentBridiExample,
    LooseBridiSymbol,
)


TEXT_BASELINES = (
    "no_cot",
    "full_english_cot",
    "short_english_cot",
    "full_chinese_cot",
    "short_chinese_cot",
    "hand_bridi_trace",
    "random_discrete_code",
)
ALL_BASELINES = (*TEXT_BASELINES, "pure_latent_bottleneck", "learned_logebonic_trace")


@dataclass(frozen=True)
class M28TextBaselineExample:
    prompt: str
    answer_id: int
    answer_label: str
    trace_token_count: int


class M28TextBaselineDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[M28TextBaselineExample],
        vocab: dict[str, int],
        *,
        max_length: int = 128,
    ) -> None:
        self.examples = list(examples)
        self.vocab = dict(vocab)
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        ids = [self.vocab.get(tok, self.vocab.get("<unk>", 1)) for tok in tokenize(row.prompt)[: self.max_length]]
        ids += [0] * (self.max_length - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "answer_id": torch.tensor(int(row.answer_id), dtype=torch.long),
            "trace_token_count": torch.tensor(float(row.trace_token_count), dtype=torch.float32),
        }


def m28_text_collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch]),
        "answer_id": torch.stack([row["answer_id"] for row in batch]),
        "trace_token_count": torch.stack([row["trace_token_count"] for row in batch]),
    }


class LatentBottleneckControl(nn.Module):
    """Prompt-only continuous bottleneck baseline with no symbolic trace."""

    def __init__(self, *, vocab_size: int, embedding_dim: int = 32, bottleneck_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
        self.to_bottleneck = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(bottleneck_dim)),
            nn.Tanh(),
        )
        self.answer_head = nn.Sequential(
            nn.Linear(int(bottleneck_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), len(ANSWER_LABELS)),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(input_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.answer_head(self.to_bottleneck(pooled))


def build_m28_text_vocab(examples: Sequence[M28TextBaselineExample]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for row in examples:
        for token in tokenize(row.prompt):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def trace_to_hand_bridi(symbols: Sequence[LooseBridiSymbol]) -> str:
    parts: list[str] = []
    for symbol in symbols:
        type_id = int(symbol.type_id)
        if type_id == LOOSE_PAD:
            break
        if type_id == LOOSE_STOP:
            parts.append("STOP")
            break
        if type_id == LOOSE_OPEN:
            parts.append(f"OPEN_F{symbol.value_id}")
        elif type_id == LOOSE_PRED:
            gismu = GISMU[int(symbol.value_id) % len(GISMU)]
            parts.append(f"PRED_{gismu}")
        elif type_id == LOOSE_MOD:
            cmavo = CMAVO[int(symbol.value_id) % len(CMAVO)]
            parts.append(f"MOD_{cmavo}")
        elif type_id == LOOSE_ARG:
            parts.append(f"ARG_E{symbol.value_id}_P{symbol.aux_id}")
        elif type_id == LOOSE_LINK:
            parts.append(f"LINK_{symbol.value_id}_F{symbol.aux_id}")
        elif type_id == LOOSE_CLOSE:
            parts.append(f"CLOSE_F{symbol.value_id}")
    return " ".join(parts)


def trace_to_full_english_cot(row: M25EmergentBridiExample) -> str:
    clauses: list[str] = []
    for frame_idx, frame in enumerate(row.frames):
        if frame.stop:
            break
        if not frame.active:
            continue
        gismu = GISMU[int(frame.gismu_id) % len(GISMU)]
        modifiers = [CMAVO[int(cmavo) % len(CMAVO)] for cmavo in frame.cmavo_ids if int(cmavo) > 0]
        args = [f"place_{idx + 1}_entity_{entity}" for idx, entity in enumerate(frame.judri_place_bindings) if int(entity) > 0]
        clauses.append(
            "frame "
            + str(frame_idx + 1)
            + " predicate "
            + gismu
            + (" modifiers " + " ".join(modifiers) if modifiers else "")
            + (" arguments " + " ".join(args) if args else "")
        )
    return " reasoning trace " + " ; ".join(clauses)


def trace_to_short_english_cot(row: M25EmergentBridiExample) -> str:
    chunks: list[str] = []
    for frame in row.frames:
        if frame.stop:
            break
        if not frame.active:
            continue
        gismu = GISMU[int(frame.gismu_id) % len(GISMU)]
        mods = ".".join(CMAVO[int(cmavo) % len(CMAVO)] for cmavo in frame.cmavo_ids if int(cmavo) > 0)
        active_args = sum(1 for entity in frame.judri_place_bindings if int(entity) > 0)
        chunks.append(f"{gismu}:{mods or 'none'}:{active_args}args")
    return " short trace " + " ".join(chunks)


def trace_to_full_chinese_cot(row: M25EmergentBridiExample) -> str:
    clauses: list[str] = []
    for frame_idx, frame in enumerate(row.frames):
        if frame.stop:
            break
        if not frame.active:
            continue
        gismu = GISMU[int(frame.gismu_id) % len(GISMU)]
        modifiers = [CMAVO[int(cmavo) % len(CMAVO)] for cmavo in frame.cmavo_ids if int(cmavo) > 0]
        args = [f"位置_{idx + 1}_实体_{entity}" for idx, entity in enumerate(frame.judri_place_bindings) if int(entity) > 0]
        clauses.append(
            "框架 "
            + str(frame_idx + 1)
            + " 谓词 "
            + gismu
            + (" 修饰符 " + " ".join(modifiers) if modifiers else "")
            + (" 参数 " + " ".join(args) if args else "")
        )
    return " 推理轨迹 " + " ; ".join(clauses)


def trace_to_short_chinese_cot(row: M25EmergentBridiExample) -> str:
    chunks: list[str] = []
    for frame in row.frames:
        if frame.stop:
            break
        if not frame.active:
            continue
        gismu = GISMU[int(frame.gismu_id) % len(GISMU)]
        mods = ".".join(CMAVO[int(cmavo) % len(CMAVO)] for cmavo in frame.cmavo_ids if int(cmavo) > 0)
        active_args = sum(1 for entity in frame.judri_place_bindings if int(entity) > 0)
        chunks.append(f"{gismu}:{mods or '无'}:{active_args}参")
    return " 短轨迹 " + " ".join(chunks)


def random_code_for_example(row: M25EmergentBridiExample, *, seed: int, max_symbols: int) -> str:
    rng = random.Random(f"{int(seed)}:{row.counterfactual_group}:{row.entity_signature}:{row.prompt}")
    count = max(1, min(int(max_symbols), len(row.loose_symbols)))
    return " random code " + " ".join(f"R{rng.randrange(0, 2048)}" for _ in range(count))


def build_m28_baseline_examples(
    examples: Sequence[M25EmergentBridiExample],
    *,
    baseline: str,
    seed: int = 0,
    max_symbols: int = 32,
) -> list[M28TextBaselineExample]:
    baseline_key = str(baseline)
    out: list[M28TextBaselineExample] = []
    for row in examples:
        if baseline_key == "no_cot":
            text = row.prompt
            trace_tokens = 0
        elif baseline_key == "full_english_cot":
            cot = trace_to_full_english_cot(row)
            text = row.prompt + cot
            trace_tokens = len(tokenize(cot))
        elif baseline_key == "short_english_cot":
            cot = trace_to_short_english_cot(row)
            text = row.prompt + cot
            trace_tokens = len(tokenize(cot))
        elif baseline_key == "full_chinese_cot":
            cot = trace_to_full_chinese_cot(row)
            text = row.prompt + cot
            trace_tokens = len(tokenize(cot))
        elif baseline_key == "short_chinese_cot":
            cot = trace_to_short_chinese_cot(row)
            text = row.prompt + cot
            trace_tokens = len(tokenize(cot))
        elif baseline_key == "hand_bridi_trace":
            trace = " hand bridi " + trace_to_hand_bridi(row.loose_symbols)
            text = row.prompt + trace
            trace_tokens = len(tokenize(trace))
        elif baseline_key == "random_discrete_code":
            trace = random_code_for_example(row, seed=int(seed), max_symbols=int(max_symbols))
            text = row.prompt + trace
            trace_tokens = len(tokenize(trace))
        else:
            raise ValueError(f"Unsupported text baseline: {baseline!r}")
        out.append(
            M28TextBaselineExample(
                prompt=text,
                answer_id=int(row.answer_id),
                answer_label=row.answer_label,
                trace_token_count=int(trace_tokens),
            )
        )
    return out


def _train_classifier(
    model: nn.Module,
    train_dataset: M28TextBaselineDataset,
    eval_dataset: M28TextBaselineDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        collate_fn=m28_text_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    model.to(device)
    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["answer_id"].to(device)
            loss = F.cross_entropy(model(input_ids), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    trace_counts: list[torch.Tensor] = []
    with torch.no_grad():
        eval_loader = DataLoader(eval_dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m28_text_collate)
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            logits.append(model(input_ids).detach().cpu())
            targets.append(batch["answer_id"].detach().cpu())
            token_counts.append(input_ids.ne(0).float().sum(dim=-1).detach().cpu())
            trace_counts.append(batch["trace_token_count"].detach().cpu())
    all_logits = torch.cat(logits, dim=0)
    all_targets = torch.cat(targets, dim=0)
    avg_tokens = float(torch.cat(token_counts, dim=0).mean().item()) if token_counts else 0.0
    avg_trace_tokens = float(torch.cat(trace_counts, dim=0).mean().item()) if trace_counts else 0.0
    strict_accuracy = _accuracy(all_logits, all_targets)
    return {
        "strict_accuracy": strict_accuracy,
        "avg_tokens": avg_tokens,
        "avg_trace_tokens": avg_trace_tokens,
        "accuracy_per_token": strict_accuracy / max(1.0, avg_tokens),
        "accuracy_per_trace_token": strict_accuracy / max(1.0, avg_trace_tokens),
    }


def run_m28_baseline_bundle(
    *,
    learned_model: Any,
    train_examples: Sequence[M25EmergentBridiExample],
    eval_examples: Sequence[M25EmergentBridiExample],
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    latent_bottleneck_dim: int = 8,
    seed: int = 28,
    max_symbols: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device_obj = torch.device(device)
    results: dict[str, dict[str, float]] = {}
    for baseline in TEXT_BASELINES:
        train_rows = build_m28_baseline_examples(train_examples, baseline=baseline, seed=int(seed), max_symbols=int(max_symbols))
        eval_rows = build_m28_baseline_examples(eval_examples, baseline=baseline, seed=int(seed), max_symbols=int(max_symbols))
        vocab = build_m28_text_vocab([*train_rows, *eval_rows])
        model = PromptOnlyControl(vocab_size=len(vocab), embedding_dim=int(embedding_dim), hidden_dim=int(hidden_dim))
        results[baseline] = _train_classifier(
            model,
            M28TextBaselineDataset(train_rows, vocab),
            M28TextBaselineDataset(eval_rows, vocab),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            device=device_obj,
            seed=int(seed),
        )
    latent_train = build_m28_baseline_examples(train_examples, baseline="no_cot", seed=int(seed), max_symbols=int(max_symbols))
    latent_eval = build_m28_baseline_examples(eval_examples, baseline="no_cot", seed=int(seed), max_symbols=int(max_symbols))
    latent_vocab = build_m28_text_vocab([*latent_train, *latent_eval])
    latent = LatentBottleneckControl(
        vocab_size=len(latent_vocab),
        embedding_dim=int(embedding_dim),
        bottleneck_dim=int(latent_bottleneck_dim),
        hidden_dim=int(hidden_dim),
    )
    results["pure_latent_bottleneck"] = _train_classifier(
        latent,
        M28TextBaselineDataset(latent_train, latent_vocab),
        M28TextBaselineDataset(latent_eval, latent_vocab),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        device=device_obj,
        seed=int(seed) + 101,
    )
    learned_metrics = _evaluate_learned_logebonic_trace(
        learned_model,
        eval_examples,
        batch_size=int(batch_size),
        device=device_obj,
    )
    results["learned_logebonic_trace"] = learned_metrics
    learned_acc = learned_metrics["strict_accuracy"]
    best_non_learned = max(value["strict_accuracy"] for key, value in results.items() if key != "learned_logebonic_trace")
    prompt_acc = results["no_cot"]["strict_accuracy"]
    return {
        "baseline_results": results,
        "summary": {
            "m28_baseline_comparison_bundle_present": 1.0,
            "m28_baseline_count": float(len(results)),
            "m28_learned_logebonic_accuracy": learned_acc,
            "m28_best_non_logebonic_baseline_accuracy": best_non_learned,
            "m28_learned_vs_best_baseline_delta": learned_acc - best_non_learned,
            "m28_learned_vs_no_cot_delta": learned_acc - prompt_acc,
            "m28_learned_accuracy_per_trace_token": learned_metrics["accuracy_per_trace_token"],
            "m28_learned_trace_token_count": learned_metrics["avg_trace_tokens"],
        },
    }


def _evaluate_learned_logebonic_trace(
    model: Any,
    examples: Sequence[M25EmergentBridiExample],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    from lojban_evolution.m29.runtime import evaluate_m29_star_runtime
    from lojban_evolution.m21.bridi import build_vocab

    payload = evaluate_m29_star_runtime(
        model=model,
        examples=examples,
        vocab=model.vocab if hasattr(model, 'vocab') else build_vocab(examples),
        batch_size=int(batch_size),
        device=device,
        seed=0,
    )
    metrics = payload["metrics"]
    strict_accuracy = float(metrics.get("strict_accuracy", 0.0) or 0.0)
    trace_tokens = float(metrics.get("trace_token_count", model.bridge.num_queries) or 0.0)
    prompt_tokens = float(metrics.get("mean_prompt_tokens", 0.0) or 0.0)
    return {
        "strict_accuracy": strict_accuracy,
        "avg_tokens": prompt_tokens + trace_tokens,
        "avg_trace_tokens": trace_tokens,
        "accuracy_per_token": strict_accuracy / max(1.0, prompt_tokens + trace_tokens),
        "accuracy_per_trace_token": strict_accuracy / max(1.0, trace_tokens),
        "zero_trace_accuracy": float(metrics.get("topology_corrupted_accuracy", 0.0) or 0.0),
        "trace_causality_delta": float(metrics.get("trace_causality_delta", 0.0) or 0.0),
    }
