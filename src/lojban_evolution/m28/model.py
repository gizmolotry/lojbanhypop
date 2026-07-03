from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import ANSWER_LABELS, build_vocab, tokenize
from lojban_evolution.m25.emergent_bridi import (
    LOOSE_ARG,
    LOOSE_CLOSE,
    LOOSE_LINK,
    LOOSE_MOD,
    LOOSE_OPEN,
    LOOSE_PAD,
    LOOSE_PRED,
    LOOSE_STOP,
    M25LooseBridiDataset,
    _aux_vocab_size,
    _value_vocab_size,
    generate_m25_emergent_bridi_examples,
    m25_collate,
    pack_loose_stream_from_outputs,
)
from lojban_evolution.m27.runtime import (
    DEFAULT_M27_ANSWER_WEIGHT,
    DEFAULT_M27_MDL_WEIGHT,
    DEFAULT_M27_RELEVANCE_MARGIN,
    DEFAULT_M27_TRACE_WEIGHT,
    M27CoconutBridiRuntime,
    compute_m27_loss,
    evaluate_m27_coconut_bridi_runtime,
    m27_promotion_gate_metrics,
    probe_m27_answer_gradient_flow,
)


TRACE_TYPE_NAMES = {
    LOOSE_PAD: "PAD",
    LOOSE_OPEN: "OPEN",
    LOOSE_PRED: "PRED",
    LOOSE_MOD: "MOD",
    LOOSE_ARG: "ARG",
    LOOSE_LINK: "LINK",
    LOOSE_CLOSE: "CLOSE",
    LOOSE_STOP: "STOP",
}


@dataclass(frozen=True)
class LogebonicSymbioteConfig:
    vocab_size: int
    max_symbols: int = 32
    embedding_dim: int = 64
    hidden_dim: int = 128
    advisor_hidden_dim: int = 64
    symbol_budget: int = 0
    max_prompt_length: int = 128
    language_layers: int = 1
    language_heads: int = 2
    enable_relevance_runtime: bool = True
    relevance_temperature: float = 1.0
    trace_weight: float = DEFAULT_M27_TRACE_WEIGHT
    answer_weight: float = DEFAULT_M27_ANSWER_WEIGHT
    mdl_weight: float = DEFAULT_M27_MDL_WEIGHT
    relevance_rank_weight: float = 0.25
    relevance_margin: float = DEFAULT_M27_RELEVANCE_MARGIN
    use_relevance_answer: bool = True

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "vocab_size": int(self.vocab_size),
            "max_symbols": int(self.max_symbols),
            "value_vocab_size": _value_vocab_size(),
            "aux_vocab_size": _aux_vocab_size(),
            "embedding_dim": int(self.embedding_dim),
            "hidden_dim": int(self.hidden_dim),
            "advisor_hidden_dim": int(self.advisor_hidden_dim),
            "symbol_budget": int(self.symbol_budget) if int(self.symbol_budget) > 0 else None,
            "max_prompt_length": int(self.max_prompt_length),
            "language_layers": int(self.language_layers),
            "language_heads": int(self.language_heads),
            "enable_relevance_runtime": bool(self.enable_relevance_runtime),
            "relevance_temperature": float(self.relevance_temperature),
        }


@dataclass(frozen=True)
class LogebonicSymbioteTrainingResult:
    report_path: Path
    checkpoint_path: Path
    metrics: dict[str, float]
    config: LogebonicSymbioteConfig
    vocab_size: int


class LogebonicSymbioteModel(nn.Module):
    """Checkpointable actual-model wrapper around the M27 recurrent symbiote.

    M27 is the assay substrate. This class is the model boundary: it carries
    the runtime, vocabulary, trace schema, answer labels, checkpoint protocol,
    and prompt-to-answer inference API in one reusable PyTorch object.
    """

    model_family = "M28"
    model_kind = "logebonic_symbiote"

    def __init__(self, config: LogebonicSymbioteConfig, *, vocab: dict[str, int]) -> None:
        super().__init__()
        self.config = config
        self.vocab = dict(vocab)
        self.id_to_token = {int(value): str(key) for key, value in self.vocab.items()}
        self.answer_labels = tuple(ANSWER_LABELS)
        self.core = M27CoconutBridiRuntime(**config.to_model_kwargs())

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[Any],
        *,
        max_symbols: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        advisor_hidden_dim: int = 64,
        enable_relevance_runtime: bool = True,
        **overrides: Any,
    ) -> "LogebonicSymbioteModel":
        vocab = build_vocab(examples)  # type: ignore[arg-type]
        config = LogebonicSymbioteConfig(
            vocab_size=len(vocab),
            max_symbols=int(max_symbols),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            advisor_hidden_dim=int(advisor_hidden_dim),
            enable_relevance_runtime=bool(enable_relevance_runtime),
            **overrides,
        )
        return cls(config, vocab=vocab)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.core(input_ids, **kwargs)

    def encode_prompt(self, prompt: str, *, max_length: int | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        resolved_max = int(max_length or self.config.max_prompt_length)
        ids = [self.vocab.get(tok, self.vocab.get("<unk>", 1)) for tok in tokenize(prompt)[:resolved_max]]
        ids += [0] * (resolved_max - len(ids))
        out = torch.tensor([ids], dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    def emit_trace(self, prompt: str, *, hard: bool = True, max_steps: int | None = None) -> list[dict[str, int | str]]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input_ids = self.encode_prompt(prompt, device=device)
            outputs = self.core(input_ids, mode="hard_free_run" if hard else "soft_train", max_steps=max_steps)
            stream = outputs["hard_trace_tokens"] if hard else pack_loose_stream_from_outputs(outputs)
        if was_training:
            self.train()
        return self.decode_trace(stream[0].detach().cpu())

    def predict(self, prompt: str, *, hard: bool = False) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input_ids = self.encode_prompt(prompt, device=device)
            outputs = self.core(input_ids, mode="hard_free_run" if hard else "soft_train")
            logits_key = "relevance_answer_logits" if "relevance_answer_logits" in outputs else "answer_logits"
            logits = outputs[logits_key]
            probs = torch.softmax(logits, dim=-1)
            answer_id = int(torch.argmax(probs, dim=-1).item())
        if was_training:
            self.train()
        return {
            "answer_id": answer_id,
            "answer_label": self.answer_labels[answer_id],
            "answer_prob": float(probs[0, answer_id].detach().cpu().item()),
            "logits_source": logits_key,
            "trace": self.emit_trace(prompt, hard=True),
        }

    def decode_trace(self, stream: torch.Tensor) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        for position, triple in enumerate(stream.long().tolist()):
            type_id, value_id, aux_id = (int(triple[0]), int(triple[1]), int(triple[2]))
            rows.append(
                {
                    "position": int(position),
                    "type_id": type_id,
                    "type": TRACE_TYPE_NAMES.get(type_id, f"TYPE_{type_id}"),
                    "value_id": value_id,
                    "aux_id": aux_id,
                }
            )
            if type_id in {LOOSE_PAD, LOOSE_STOP}:
                break
        return rows

    def trace_schema(self) -> dict[str, Any]:
        return {
            "format": "loose_bridi_triple_stream",
            "triple_fields": ["type_id", "value_id", "aux_id"],
            "type_vocabulary": {str(key): value for key, value in TRACE_TYPE_NAMES.items()},
            "max_symbols": int(self.config.max_symbols),
            "value_vocab_size": _value_vocab_size(),
            "aux_vocab_size": _aux_vocab_size(),
            "answer_labels": list(self.answer_labels),
        }

    def checkpoint_payload(self, *, metrics: dict[str, Any] | None = None, training_state: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format": "logebonic_symbiote_checkpoint_v1",
            "family": self.model_family,
            "model_kind": self.model_kind,
            "config": asdict(self.config),
            "vocab": self.vocab,
            "answer_labels": list(self.answer_labels),
            "trace_schema": self.trace_schema(),
            "state_dict": self.state_dict(),
            "metrics": dict(metrics or {}),
            "training_state": dict(training_state or {}),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metrics: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metrics=metrics, training_state=training_state), resolved)
        return resolved

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "LogebonicSymbioteModel":
        payload = torch.load(Path(path), map_location=map_location)
        if payload.get("format") != "logebonic_symbiote_checkpoint_v1":
            raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")
        config = LogebonicSymbioteConfig(**payload["config"])
        model = cls(config, vocab=payload["vocab"])
        model.load_state_dict(payload["state_dict"])
        return model


def load_logebonic_symbiote_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> LogebonicSymbioteModel:
    return LogebonicSymbioteModel.load_checkpoint(path, map_location=map_location)


def _safe_run_id(run_id: str | None) -> str:
    raw = (run_id or "m28_logebonic_model_smoke").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or "m28_logebonic_model_smoke"


def train_logebonic_symbiote_model(
    *,
    train_size: int = 6000,
    eval_size: int = 1500,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 28,
    max_frames: int = 6,
    max_symbols: int = 32,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    symbol_budget: int = 0,
    enable_relevance_runtime: bool = True,
    relevance_rank_weight: float = 0.25,
    use_relevance_answer: bool = True,
    run_baselines: bool = False,
    baseline_epochs: int = 2,
    resume_checkpoint: str | Path | None = None,
    checkpoint_every_epochs: int = 0,
    use_amp: bool = False,
    device: str | torch.device = "cpu",
    output_root: str | Path = "artifacts/runs/telemetry/raw/ablation/hypercube/m28_logebonic_symbiote_model",
    run_id: str | None = None,
) -> LogebonicSymbioteTrainingResult:
    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    train_examples = generate_m25_emergent_bridi_examples(
        int(train_size),
        seed=int(seed),
        max_frames=int(max_frames),
        max_symbols=int(max_symbols),
    )
    eval_examples = generate_m25_emergent_bridi_examples(
        int(eval_size),
        seed=int(seed) + 999,
        max_frames=int(max_frames),
        max_symbols=int(max_symbols),
    )
    run_dir = Path(output_root) / _safe_run_id(run_id)
    resume_payload: dict[str, Any] | None = None
    start_epoch = 0
    history: list[dict[str, float]] = []
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_payload = torch.load(resume_path, map_location=device_obj)
        model = LogebonicSymbioteModel.load_checkpoint(resume_path, map_location=device_obj).to(device_obj)
        training_state = resume_payload.get("training_state", {}) if isinstance(resume_payload, dict) else {}
        start_epoch = int(training_state.get("epoch", 0) or 0)
        raw_history = training_state.get("history", [])
        if isinstance(raw_history, list):
            history = [dict(row) for row in raw_history if isinstance(row, dict)]
    else:
        model = LogebonicSymbioteModel.from_examples(
            [*train_examples, *eval_examples],
            max_symbols=int(max_symbols),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            advisor_hidden_dim=int(advisor_hidden_dim),
            symbol_budget=int(symbol_budget),
            enable_relevance_runtime=bool(enable_relevance_runtime),
            relevance_rank_weight=float(relevance_rank_weight),
            use_relevance_answer=bool(use_relevance_answer),
        ).to(device_obj)
    dataset = M25LooseBridiDataset(train_examples, model.vocab, max_symbols=model.config.max_symbols)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed) + start_epoch),
        collate_fn=m25_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    if resume_payload is not None:
        optimizer_state = resume_payload.get("training_state", {}).get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
    amp_enabled = bool(use_amp) and device_obj.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resume_payload is not None:
        scaler_state = resume_payload.get("training_state", {}).get("scaler_state_dict")
        if amp_enabled and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
    model.train()
    for epoch in range(start_epoch, int(epochs)):
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device_obj)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                teacher_outputs = model(input_ids, teacher_trace=batch["stream_targets"].to(device_obj))
                soft_outputs = model(input_ids)
                loss, metrics = compute_m27_loss(
                    teacher_outputs,
                    batch,
                    answer_outputs=soft_outputs,
                    trace_weight=model.config.trace_weight,
                    answer_weight=model.config.answer_weight,
                    mdl_weight=model.config.mdl_weight,
                    relevance_rank_weight=model.config.relevance_rank_weight,
                    relevance_margin=model.config.relevance_margin,
                    use_relevance_answer=model.config.use_relevance_answer,
                )
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
        if int(checkpoint_every_epochs) > 0 and (epoch + 1) % int(checkpoint_every_epochs) == 0:
            model.save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}_logebonic_symbiote.pt",
                metrics=history[-1],
                training_state={
                    "epoch": int(epoch + 1),
                    "history": history,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if amp_enabled else {},
                    "amp_enabled": bool(amp_enabled),
                },
            )

    eval_payload = evaluate_m27_coconut_bridi_runtime(
        model=model.core,
        examples=eval_examples,
        vocab=model.vocab,
        batch_size=int(batch_size),
        device=device_obj,
        seed=int(seed),
    )
    probe_batch = m25_collate([M25LooseBridiDataset(eval_examples, model.vocab, max_symbols=model.config.max_symbols)[i] for i in range(min(8, len(eval_examples)))])
    metrics = dict(eval_payload["metrics"])
    metrics.update(probe_m27_answer_gradient_flow(model.core, probe_batch).as_dict())
    metrics.update(
        {
            "m28_actual_model_artifact": 1.0,
            "checkpoint_roundtrip_required": 1.0,
            "model_inference_api_present": 1.0,
            "trace_schema_saved": 1.0,
            "m27_training_answer_loss_uses_soft_free_run_trace": 1.0,
            "m27_training_trace_loss_uses_teacher_forcing": 1.0,
            "answer_loss_uses_soft_free_run_trace": 1.0,
            "trace_loss_uses_teacher_forcing": 1.0,
            "m28_trace_causality_delta": float(metrics.get("predicted_vs_zero_delta", 0.0) or 0.0),
        }
    )
    baseline_payload: dict[str, Any] = {
        "baseline_results": {},
        "summary": {"m28_baseline_comparison_bundle_present": 0.0},
    }
    if bool(run_baselines):
        from lojban_evolution.m28.baselines import run_m28_baseline_bundle

        baseline_payload = run_m28_baseline_bundle(
            learned_model=model,
            train_examples=train_examples,
            eval_examples=eval_examples,
            epochs=int(baseline_epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            embedding_dim=max(8, int(embedding_dim)),
            hidden_dim=max(16, int(advisor_hidden_dim)),
            seed=int(seed),
            max_symbols=int(max_symbols),
            device=device_obj,
        )
    metrics.update({key: float(value) for key, value in baseline_payload.get("summary", {}).items() if isinstance(value, (int, float))})
    metrics.update(m27_promotion_gate_metrics(metrics))
    checkpoint_path = run_dir / "checkpoints" / "final_logebonic_symbiote.pt"
    model.save_checkpoint(
        checkpoint_path,
        metrics=metrics,
        training_state={
            "epoch": int(epochs),
            "history": history,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else {},
            "amp_enabled": bool(amp_enabled),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
        },
    )
    reloaded = LogebonicSymbioteModel.load_checkpoint(checkpoint_path, map_location=device_obj).to(device_obj)
    metrics["checkpoint_roundtrip_pass"] = 1.0 if reloaded.config == model.config and reloaded.vocab == model.vocab else 0.0
    sample_prediction = reloaded.predict(eval_examples[0].prompt) if eval_examples else {}
    metrics["model_inference_api_pass"] = 1.0 if sample_prediction.get("trace") else 0.0
    model.save_checkpoint(
        checkpoint_path,
        metrics=metrics,
        training_state={
            "epoch": int(epochs),
            "history": history,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else {},
            "amp_enabled": bool(amp_enabled),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
        },
    )
    report_path = run_dir / "m28_logebonic_model_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "track": "M28",
        "run_id": run_dir.name,
        "config": asdict(model.config),
        "training_config": {
            "requested_epochs": int(epochs),
            "start_epoch": int(start_epoch),
            "checkpoint_every_epochs": int(checkpoint_every_epochs),
            "use_amp_requested": bool(use_amp),
            "amp_enabled": bool(amp_enabled),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
        },
        "checkpoint_path": str(checkpoint_path),
        "vocab_size": len(model.vocab),
        "trace_schema": model.trace_schema(),
        "metrics": metrics,
        "baseline_comparison": baseline_payload,
        "history": history,
        "surface_metrics": eval_payload.get("surface_metrics", {}),
        "sample_prediction": sample_prediction,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return LogebonicSymbioteTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        config=model.config,
        vocab_size=len(model.vocab),
    )
