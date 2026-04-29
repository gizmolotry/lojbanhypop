from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from lojban_evolution.m19.engine import (
    M19SymbioteBridge,
    compute_m19_anti_collapse,
    compute_query_repulsion_loss,
    ensure_special_tokens,
    m19_injection_hook,
)
from lojban_evolution.m19.family import (
    M19_FAMILY_VERSION,
    M19_HIDDEN_SIZE,
    M19_REGISTRY,
    M19_SCRATCHPAD_TOKEN,
    M19_SYMBIOTE_END_TOKEN,
)
from lojban_evolution.m19.training import maybe_apply_surface_augmentations
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


class M19Dataset(Dataset):
    def __init__(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _dtype_for_runtime(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _track_key(track: str) -> str:
    candidate = str(track or "").strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _split_sample(raw_text: str, mode: str) -> tuple[str, str]:
    if mode == "crystal":
        question = raw_text.split("QUESTION:")[1].split("TRACE:")[0].strip()
        answer = raw_text.split("ANSWER:")[1].strip()
    else:
        question = raw_text.split("Question:")[1].split("Final answer:")[0].strip()
        answer = raw_text.split("Final answer:")[1].strip()
    return question, answer


def _default_run_dir(output_root: Path, run_id: str) -> Path:
    return output_root / run_id


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    registry = M19_REGISTRY[_track_key(args.track)]
    output_root = Path(args.output_root)
    default_root = Path(M19_REGISTRY["M19"]["output_roots"]["train"])
    if _track_key(args.track) != "M19" and output_root == default_root:
        output_root = Path(registry["output_roots"]["train"])
    assert_output_path_allowed("M", output_root)
    run_dir = _default_run_dir(output_root, args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else (run_dir / "bridge_final.pt")
    report_path = Path(args.report_output) if args.report_output else (run_dir / registry["report_names"]["train"])
    assert_output_path_allowed("M", report_path)
    if not str(checkpoint_path).strip():
        raise ValueError("checkpoint_output cannot be empty")
    if ".." in checkpoint_path.parts:
        raise ValueError("checkpoint_output may not include path traversal")
    return run_dir, checkpoint_path, report_path


def _dynamic_mode_enabled(args: argparse.Namespace) -> bool:
    return bool(args.dynamic_pacing or _track_key(args.track) == "M19.4")


def _heuristic_target_steps(question: str, answer: str, min_steps: int, max_steps: int) -> int:
    question_tokens = question.split()
    answer_tokens = answer.split()
    punctuation_weight = sum(question.count(ch) for ch in ("?", ",", ";", ":"))
    connective_weight = sum(question.lower().count(word) for word in (" and ", " or ", " if ", " then ", " while "))
    complexity = len(question_tokens) + len(answer_tokens) + punctuation_weight + (2 * connective_weight)
    ladder = [4, 8, 12, 16, 24, 32, 48, 64]
    allowed = [step for step in ladder if min_steps <= step <= max_steps]
    if not allowed:
        return max(min_steps, min(max_steps, 4))
    idx = min(len(allowed) - 1, max(0, complexity // 8))
    return int(allowed[idx])


def train_m19(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = _dtype_for_runtime(device)
    _set_seed(int(args.seed))

    run_dir, checkpoint_path, report_path = _resolve_paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    track_key = _track_key(args.track)
    registry = M19_REGISTRY[track_key]
    dynamic_mode = _dynamic_mode_enabled(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if device != "cuda":
        backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    token_ids = ensure_special_tokens(
        backbone,
        tokenizer,
        [str(args.scratchpad_token), str(args.symbiote_end_token)],
    )
    scratchpad_token_id = int(token_ids[str(args.scratchpad_token)])
    symbiote_end_token_id = int(token_ids[str(args.symbiote_end_token)])

    bridge = M19SymbioteBridge(
        hidden_size=int(args.hidden_size),
        bottleneck_dim=int(args.bottleneck_dim),
        scratchpad_len=int(args.scratchpad_length),
        num_queries=int(args.num_queries),
        max_latent_steps=int(args.max_latent_steps),
    ).to(device=device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(args.learning_rate))

    dataset = M19Dataset(Path(args.data_path))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    epoch_metrics: list[dict[str, float]] = []
    epoch_checkpoint_paths: list[str] = []
    total_steps = 0
    curriculum_targets: list[int] = []
    renamed_steps = 0
    format_flattened_steps = 0

    print(
        f"--- M19 TRAINING: track={track_key} cell={args.cell_id} q={args.num_queries} d={args.bottleneck_dim} "
        f"s={args.scratchpad_length} max={args.max_latent_steps} dynamic={dynamic_mode} epochs={args.epochs} ---"
    )

    for epoch in range(int(args.epochs)):
        bridge.train()
        loss_sum = 0.0
        task_sum = 0.0
        topo_sum = 0.0
        ac_sum = 0.0
        repulsion_sum = 0.0
        query_embed_cosine_sum = 0.0
        scratch_trace_cosine_sum = 0.0
        operator_entropy_sum = 0.0
        for batch_dict in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            raw_text = str(batch_dict["text"][0])
            mode = str(batch_dict["mode"][0])
            question, answer = _split_sample(raw_text, mode)
            question, answer, augmentation_flags = maybe_apply_surface_augmentations(
                question,
                answer,
                entity_rename_probability=float(args.entity_rename_augmentation_prob),
                format_flatten_probability=float(args.format_flatten_augmentation_prob),
            )
            if augmentation_flags.get("entity_renamed"):
                renamed_steps += 1
            if augmentation_flags.get("format_flattened"):
                format_flattened_steps += 1

            if dynamic_mode:
                target_steps = _heuristic_target_steps(
                    question,
                    answer,
                    min_steps=int(args.min_latent_steps),
                    max_steps=int(args.max_latent_steps),
                )
            else:
                target_steps = int(args.scratchpad_length)
            curriculum_targets.append(int(target_steps))
            scratch_tokens = " ".join([str(args.scratchpad_token)] * int(target_steps))
            prompt_core = f"Solve the logic question.\n\nQuestion: {question}\n"
            bridge_prompt = f"{prompt_core}{scratch_tokens}"
            prompt = f"{bridge_prompt} {str(args.symbiote_end_token)}\nFinal answer:"
            full_text = f"{prompt} {answer}"

            inputs_prompt = tokenizer(bridge_prompt, return_tensors="pt").to(device)
            inputs_full = tokenizer(full_text, return_tensors="pt").to(device)

            with torch.no_grad():
                out_prompt = backbone(**inputs_prompt, output_hidden_states=True)
                h_tap = out_prompt.hidden_states[int(args.tap_layer)]

            lengths = torch.tensor([int(target_steps)], device=device, dtype=torch.long)
            delta, l_topo, op_logits, telemetry = bridge(h_tap, active_steps=int(target_steps), lengths=lengths)
            bridge.update_halt_centroid(telemetry["trace"], lengths=lengths)
            l_repulse = compute_query_repulsion_loss(
                bridge.query_embeds,
                margin=float(args.query_repulsion_margin),
            )
            scratchpad_mask = inputs_full.input_ids.eq(scratchpad_token_id)
            labels = inputs_full.input_ids.clone()
            labels[0, : len(tokenizer(prompt_core, return_tensors="pt").input_ids[0])] = -100

            with m19_injection_hook(backbone, int(args.tap_layer), scratchpad_mask, delta):
                out_full = backbone(**inputs_full, labels=labels)

            l_task = out_full.loss
            l_ac = compute_m19_anti_collapse(op_logits, min_entropy=float(args.min_entropy))
            loss = (
                l_task
                + float(args.topology_weight) * l_topo
                + float(args.anti_collapse_weight) * l_ac
                + float(args.query_repulsion_weight) * l_repulse
            )
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().item())
            task_sum += float(l_task.detach().item())
            topo_sum += float(l_topo.detach().item())
            ac_sum += float(l_ac.detach().item())
            repulsion_sum += float(l_repulse.detach().item())
            health = telemetry.get("dictionary_health", {})
            query_embed_cosine_sum += float(health.get("query_embed_pairwise_cosine_mean", 0.0))
            scratch_trace_cosine_sum += float(health.get("scratch_trace_pairwise_cosine_mean", 0.0))
            operator_entropy_sum += float(health.get("operator_operator_entropy_ratio_mean", 0.0))
            total_steps += 1

        denom = max(1, len(loader))
        epoch_metrics.append(
            {
                "epoch": float(epoch + 1),
                "mean_loss": loss_sum / denom,
                "mean_task_loss": task_sum / denom,
                "mean_topology_loss": topo_sum / denom,
                "mean_anti_collapse_loss": ac_sum / denom,
                "mean_query_repulsion_loss": repulsion_sum / denom,
                "mean_target_steps": float(sum(curriculum_targets[-denom:]) / max(1, min(denom, len(curriculum_targets)))),
                "mean_query_embed_pairwise_cosine": query_embed_cosine_sum / denom,
                "mean_scratch_trace_pairwise_cosine": scratch_trace_cosine_sum / denom,
                "mean_operator_entropy_ratio": operator_entropy_sum / denom,
            }
        )
        print(f"Epoch {epoch + 1} Mean Loss: {epoch_metrics[-1]['mean_loss']:.4f}")
        if bool(args.save_epoch_checkpoints):
            epoch_checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.stem}.epoch_{epoch + 1}{checkpoint_path.suffix}"
            epoch_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bridge.state_dict(), epoch_checkpoint_path)
            epoch_checkpoint_paths.append(str(epoch_checkpoint_path).replace("\\", "/"))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), checkpoint_path)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.train", "scripts/m19/train_m19_mainline.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=str(checkpoint_path).replace("\\", "/"),
            dataset_profile="m19_mixed_curriculum_v1",
            difficulty_tier="mixed",
        ),
        "track": "M19.train",
        "family_contract": {
            "family_version": M19_FAMILY_VERSION,
            "family_name": registry["family"],
            "implementation_label": registry["implementation_label"],
            "runner_script": registry["runner_scripts"]["train"],
            "dag": registry["dags"]["train"],
        },
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path).replace("\\", "/"),
            "cell_id": str(args.cell_id),
            "track": track_key,
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "min_latent_steps": int(args.min_latent_steps),
            "max_latent_steps": int(args.max_latent_steps),
            "scratchpad_token": str(args.scratchpad_token),
            "symbiote_end_token": str(args.symbiote_end_token),
            "dynamic_pacing": dynamic_mode,
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "tap_layer": int(args.tap_layer),
            "hidden_size": int(args.hidden_size),
            "topology_weight": float(args.topology_weight),
            "anti_collapse_weight": float(args.anti_collapse_weight),
            "min_entropy": float(args.min_entropy),
            "query_repulsion_weight": float(args.query_repulsion_weight),
            "query_repulsion_margin": float(args.query_repulsion_margin),
            "local_files_only": bool(args.local_files_only),
            "entity_rename_augmentation_prob": float(args.entity_rename_augmentation_prob),
            "format_flatten_augmentation_prob": float(args.format_flatten_augmentation_prob),
            "save_epoch_checkpoints": bool(args.save_epoch_checkpoints),
        },
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "checkpoint_output_path": str(checkpoint_path).replace("\\", "/"),
        "epoch_checkpoint_paths": epoch_checkpoint_paths,
        "report_path": str(report_path).replace("\\", "/"),
        "scratchpad_token_id": int(scratchpad_token_id),
        "symbiote_end_token_id": int(symbiote_end_token_id),
        "dataset_size": len(dataset),
        "total_steps": int(total_steps),
        "halt_centroid_norm": float(bridge.halt_centroid.norm().item()),
        "halt_centroid_samples": float(bridge.halt_centroid_samples.item()),
        "curriculum_target_steps": curriculum_targets[: min(len(curriculum_targets), 256)],
        "curriculum_target_step_mean": (sum(curriculum_targets) / max(1, len(curriculum_targets))),
        "entity_rename_augmented_steps": int(renamed_steps),
        "entity_rename_augmented_rate": (float(renamed_steps) / max(1, int(total_steps))),
        "format_flatten_augmented_steps": int(format_flattened_steps),
        "format_flatten_augmented_rate": (float(format_flattened_steps) / max(1, int(total_steps))),
        "epoch_metrics": epoch_metrics,
        "final_metrics": epoch_metrics[-1] if epoch_metrics else {},
        "epoch_mean_losses": [float(row["mean_loss"]) for row in epoch_metrics],
        "final_mean_loss": float(epoch_metrics[-1]["mean_loss"]) if epoch_metrics else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"M19 checkpoint saved to {checkpoint_path}")
    print(f"M19 train manifest written to {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Train the M19 q-former symbiote runway bridge.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", type=Path, default=Path(registry["dataset_defaults"]["train"]))
    parser.add_argument("--cell-id", type=str, default="M19.MAINLINE")
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--min-latent-steps", type=int, default=4)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--topology-weight", type=float, default=0.1)
    parser.add_argument("--anti-collapse-weight", type=float, default=0.05)
    parser.add_argument("--min-entropy", type=float, default=0.85)
    parser.add_argument("--query-repulsion-weight", type=float, default=0.0)
    parser.add_argument("--query-repulsion-margin", type=float, default=0.15)
    parser.add_argument("--scratchpad-token", type=str, default=M19_SCRATCHPAD_TOKEN)
    parser.add_argument("--symbiote-end-token", type=str, default=M19_SYMBIOTE_END_TOKEN)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["train"]))
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--checkpoint-output-path", type=Path, default=None)
    parser.add_argument("--report-output-path", type=Path, default=None)
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--dynamic-pacing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--entity-rename-augmentation-prob", type=float, default=0.0)
    parser.add_argument("--format-flatten-augmentation-prob", type=float, default=0.0)
    parser.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.checkpoint_output is None and args.checkpoint_output_path is not None:
        args.checkpoint_output = args.checkpoint_output_path
    if args.report_output is None and args.report_output_path is not None:
        args.report_output = args.report_output_path
    return args


if __name__ == "__main__":
    train_m19(parse_args())
