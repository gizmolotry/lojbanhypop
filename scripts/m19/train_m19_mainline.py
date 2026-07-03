from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

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
from lojban_evolution.m19.training import (
    build_surface_consistency_variants,
    maybe_apply_surface_augmentations,
)
from lojban_evolution.m19.typed_physics import (
    FAMILY_TO_ID,
    build_typed_targets,
    family_separation_loss,
    load_typed_physics_config,
    mean_entropy_from_logits,
    operator_confidence_cap_loss,
    parse_typed_slot_layout,
    slot_usage_balance_loss,
    symbolic_trace_alignment_score,
)
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


def _materialize_row(batch_dict: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in batch_dict.items():
        if isinstance(value, list):
            row[key] = value[0]
        else:
            row[key] = value
    return row


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


def _build_bridge_prompt(question: str, scratchpad_token: str, target_steps: int) -> tuple[str, str]:
    prompt_core = f"Solve the logic question.\n\nQuestion: {question}\n"
    scratch_tokens = " ".join([str(scratchpad_token)] * int(target_steps))
    return prompt_core, f"{prompt_core}{scratch_tokens}"


def _mean_pool_masked_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    active_mask = labels.ne(-100)
    if not bool(active_mask.any()):
        return logits.new_zeros((logits.shape[0], logits.shape[-1]))
    masked_logits = logits.masked_fill(~active_mask.unsqueeze(-1), 0.0)
    counts = active_mask.sum(dim=1, keepdim=True).clamp_min(1).to(logits.dtype)
    return masked_logits.sum(dim=1) / counts


def _pointer_necessity_contrast_loss(
    full_loss: torch.Tensor,
    ablated_loss: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return torch.relu(full_loss + full_loss.new_tensor(float(margin)) - ablated_loss)


def _forward_surface(
    *,
    backbone: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    bridge: M19SymbioteBridge,
    question: str,
    answer: str,
    target_steps: int,
    args: argparse.Namespace,
    device: str,
    scratchpad_token_id: int,
    lengths: torch.Tensor,
    gumbel_temperature: float,
    typed_slot_layout: list[str],
    slot_family_targets: torch.Tensor | None,
    typed_physics_config: Any,
    mode: str,
    raw_text: str,
    bridge_channel_mode: str = "full",
) -> dict[str, Any]:
    prompt_core, bridge_prompt = _build_bridge_prompt(question, str(args.scratchpad_token), int(target_steps))
    prompt = f"{bridge_prompt} {str(args.symbiote_end_token)}\nFinal answer:"
    full_text = f"{prompt} {answer}"

    inputs_prompt = tokenizer(bridge_prompt, return_tensors="pt").to(device)
    inputs_full = tokenizer(full_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out_prompt = backbone(**inputs_prompt, output_hidden_states=True)
        h_tap = out_prompt.hidden_states[int(args.tap_layer)]

    delta, l_topo, op_logits, telemetry = bridge(
        h_tap,
        active_steps=int(target_steps),
        lengths=lengths,
        gumbel_temperature=gumbel_temperature,
        bridge_channel_mode=bridge_channel_mode,
    )

    scratchpad_mask = inputs_full.input_ids.eq(scratchpad_token_id)
    labels = inputs_full.input_ids.clone()
    labels[0, : len(tokenizer(prompt_core, return_tensors="pt").input_ids[0])] = -100

    with m19_injection_hook(backbone, int(args.tap_layer), scratchpad_mask, delta):
        out_full = backbone(**inputs_full, labels=labels)

    typed_family_loss = torch.zeros((), device=device, dtype=delta.dtype)
    arity_loss = torch.zeros((), device=device, dtype=delta.dtype)
    family_sep_loss = torch.zeros((), device=device, dtype=delta.dtype)
    slot_balance = torch.zeros((), device=device, dtype=delta.dtype)
    operator_balance = torch.zeros((), device=device, dtype=delta.dtype)
    typed_family_accuracy = 0.0
    symbolic_alignment = 0.0
    arity_violation = 0.0
    masked_zero_metric = telemetry.get("masked_pointer_zero_rate")
    masked_zero_rate = float(masked_zero_metric) if masked_zero_metric is not None else 0.0
    hyper_metrics = telemetry.get("hyperbolic_metrics", {})

    if typed_slot_layout and telemetry.get("slot_family_logits") is not None and slot_family_targets is not None:
        slot_family_logits = telemetry["slot_family_logits"][0]
        typed_family_loss = F.cross_entropy(slot_family_logits, slot_family_targets)
        typed_family_accuracy = float((slot_family_logits.argmax(dim=-1) == slot_family_targets).float().mean().item())
        family_sep_loss = family_separation_loss(telemetry["query_state"], typed_slot_layout)
        if telemetry.get("judri_mask") is not None:
            slot_balance = slot_usage_balance_loss(telemetry["judri_mask"])

    targets = (
        build_typed_targets(raw_text=raw_text, mode=mode, config=typed_physics_config, row={"text": raw_text, "mode": mode})
        if typed_physics_config is not None
        else None
    )
    typed_supervision_step = 0
    if targets is not None and targets.has_supervision and telemetry.get("slot_family_logits") is not None:
        typed_supervision_step = 1
        family_hist = torch.tensor(targets.family_histogram, device=device, dtype=delta.dtype)
        symbolic_alignment = float(
            symbolic_trace_alignment_score(telemetry["slot_family_logits"][0], family_hist).detach().item()
        )
        if telemetry.get("arity_logits") is not None and targets.primary_arity is not None:
            arity_target = torch.tensor([int(targets.primary_arity) - 1], device=device, dtype=torch.long)
            arity_logits = telemetry["arity_logits"]
            arity_loss = F.cross_entropy(arity_logits, arity_target)
            arity_prediction = int(arity_logits.argmax(dim=-1)[0].item()) + 1
            arity_violation = 0.0 if arity_prediction == int(targets.primary_arity) else 1.0

    pooled_logits = _mean_pool_masked_logits(out_full.logits.float(), labels)
    operator_balance = operator_confidence_cap_loss(
        op_logits,
        max_top1_share=float(args.operator_top1_cap),
    )

    return {
        "prompt_core": prompt_core,
        "bridge_prompt": bridge_prompt,
        "prompt": prompt,
        "full_text": full_text,
        "inputs_full": inputs_full,
        "labels": labels,
        "delta": delta,
        "l_topo": l_topo,
        "op_logits": op_logits,
        "telemetry": telemetry,
        "out_full": out_full,
        "typed_family_loss": typed_family_loss,
        "arity_loss": arity_loss,
        "family_sep_loss": family_sep_loss,
        "slot_balance": slot_balance,
        "operator_balance": operator_balance,
        "typed_family_accuracy": typed_family_accuracy,
        "symbolic_alignment": symbolic_alignment,
        "arity_violation": arity_violation,
        "masked_zero_rate": masked_zero_rate,
        "hyper_metrics": hyper_metrics,
        "bridge_channel_mode": telemetry.get("bridge_channel_mode", bridge_channel_mode),
        "typed_supervision_step": typed_supervision_step,
        "pooled_logits": pooled_logits,
    }


def train_m19(args: argparse.Namespace) -> dict[str, Any]:
    wandb.init(
        project="lojban-m19",
        name=args.run_id,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )

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
        typed_slot_layout=parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else None,
        geometry_mode=str(args.geometry_mode),
        arity_router_mode=str(args.arity_router_mode),
        gumbel_hard=bool(args.gumbel_hard),
        poincare_curvature=float(args.poincare_curvature),
    ).to(device=device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(args.learning_rate))
    typed_physics_config = load_typed_physics_config(args.typed_physics_config) if str(args.typed_physics_config).strip() else None
    typed_slot_layout = list(bridge.typed_slot_layout)
    slot_family_targets = (
        torch.tensor([FAMILY_TO_ID[family] for family in typed_slot_layout], device=device, dtype=torch.long)
        if typed_slot_layout
        else None
    )

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
        typed_family_sum = 0.0
        arity_sum = 0.0
        family_sep_sum = 0.0
        slot_balance_sum = 0.0
        operator_balance_sum = 0.0
        surface_consistency_sum = 0.0
        pointer_necessity_sum = 0.0
        pointer_necessity_gap_sum = 0.0
        pointer_necessity_active_count = 0
        consistency_variant_count = 0
        typed_accuracy_sum = 0.0
        typed_alignment_sum = 0.0
        typed_violation_sum = 0.0
        masked_zero_sum = 0.0
        family_entropy_sum = 0.0
        hyperbolic_gap_sum = 0.0
        hyperbolic_violation_sum = 0.0
        hyperbolic_geodesic_sum = 0.0
        hyperbolic_clip_sum = 0.0
        typed_supervision_steps = 0
        query_embed_cosine_sum = 0.0
        scratch_trace_cosine_sum = 0.0
        operator_entropy_sum = 0.0
        for batch_dict in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            row = _materialize_row(batch_dict)
            raw_text = str(row["text"])
            mode = str(row["mode"])
            base_question, base_answer = _split_sample(raw_text, mode)
            question, answer, augmentation_flags = maybe_apply_surface_augmentations(
                base_question,
                base_answer,
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
            progress = (total_steps + 1) / max(1, int(args.epochs) * len(loader))
            gumbel_temperature = float(args.gumbel_temp_start) + (
                (float(args.gumbel_temp_end) - float(args.gumbel_temp_start)) * progress
            )
            lengths = torch.tensor([int(target_steps)], device=device, dtype=torch.long)
            primary_surface = _forward_surface(
                backbone=backbone,
                tokenizer=tokenizer,
                bridge=bridge,
                question=question,
                answer=answer,
                target_steps=int(target_steps),
                args=args,
                device=device,
                scratchpad_token_id=scratchpad_token_id,
                lengths=lengths,
                gumbel_temperature=gumbel_temperature,
                typed_slot_layout=typed_slot_layout,
                slot_family_targets=slot_family_targets,
                typed_physics_config=typed_physics_config,
                mode=mode,
                raw_text=raw_text,
            )
            delta = primary_surface["delta"]
            l_topo = primary_surface["l_topo"]
            op_logits = primary_surface["op_logits"]
            telemetry = primary_surface["telemetry"]
            out_full = primary_surface["out_full"]
            bridge.update_halt_centroid(telemetry["trace"], lengths=lengths)
            l_repulse = compute_query_repulsion_loss(
                bridge.query_dictionary(),
                margin=float(args.query_repulsion_margin),
            )
            l_task = out_full.loss
            l_ac = compute_m19_anti_collapse(op_logits, min_entropy=float(args.min_entropy))
            typed_family_loss = primary_surface["typed_family_loss"]
            arity_loss = primary_surface["arity_loss"]
            family_sep_loss = primary_surface["family_sep_loss"]
            slot_balance = primary_surface["slot_balance"]
            operator_balance = primary_surface["operator_balance"]
            typed_family_accuracy = float(primary_surface["typed_family_accuracy"])
            symbolic_alignment = float(primary_surface["symbolic_alignment"])
            arity_violation = float(primary_surface["arity_violation"])
            masked_zero_rate = float(primary_surface["masked_zero_rate"])
            hyper_metrics = primary_surface["hyper_metrics"]
            typed_supervision_steps += int(primary_surface["typed_supervision_step"])

            surface_consistency_loss = torch.zeros((), device=device, dtype=delta.dtype)
            consistency_variants = build_surface_consistency_variants(
                base_question,
                base_answer,
                include_entity_rename=bool(args.surface_consistency_entity_rename),
                include_format_flatten=bool(args.surface_consistency_format_flatten),
                include_combined=bool(args.surface_consistency_combined),
            )
            if float(args.surface_consistency_weight) > 0.0 and consistency_variants:
                variant_losses: list[torch.Tensor] = []
                max_variants = max(1, int(args.surface_consistency_max_variants))
                for variant in consistency_variants[:max_variants]:
                    variant_surface = _forward_surface(
                        backbone=backbone,
                        tokenizer=tokenizer,
                        bridge=bridge,
                        question=str(variant["question"]),
                        answer=str(variant["answer"]),
                        target_steps=int(target_steps),
                        args=args,
                        device=device,
                        scratchpad_token_id=scratchpad_token_id,
                        lengths=lengths,
                        gumbel_temperature=gumbel_temperature,
                        typed_slot_layout=typed_slot_layout,
                        slot_family_targets=slot_family_targets,
                        typed_physics_config=typed_physics_config,
                        mode=mode,
                        raw_text=raw_text,
                    )
                    variant_loss = F.mse_loss(variant_surface["delta"].float(), delta.float())
                    variant_loss = variant_loss + F.mse_loss(
                        variant_surface["pooled_logits"],
                        primary_surface["pooled_logits"],
                    )
                    if float(args.surface_consistency_task_weight) > 0.0:
                        variant_loss = variant_loss + (
                            float(args.surface_consistency_task_weight) * variant_surface["out_full"].loss
                        )
                    if telemetry.get("slot_family_logits") is not None and variant_surface["telemetry"].get("slot_family_logits") is not None:
                        variant_loss = variant_loss + F.mse_loss(
                            variant_surface["telemetry"]["slot_family_logits"].float(),
                            telemetry["slot_family_logits"].float(),
                        )
                    if telemetry.get("judri_mask") is not None and variant_surface["telemetry"].get("judri_mask") is not None:
                        variant_loss = variant_loss + F.mse_loss(
                            variant_surface["telemetry"]["judri_mask"].float(),
                            telemetry["judri_mask"].float(),
                        )
                    variant_losses.append(variant_loss)
                if variant_losses:
                    surface_consistency_loss = torch.stack(variant_losses).mean()
                    consistency_variant_count += len(variant_losses)
            pointer_necessity_loss = torch.zeros((), device=device, dtype=delta.dtype)
            pointer_necessity_gap = 0.0
            if float(args.pointer_necessity_weight) > 0.0 and "judri" in typed_slot_layout:
                ablated_surface = _forward_surface(
                    backbone=backbone,
                    tokenizer=tokenizer,
                    bridge=bridge,
                    question=question,
                    answer=answer,
                    target_steps=int(target_steps),
                    args=args,
                    device=device,
                    scratchpad_token_id=scratchpad_token_id,
                    lengths=lengths,
                    gumbel_temperature=gumbel_temperature,
                    typed_slot_layout=typed_slot_layout,
                    slot_family_targets=slot_family_targets,
                    typed_physics_config=typed_physics_config,
                    mode=mode,
                    raw_text=raw_text,
                    bridge_channel_mode=str(args.pointer_necessity_ablation_mode),
                )
                ablated_task_loss = ablated_surface["out_full"].loss
                pointer_necessity_loss = _pointer_necessity_contrast_loss(
                    l_task,
                    ablated_task_loss,
                    margin=float(args.pointer_necessity_margin),
                )
                pointer_necessity_gap = float((ablated_task_loss.detach() - l_task.detach()).item())
                pointer_necessity_active_count += 1
            loss = (
                l_task
                + float(args.topology_weight) * l_topo
                + float(args.anti_collapse_weight) * l_ac
                + float(args.query_repulsion_weight) * l_repulse
                + float(args.typed_family_weight) * typed_family_loss
                + float(args.typed_arity_weight) * arity_loss
                + float(args.family_separation_weight) * family_sep_loss
                + float(args.slot_usage_balance_weight) * slot_balance
                + float(args.operator_balance_weight) * operator_balance
                + float(args.surface_consistency_weight) * surface_consistency_loss
                + float(args.pointer_necessity_weight) * pointer_necessity_loss
            )
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch/loss": float(loss.detach().item()),
                "batch/task_loss": float(l_task.detach().item()),
                "batch/topo_loss": float(l_topo.detach().item()),
                "batch/typed_family_loss": float(typed_family_loss.detach().item()),
                "batch/arity_loss": float(arity_loss.detach().item()),
                "batch/masked_zero_rate": float(masked_zero_rate)
            }, step=total_steps)

            loss_sum += float(loss.detach().item())
            task_sum += float(l_task.detach().item())
            topo_sum += float(l_topo.detach().item())
            ac_sum += float(l_ac.detach().item())
            repulsion_sum += float(l_repulse.detach().item())
            typed_family_sum += float(typed_family_loss.detach().item())
            arity_sum += float(arity_loss.detach().item())
            family_sep_sum += float(family_sep_loss.detach().item())
            slot_balance_sum += float(slot_balance.detach().item())
            operator_balance_sum += float(operator_balance.detach().item())
            surface_consistency_sum += float(surface_consistency_loss.detach().item())
            pointer_necessity_sum += float(pointer_necessity_loss.detach().item())
            pointer_necessity_gap_sum += float(pointer_necessity_gap)
            typed_accuracy_sum += float(typed_family_accuracy)
            typed_alignment_sum += float(symbolic_alignment)
            typed_violation_sum += float(arity_violation)
            masked_zero_sum += float(masked_zero_rate)
            family_entropy_sum += float(telemetry.get("slot_family_entropy", 0.0))
            hyperbolic_gap_sum += float(hyper_metrics.get("predicate_pointer_radial_gap", 0.0))
            hyperbolic_violation_sum += float(hyper_metrics.get("family_radius_violation_rate", 0.0))
            hyperbolic_geodesic_sum += float(hyper_metrics.get("hyperbolic_geodesic_margin", 0.0))
            hyperbolic_clip_sum += float(hyper_metrics.get("hyperbolic_projection_clip_rate", 0.0))
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
                "mean_typed_family_loss": typed_family_sum / denom,
                "mean_typed_arity_loss": arity_sum / denom,
                "mean_family_separation_loss": family_sep_sum / denom,
                "mean_slot_usage_balance_loss": slot_balance_sum / denom,
                "mean_operator_balance_loss": operator_balance_sum / denom,
                "mean_surface_consistency_loss": surface_consistency_sum / denom,
                "mean_pointer_necessity_loss": pointer_necessity_sum / denom,
                "mean_pointer_necessity_gap": pointer_necessity_gap_sum / max(1, pointer_necessity_active_count),
                "pointer_necessity_active_steps": float(pointer_necessity_active_count),
                "surface_consistency_variant_count": float(consistency_variant_count),
                "mean_target_steps": float(sum(curriculum_targets[-denom:]) / max(1, min(denom, len(curriculum_targets)))),
                "mean_query_embed_pairwise_cosine": query_embed_cosine_sum / denom,
                "mean_scratch_trace_pairwise_cosine": scratch_trace_cosine_sum / denom,
                "mean_operator_entropy_ratio": operator_entropy_sum / denom,
                "typed_family_accuracy": typed_accuracy_sum / denom,
                "arity_violation_rate": (typed_violation_sum / max(1, typed_supervision_steps)) if typed_supervision_steps else 0.0,
                "masked_pointer_zero_rate": masked_zero_sum / denom,
                "family_slot_entropy": family_entropy_sum / denom,
                "symbolic_trace_alignment": (typed_alignment_sum / max(1, typed_supervision_steps)) if typed_supervision_steps else 0.0,
                "predicate_pointer_radial_gap": hyperbolic_gap_sum / denom,
                "family_radius_violation_rate": hyperbolic_violation_sum / denom,
                "hyperbolic_geodesic_margin": hyperbolic_geodesic_sum / denom,
                "hyperbolic_projection_clip_rate": hyperbolic_clip_sum / denom,
                "typed_supervision_steps": float(typed_supervision_steps),
            }
        )
        epoch_log = {f"epoch_mean/{k}": v for k, v in epoch_metrics[-1].items() if k != "epoch"}
        epoch_log["epoch"] = float(epoch + 1)
        wandb.log(epoch_log, step=total_steps)
        print(f"Epoch {epoch + 1} Mean Loss: {epoch_metrics[-1]['mean_loss']:.4f}")
        if bool(args.save_epoch_checkpoints):
            epoch_checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.stem}.epoch_{epoch + 1}{checkpoint_path.suffix}"
            epoch_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bridge.state_dict(), epoch_checkpoint_path)
            epoch_checkpoint_paths.append(str(epoch_checkpoint_path).replace("\\", "/"))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), checkpoint_path)

    print(f"M19 checkpoint saved to {checkpoint_path}")
    wandb.save(str(checkpoint_path))
    wandb.finish()
    return {}


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
    parser.add_argument("--typed-physics-config", type=str, default="")
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--typed-family-weight", type=float, default=0.05)
    parser.add_argument("--typed-arity-weight", type=float, default=0.05)
    parser.add_argument("--family-separation-weight", type=float, default=0.02)
    parser.add_argument("--slot-usage-balance-weight", type=float, default=0.01)
    parser.add_argument("--operator-balance-weight", type=float, default=0.0)
    parser.add_argument("--operator-top1-cap", type=float, default=0.30)
    parser.add_argument("--surface-consistency-weight", type=float, default=0.0)
    parser.add_argument("--surface-consistency-entity-rename", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-format-flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-combined", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-max-variants", type=int, default=2)
    parser.add_argument("--surface-consistency-task-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--pointer-necessity-ablation-mode", type=str, default="no_judri")
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
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
    track_key = _track_key(args.track)
    defaults = M19_REGISTRY.get(track_key, {}).get("defaults", {})
    if defaults:
        if not str(args.typed_physics_config).strip() and defaults.get("typed_physics_config"):
            args.typed_physics_config = str(defaults["typed_physics_config"])
        if not str(args.typed_slot_layout).strip() and defaults.get("typed_slot_layout"):
            args.typed_slot_layout = str(defaults["typed_slot_layout"])
        if str(args.arity_router_mode).strip() == "soft" and defaults.get("arity_router_mode"):
            args.arity_router_mode = str(defaults["arity_router_mode"])
        if str(args.geometry_mode).strip() == "euclidean" and defaults.get("geometry_mode"):
            args.geometry_mode = str(defaults["geometry_mode"])
    return args


if __name__ == "__main__":
    train_m19(parse_args())
