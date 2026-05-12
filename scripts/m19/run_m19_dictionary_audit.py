from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from lojban_evolution.m19.engine import (
    M19SymbioteBridge,
    ensure_special_tokens,
    pairwise_cosine_stats,
)
from lojban_evolution.m19.family import (
    M19_FAMILY_VERSION,
    M19_HIDDEN_SIZE,
    M19_REGISTRY,
    M19_SCRATCHPAD_TOKEN,
    M19_SYMBIOTE_END_TOKEN,
)
from lojban_evolution.m19.typed_physics import (
    build_typed_targets,
    load_typed_physics_config,
    parse_typed_slot_layout,
    symbolic_trace_alignment_score,
)
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


def _dtype_for_runtime(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_report_path(args: argparse.Namespace) -> Path:
    registry = M19_REGISTRY[str(args.track).strip()] if str(args.track).strip() in M19_REGISTRY else M19_REGISTRY["M19"]
    output_root = Path(args.output_root)
    if output_root == Path(M19_REGISTRY["M19"]["output_roots"]["dictionary_audit"]) and "dictionary_audit" in registry.get("output_roots", {}):
        output_root = Path(registry["output_roots"]["dictionary_audit"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / args.run_id
    report_name = registry.get("report_names", {}).get("dictionary_audit", M19_REGISTRY["M19"]["report_names"]["dictionary_audit"])
    report_path = Path(args.output_path) if args.output_path else (run_dir / report_name)
    validate_series_outputs("M", [output_root], [run_dir])
    assert_output_path_allowed("M", report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _load_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[: int(limit)]


def _infer_max_latent_steps(bridge_state: dict[str, Any], configured_steps: int) -> int:
    configured = max(1, int(configured_steps))
    candidates: list[int] = []
    for key in ("output_map.bias", "collar.spatial_embeddings"):
        tensor = bridge_state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            candidates.append(int(tensor.shape[0]))
    return max([configured, *candidates])


def _parse_checkpoint_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for idx, raw_spec in enumerate(specs):
        label, sep, path_value = str(raw_spec).partition("=")
        if sep:
            parsed.append((label.strip(), Path(path_value.strip())))
        else:
            candidate = Path(label.strip())
            parsed.append((f"checkpoint_{idx + 1}", candidate))
    if not parsed:
        raise ValueError("At least one --bridge-spec value is required.")
    return parsed


def _sample_prompt(prompt: str, scratchpad_token: str, scratchpad_length: int) -> str:
    scratch = " ".join([scratchpad_token] * int(scratchpad_length))
    return f"Solve the logic question.\n\nQuestion: {prompt}\n{scratch}"


def _prompt_from_row(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    raw_text = str(row.get("text", ""))
    mode = str(row.get("mode", "")).lower()
    if mode == "crystal" and "QUESTION:" in raw_text:
        return raw_text.split("QUESTION:", 1)[1].split("TRACE:", 1)[0].strip()
    if "Question:" in raw_text:
        return raw_text.split("Question:", 1)[1].split("Final answer:", 1)[0].strip()
    return raw_text.strip()


def _mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in metric_rows) / max(1, len(metric_rows)))
        for key in keys
    }


def run_m19_dictionary_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = _dtype_for_runtime(device)
    _set_seed(int(args.seed))

    report_path = _resolve_report_path(args)
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
    ensure_special_tokens(backbone, tokenizer, [str(args.scratchpad_token), str(args.symbiote_end_token)])
    typed_slot_layout = parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else []
    typed_physics_config = load_typed_physics_config(args.typed_physics_config) if str(args.typed_physics_config).strip() else None

    samples = _load_dataset(Path(args.dataset_path), int(args.eval_size))
    checkpoint_specs = _parse_checkpoint_specs(list(args.bridge_spec))
    checkpoint_rows: list[dict[str, Any]] = []

    for label, bridge_path in checkpoint_specs:
        bridge_state = torch.load(bridge_path, map_location=device)
        max_latent_steps = _infer_max_latent_steps(bridge_state, int(args.max_latent_steps))
        bridge = M19SymbioteBridge(
            hidden_size=int(args.hidden_size),
            bottleneck_dim=int(args.bottleneck_dim),
            scratchpad_len=int(args.scratchpad_length),
            num_queries=int(args.num_queries),
            max_latent_steps=max_latent_steps,
            typed_slot_layout=typed_slot_layout if typed_slot_layout else None,
            geometry_mode=str(args.geometry_mode),
            arity_router_mode=str(args.arity_router_mode),
            gumbel_hard=bool(args.gumbel_hard),
            poincare_curvature=float(args.poincare_curvature),
        ).to(device=device, dtype=model_dtype)
        bridge_load = bridge.load_state_dict(bridge_state, strict=False)
        bridge.eval()

        lengths = torch.tensor([int(args.scratchpad_length)], device=device, dtype=torch.long)
        telemetry_rows: list[dict[str, float]] = []
        top1_counts: dict[int, int] = {}
        valid_positions = 0
        typed_family_acc_values: list[float] = []
        arity_violation_values: list[float] = []
        masked_zero_values: list[float] = []
        family_entropy_values: list[float] = []
        symbolic_alignment_values: list[float] = []
        radial_gap_values: list[float] = []
        radial_violation_values: list[float] = []
        geodesic_margin_values: list[float] = []
        clip_rate_values: list[float] = []
        typed_supervision_steps = 0

        for item in tqdm(samples, desc=f"dictionary-audit:{label}"):
            prompt = _sample_prompt(_prompt_from_row(item), str(args.scratchpad_token), int(args.scratchpad_length))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_prompt = backbone(**inputs, output_hidden_states=True)
                h_tap = out_prompt.hidden_states[int(args.tap_layer)]
                _, _, op_logits, telemetry = bridge(
                    h_tap,
                    active_steps=int(args.scratchpad_length),
                    lengths=lengths,
                    gumbel_temperature=float(args.gumbel_temp_end),
                )

            telemetry_rows.append({key: float(value) for key, value in telemetry.get("dictionary_health", {}).items()})
            slot_family_ids = telemetry.get("slot_family_ids") or []
            slot_family_logits = telemetry.get("slot_family_logits")
            if slot_family_logits is not None and slot_family_ids:
                family_targets = torch.tensor(slot_family_ids, device=slot_family_logits.device, dtype=torch.long)
                typed_family_acc_values.append(
                    float((slot_family_logits[0].argmax(dim=-1) == family_targets).float().mean().item())
                )
            masked_pointer_zero_rate = telemetry.get("masked_pointer_zero_rate")
            if masked_pointer_zero_rate is not None:
                masked_zero_values.append(float(masked_pointer_zero_rate))
            family_entropy_values.append(float(telemetry.get("slot_family_entropy", 0.0)))
            hyper_metrics = telemetry.get("hyperbolic_metrics", {})
            radial_gap_values.append(float(hyper_metrics.get("predicate_pointer_radial_gap", 0.0)))
            radial_violation_values.append(float(hyper_metrics.get("family_radius_violation_rate", 0.0)))
            geodesic_margin_values.append(float(hyper_metrics.get("hyperbolic_geodesic_margin", 0.0)))
            clip_rate_values.append(float(hyper_metrics.get("hyperbolic_projection_clip_rate", 0.0)))
            if typed_physics_config is not None:
                mode = str(item.get("mode", ""))
                raw_text = str(item.get("text", ""))
                targets = build_typed_targets(raw_text=raw_text, mode=mode, config=typed_physics_config, row=item)
                if targets.has_supervision:
                    typed_supervision_steps += 1
                    family_hist = torch.tensor(targets.family_histogram, device=h_tap.device, dtype=h_tap.dtype)
                    if slot_family_logits is not None:
                        symbolic_alignment_values.append(
                            float(symbolic_trace_alignment_score(slot_family_logits[0], family_hist).detach().item())
                        )
                    if telemetry.get("arity_logits") is not None and targets.primary_arity is not None:
                        arity_prediction = int(telemetry["arity_logits"].argmax(dim=-1)[0].item()) + 1
                        arity_violation_values.append(0.0 if arity_prediction == int(targets.primary_arity) else 1.0)
            top1 = op_logits.argmax(dim=-1)[0, : int(args.scratchpad_length)].tolist()
            valid_positions += len(top1)
            for op_id in top1:
                top1_counts[int(op_id)] = top1_counts.get(int(op_id), 0) + 1

        query_embed_stats = pairwise_cosine_stats(bridge.query_dictionary().detach())
        aggregate = _mean_metrics(telemetry_rows)
        dominant_share = (max(top1_counts.values()) / max(1, valid_positions)) if top1_counts else 0.0
        unique_top1 = len(top1_counts)
        checkpoint_rows.append(
            {
                "label": label,
                "bridge_path": str(bridge_path).replace("\\", "/"),
                "checkpoint_missing_keys": list(bridge_load.missing_keys),
                "checkpoint_unexpected_keys": list(bridge_load.unexpected_keys),
                "max_latent_steps": int(max_latent_steps),
                "query_embed_stats": query_embed_stats,
                "aggregate_dictionary_health": aggregate,
                "operator_top1_unique_count": int(unique_top1),
                "operator_top1_dominant_share": float(dominant_share),
                "typed_faithfulness": {
                    "typed_family_accuracy": (sum(typed_family_acc_values) / max(1, len(typed_family_acc_values))) if typed_family_acc_values else 0.0,
                    "arity_violation_rate": (sum(arity_violation_values) / max(1, len(arity_violation_values))) if arity_violation_values else 0.0,
                    "masked_pointer_zero_rate": (sum(masked_zero_values) / max(1, len(masked_zero_values))) if masked_zero_values else None,
                    "family_slot_entropy": (sum(family_entropy_values) / max(1, len(family_entropy_values))) if family_entropy_values else 0.0,
                    "symbolic_trace_alignment": (sum(symbolic_alignment_values) / max(1, len(symbolic_alignment_values))) if symbolic_alignment_values else 0.0,
                    "predicate_pointer_radial_gap": (sum(radial_gap_values) / max(1, len(radial_gap_values))) if radial_gap_values else 0.0,
                    "family_radius_violation_rate": (sum(radial_violation_values) / max(1, len(radial_violation_values))) if radial_violation_values else 0.0,
                    "hyperbolic_geodesic_margin": (sum(geodesic_margin_values) / max(1, len(geodesic_margin_values))) if geodesic_margin_values else 0.0,
                    "hyperbolic_projection_clip_rate": (sum(clip_rate_values) / max(1, len(clip_rate_values))) if clip_rate_values else 0.0,
                    "typed_supervision_steps": int(typed_supervision_steps),
                },
            }
        )

    comparisons: list[dict[str, Any]] = []
    if len(checkpoint_rows) >= 2:
        base = checkpoint_rows[0]
        for row in checkpoint_rows[1:]:
            comparisons.append(
                {
                    "lhs": base["label"],
                    "rhs": row["label"],
                    "query_embed_pairwise_cosine_mean_gap": float(
                        row["query_embed_stats"]["pairwise_cosine_mean"] - base["query_embed_stats"]["pairwise_cosine_mean"]
                    ),
                    "query_embed_pairwise_cosine_max_gap": float(
                        row["query_embed_stats"]["pairwise_cosine_max"] - base["query_embed_stats"]["pairwise_cosine_max"]
                    ),
                    "query_embed_anisotropy_gap": float(
                        row["query_embed_stats"]["anisotropy"] - base["query_embed_stats"]["anisotropy"]
                    ),
                    "scratch_trace_pairwise_cosine_mean_gap": float(
                        row["aggregate_dictionary_health"].get("scratch_trace_pairwise_cosine_mean", 0.0)
                        - base["aggregate_dictionary_health"].get("scratch_trace_pairwise_cosine_mean", 0.0)
                    ),
                    "scratch_trace_pairwise_cosine_max_gap": float(
                        row["aggregate_dictionary_health"].get("scratch_trace_pairwise_cosine_max", 0.0)
                        - base["aggregate_dictionary_health"].get("scratch_trace_pairwise_cosine_max", 0.0)
                    ),
                    "operator_entropy_ratio_gap": float(
                        row["aggregate_dictionary_health"].get("operator_operator_entropy_ratio_mean", 0.0)
                        - base["aggregate_dictionary_health"].get("operator_operator_entropy_ratio_mean", 0.0)
                    ),
                    "operator_top1_dominant_share_gap": float(
                        row["operator_top1_dominant_share"] - base["operator_top1_dominant_share"]
                    ),
                    "typed_family_accuracy_gap": float(
                        row["typed_faithfulness"].get("typed_family_accuracy", 0.0)
                        - base["typed_faithfulness"].get("typed_family_accuracy", 0.0)
                    ),
                    "arity_violation_gap": float(
                        row["typed_faithfulness"].get("arity_violation_rate", 0.0)
                        - base["typed_faithfulness"].get("arity_violation_rate", 0.0)
                    ),
                }
            )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.dictionary_audit", "scripts/m19/run_m19_dictionary_audit.py"),
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=[str(path).replace("\\", "/") for _, path in checkpoint_specs],
            checkpoint_out=None,
            dataset_profile=Path(args.dataset_path).stem,
            difficulty_tier="dictionary_audit",
        ),
        "track": "M19.dictionary_audit",
        "family_contract": {
            "family_version": M19_FAMILY_VERSION,
            "runner_script": M19_REGISTRY["M19"]["runner_scripts"]["dictionary_audit"],
            "dag": M19_REGISTRY["M19"]["dags"]["dictionary_audit"],
        },
        "config": {
            "base_model": str(args.base_model),
            "dataset_path": str(args.dataset_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "scratchpad_length": int(args.scratchpad_length),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "hidden_size": int(args.hidden_size),
            "tap_layer": int(args.tap_layer),
            "seed": int(args.seed),
            "scratchpad_token": str(args.scratchpad_token),
            "symbiote_end_token": str(args.symbiote_end_token),
            "typed_physics_config": str(args.typed_physics_config).replace("\\", "/") if str(args.typed_physics_config).strip() else None,
            "typed_slot_layout": typed_slot_layout,
            "arity_router_mode": str(args.arity_router_mode),
            "gumbel_hard": bool(args.gumbel_hard),
            "gumbel_temp_end": float(args.gumbel_temp_end),
            "geometry_mode": str(args.geometry_mode),
            "poincare_curvature": float(args.poincare_curvature),
            "bridge_specs": [
                {"label": label, "bridge_path": str(path).replace("\\", "/")}
                for label, path in checkpoint_specs
            ],
        },
        "checkpoints": checkpoint_rows,
        "comparisons": comparisons,
        "report_path": str(report_path).replace("\\", "/"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"M19 dictionary audit report written to {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Compare M19 bridge dictionary health across checkpoints.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-spec", action="append", required=True, help="label=checkpoint_path")
    parser.add_argument("--dataset-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--scratchpad-token", type=str, default=M19_SCRATCHPAD_TOKEN)
    parser.add_argument("--symbiote-end-token", type=str, default=M19_SYMBIOTE_END_TOKEN)
    parser.add_argument("--typed-physics-config", type=str, default="")
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["dictionary_audit"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    track_key = str(args.track).strip() if str(args.track).strip() in M19_REGISTRY else "M19"
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
        if not args.gumbel_hard and str(defaults.get("arity_router_mode", "")).strip() == "gumbel_hard":
            args.gumbel_hard = True
    return args


if __name__ == "__main__":
    run_m19_dictionary_audit(parse_args())
