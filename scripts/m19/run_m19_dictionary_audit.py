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
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / args.run_id
    report_path = Path(args.output_path) if args.output_path else (run_dir / M19_REGISTRY["M19"]["report_names"]["dictionary_audit"])
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
        ).to(device=device, dtype=model_dtype)
        bridge_load = bridge.load_state_dict(bridge_state, strict=False)
        bridge.eval()

        lengths = torch.tensor([int(args.scratchpad_length)], device=device, dtype=torch.long)
        telemetry_rows: list[dict[str, float]] = []
        top1_counts: dict[int, int] = {}
        valid_positions = 0

        for item in tqdm(samples, desc=f"dictionary-audit:{label}"):
            prompt = _sample_prompt(str(item["prompt"]), str(args.scratchpad_token), int(args.scratchpad_length))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_prompt = backbone(**inputs, output_hidden_states=True)
                h_tap = out_prompt.hidden_states[int(args.tap_layer)]
                _, _, op_logits, telemetry = bridge(h_tap, active_steps=int(args.scratchpad_length), lengths=lengths)

            telemetry_rows.append({key: float(value) for key, value in telemetry.get("dictionary_health", {}).items()})
            top1 = op_logits.argmax(dim=-1)[0, : int(args.scratchpad_length)].tolist()
            valid_positions += len(top1)
            for op_id in top1:
                top1_counts[int(op_id)] = top1_counts.get(int(op_id), 0) + 1

        query_embed_stats = pairwise_cosine_stats(bridge.query_embeds.detach())
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
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["dictionary_audit"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run_m19_dictionary_audit(parse_args())
