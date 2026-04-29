from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the typed-physics M19 branches under one ledgered suite.")
    parser.add_argument("--track", type=str, default="M19.31")
    parser.add_argument("--base-model", default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", default="artifacts/datasets/m19_mixed_curriculum_v1.jsonl")
    parser.add_argument("--eval-data-path", default="artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl")
    parser.add_argument("--audit-data-path", default="artifacts/datasets/sanity_check_v1.jsonl")
    parser.add_argument("--dictionary-data-path", default="artifacts/datasets/m19_mixed_curriculum_v1.jsonl")
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--dictionary-eval-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--typed-physics-config", type=str, default="configs/m19_typed_physics_ontology.json")
    parser.add_argument("--typed-slot-layout", type=str, default="gismu:2,cmavo:2,judri:4")
    parser.add_argument("--arity-router-mode", type=str, default="")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m19_typed_physics_suite"))
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/models/m19/typed_physics_suite"))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    track_key = str(args.track).strip() if str(args.track).strip() in M19_REGISTRY else "M19.31"
    registry = M19_REGISTRY[track_key]
    defaults = registry.get("defaults", {})
    arity_router_mode = str(args.arity_router_mode).strip() or str(defaults.get("arity_router_mode", "soft"))
    geometry_mode = str(args.geometry_mode).strip() or str(defaults.get("geometry_mode", "euclidean"))
    gumbel_hard = bool(args.gumbel_hard or arity_router_mode == "gumbel_hard")

    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.model_output_root / run_id / f"{track_key.replace('.', '_')}_bridge.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_report_path = run_dir / "train_report.json"
    audit_report_path = run_dir / "audit_report.json"
    benchmark_report_path = run_dir / "benchmark_report.json"
    dictionary_report_path = run_dir / "dictionary_audit_report.json"
    unified_eval_dir = run_dir / "direct_unified_eval"

    eval_bridge_args = [
        "--track",
        track_key,
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--max-latent-steps",
        str(int(args.max_latent_steps)),
        "--typed-slot-layout",
        str(args.typed_slot_layout),
        "--arity-router-mode",
        str(arity_router_mode),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--seed",
        str(int(args.seed)),
    ]
    train_bridge_args = [
        *eval_bridge_args,
        "--typed-physics-config",
        str(args.typed_physics_config),
    ]
    if gumbel_hard:
        eval_bridge_args.append("--gumbel-hard")
        train_bridge_args.append("--gumbel-hard")

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "m19" / "train_m19_mainline.py"),
            "--base-model",
            str(args.base_model),
            "--data-path",
            str(args.data_path),
            "--epochs",
            str(int(args.epochs)),
            "--gumbel-temp-start",
            str(float(args.gumbel_temp_start)),
            "--checkpoint-output-path",
            str(checkpoint_path),
            "--report-output-path",
            str(train_report_path),
            *train_bridge_args,
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
            "--base-model",
            str(args.base_model),
            "--bridge-path",
            str(checkpoint_path),
            "--data-path",
            str(args.audit_data_path),
            "--eval-size",
            str(int(args.audit_eval_size)),
            "--output-path",
            str(audit_report_path),
            *eval_bridge_args,
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
            "--base-model",
            str(args.base_model),
            "--bridge-path",
            str(checkpoint_path),
            "--eval-data-path",
            str(args.eval_data_path),
            "--eval-size",
            str(int(args.eval_size)),
            "--output-path",
            str(benchmark_report_path),
            *eval_bridge_args,
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "m19" / "run_m19_dictionary_audit.py"),
            "--base-model",
            str(args.base_model),
            "--bridge-spec",
            f"{track_key}={checkpoint_path}",
            "--dataset-path",
            str(args.dictionary_data_path),
            "--eval-size",
            str(int(args.dictionary_eval_size)),
            "--output-path",
            str(dictionary_report_path),
            *eval_bridge_args,
        ],
        repo_root,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "control_plane" / "run_direct_unified_eval.py"),
            "--family",
            "M19",
            "--track",
            track_key,
            "--benchmark-report",
            str(benchmark_report_path),
            "--audit-report",
            str(audit_report_path),
            "--dictionary-audit-report",
            str(dictionary_report_path),
            "--output-root",
            str(unified_eval_dir),
            "--run-id",
            "bundle",
        ],
        repo_root,
    )

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": track_key,
        "run_id": run_id,
        "report_paths": {
            "train_report": str(train_report_path).replace("\\", "/"),
            "audit_report": str(audit_report_path).replace("\\", "/"),
            "benchmark_report": str(benchmark_report_path).replace("\\", "/"),
            "dictionary_audit_report": str(dictionary_report_path).replace("\\", "/"),
            "direct_unified_eval_root": str(unified_eval_dir).replace("\\", "/"),
        },
        "config": {
            "typed_physics_config": str(args.typed_physics_config),
            "typed_slot_layout": str(args.typed_slot_layout),
            "arity_router_mode": arity_router_mode,
            "gumbel_hard": gumbel_hard,
            "geometry_mode": geometry_mode,
            "poincare_curvature": float(args.poincare_curvature),
        },
    }
    (run_dir / "m19_typed_physics_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"M19 typed physics suite complete: {run_dir}")


if __name__ == "__main__":
    main()
