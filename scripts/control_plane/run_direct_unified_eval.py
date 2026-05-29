from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.direct_unified_eval import (  # noqa: E402
    DIRECT_UNIFIED_EVAL_OUTPUT_ROOT,
    build_direct_unified_eval_manifest,
    render_direct_unified_eval_markdown,
)
from lojban_evolution.m19.family import M19_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_history_backfill"
DEFAULT_TRACK_BY_FAMILY = {
    "M19": "M19",
    "M20": "M20.1",
    "M21": "M21.1",
    "M22": "M22",
    "M23": "M23",
    "M24": "M24",
    "M25": "M25",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one direct, contract-aware unified eval manifest.")
    parser.add_argument("--family", type=str, default="M19")
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--output-root", type=Path, default=DIRECT_UNIFIED_EVAL_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--benchmark-report", type=Path, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--integrity-report", type=Path, default=None)
    parser.add_argument("--replication-report", type=Path, default=None)
    parser.add_argument("--stability-report", type=Path, default=None)
    parser.add_argument("--kill-test-report", type=Path, default=None)
    parser.add_argument("--dictionary-audit-report", type=Path, default=None)
    parser.add_argument("--m20-suite-report", type=Path, default=None)
    parser.add_argument("--m20-lock-report", type=Path, default=None)
    parser.add_argument("--m20-induction-report", type=Path, default=None)
    parser.add_argument("--m21-suite-report", type=Path, default=None)
    parser.add_argument("--m21-synthetic-assay-report", type=Path, default=None)
    parser.add_argument("--m21-actual-bridge-report", type=Path, default=None)
    parser.add_argument("--m21-lock-report", type=Path, default=None)
    parser.add_argument("--m21-pointer-microgrid-report", type=Path, default=None)
    parser.add_argument("--m21-gauntlet-report", type=Path, default=None)
    parser.add_argument("--m21-adversarial-audit-report", type=Path, default=None)
    parser.add_argument("--m22-generalization-report", type=Path, default=None)
    parser.add_argument("--m23-relevance-report", type=Path, default=None)
    parser.add_argument("--m24-compression-report", type=Path, default=None)
    parser.add_argument("--m25-emergent-report", type=Path, default=None)

    parser.add_argument("--execute-m19-direct", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-model", type=str, default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bridge-path", type=Path, default=Path("artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt"))
    parser.add_argument("--eval-data-path", type=Path, default=Path("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl"))
    parser.add_argument("--audit-data-path", type=Path, default=Path("artifacts/datasets/sanity_check_v1.jsonl"))
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--min-latent-steps", type=int, default=4)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_key = str(args.family).strip().upper()
    if family_key not in {"M19", "M20", "M21", "M22", "M23", "M24", "M25"}:
        raise NotImplementedError("Direct unified eval runner currently supports the M19, M20, M21, M22, M23, M24, and M25 families only.")
    if family_key != "M19" and bool(args.execute_m19_direct):
        raise ValueError("--execute-m19-direct is only valid for family M19.")
    resolved_track = str(args.track).strip() or DEFAULT_TRACK_BY_FAMILY[family_key]

    output_root, output_root_rel = _validated_output_root(args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("direct_unified_eval_%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    validate_series_outputs("M", [output_root_rel], [f"{output_root_rel}/{run_id}"])
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_report = args.benchmark_report
    audit_report = args.audit_report
    integrity_report = args.integrity_report
    if family_key == "M19" and bool(args.execute_m19_direct):
        direct_dir = output_dir / "direct"
        direct_dir.mkdir(parents=True, exist_ok=True)
        benchmark_report, audit_report = _run_m19_direct(args, direct_dir)

    history_manifest = args.history_manifest or _latest_named_manifest(DEFAULT_HISTORY_ROOT, "ablation_history_manifest.json")
    manifest = build_direct_unified_eval_manifest(
        family_key=family_key,
        track=resolved_track,
        benchmark_report_path=benchmark_report,
        audit_report_path=audit_report,
        integrity_report_path=integrity_report,
        replication_report_path=args.replication_report,
        stability_report_path=args.stability_report,
        kill_test_report_path=args.kill_test_report,
        dictionary_audit_report_path=args.dictionary_audit_report,
        m20_suite_report_path=args.m20_suite_report,
        m20_lock_report_path=args.m20_lock_report,
        m20_induction_report_path=args.m20_induction_report,
        m21_suite_report_path=args.m21_suite_report,
        m21_synthetic_assay_report_path=args.m21_synthetic_assay_report,
        m21_actual_bridge_report_path=args.m21_actual_bridge_report,
        m21_lock_report_path=args.m21_lock_report,
        m21_pointer_microgrid_report_path=args.m21_pointer_microgrid_report,
        m21_gauntlet_report_path=args.m21_gauntlet_report,
        m21_adversarial_audit_report_path=args.m21_adversarial_audit_report,
        m22_generalization_report_path=args.m22_generalization_report,
        m23_relevance_report_path=args.m23_relevance_report,
        m24_compression_report_path=args.m24_compression_report,
        m25_emergent_report_path=args.m25_emergent_report,
        history_manifest_path=history_manifest,
    )
    manifest["run_id"] = run_id
    manifest["history_manifest"] = _repo_string(history_manifest) if history_manifest else None
    manifest["config"] = {
        "family": str(args.family),
        "track": resolved_track,
        "execute_m19_direct": bool(args.execute_m19_direct),
        "benchmark_report": _repo_string(benchmark_report) if benchmark_report else None,
        "audit_report": _repo_string(audit_report) if audit_report else None,
        "integrity_report": _repo_string(integrity_report) if integrity_report else None,
        "replication_report": _repo_string(args.replication_report) if args.replication_report else None,
        "stability_report": _repo_string(args.stability_report) if args.stability_report else None,
        "kill_test_report": _repo_string(args.kill_test_report) if args.kill_test_report else None,
        "dictionary_audit_report": _repo_string(args.dictionary_audit_report) if args.dictionary_audit_report else None,
        "m20_suite_report": _repo_string(args.m20_suite_report) if args.m20_suite_report else None,
        "m20_lock_report": _repo_string(args.m20_lock_report) if args.m20_lock_report else None,
        "m20_induction_report": _repo_string(args.m20_induction_report) if args.m20_induction_report else None,
        "m21_suite_report": _repo_string(args.m21_suite_report) if args.m21_suite_report else None,
        "m21_synthetic_assay_report": _repo_string(args.m21_synthetic_assay_report) if args.m21_synthetic_assay_report else None,
        "m21_actual_bridge_report": _repo_string(args.m21_actual_bridge_report) if args.m21_actual_bridge_report else None,
        "m21_lock_report": _repo_string(args.m21_lock_report) if args.m21_lock_report else None,
        "m21_pointer_microgrid_report": _repo_string(args.m21_pointer_microgrid_report) if args.m21_pointer_microgrid_report else None,
        "m21_gauntlet_report": _repo_string(args.m21_gauntlet_report) if args.m21_gauntlet_report else None,
        "m21_adversarial_audit_report": _repo_string(args.m21_adversarial_audit_report) if args.m21_adversarial_audit_report else None,
        "m22_generalization_report": _repo_string(args.m22_generalization_report) if args.m22_generalization_report else None,
        "m23_relevance_report": _repo_string(args.m23_relevance_report) if args.m23_relevance_report else None,
        "m24_compression_report": _repo_string(args.m24_compression_report) if args.m24_compression_report else None,
        "m25_emergent_report": _repo_string(args.m25_emergent_report) if args.m25_emergent_report else None,
    }

    manifest_path = output_dir / "direct_unified_eval_manifest.json"
    summary_path = output_dir / "direct_unified_eval_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(render_direct_unified_eval_markdown(manifest), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")


def _run_m19_direct(args: argparse.Namespace, direct_dir: Path) -> tuple[Path, Path | None]:
    benchmark_report = direct_dir / "m19_benchmark_report.json"
    audit_report = direct_dir / "m19_audit_report.json"
    track_key = str(args.track).strip() if str(args.track).strip() in M19_REGISTRY else "M19"
    benchmark_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(args.bridge_path),
        "--eval-data-path",
        str(args.eval_data_path),
        "--eval-size",
        str(int(args.eval_size)),
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--min-latent-steps",
        str(int(args.min_latent_steps)),
        "--max-latent-steps",
        str(int(args.max_latent_steps)),
        "--random-scale",
        str(float(args.random_scale)),
        "--typed-slot-layout",
        str(args.typed_slot_layout),
        "--arity-router-mode",
        str(args.arity_router_mode),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(args.geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--seed",
        str(int(args.seed)),
        "--track",
        track_key,
        "--cell-id",
        str(args.cell_id),
        "--output-path",
        str(benchmark_report),
    ]
    if track_key == "M19.4":
        benchmark_cmd.append("--dynamic-pacing")
    if bool(args.gumbel_hard):
        benchmark_cmd.append("--gumbel-hard")
    _run(benchmark_cmd)

    if track_key == "M19.4":
        return benchmark_report, None

    audit_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "m19" / "run_m19_audit.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(args.bridge_path),
        "--dataset-path",
        str(args.audit_data_path),
        "--eval-size",
        str(int(args.audit_eval_size)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--random-scale",
        str(float(args.random_scale)),
        "--typed-slot-layout",
        str(args.typed_slot_layout),
        "--arity-router-mode",
        str(args.arity_router_mode),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(args.geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--seed",
        str(int(args.seed)),
        "--track",
        track_key,
        "--cell-id",
        str(args.cell_id),
        "--output-path",
        str(audit_report),
    ]
    if bool(args.gumbel_hard):
        audit_cmd.append("--gumbel-hard")
    _run(audit_cmd)
    return benchmark_report, audit_report


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _validated_output_root(path: Path) -> tuple[Path, str]:
    candidate = Path(path)
    try:
        rel = candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel = str(candidate).replace("\\", "/")
    validated = assert_output_path_allowed("M", rel)
    return REPO_ROOT / validated, validated


def _latest_named_manifest(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    matches = [path for path in root.rglob(file_name) if "__pycache__" not in path.parts]
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def _repo_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    main()
