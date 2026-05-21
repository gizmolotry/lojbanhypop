from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m22.family import M22_REGISTRY  # noqa: E402
from lojban_evolution.m22.generalization import build_m22_semantic_generalization_payload  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m22_semantic_generalization_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m22_semantic_generalization_{_timestamp()}"


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M22_REGISTRY["M22"]
    parser = argparse.ArgumentParser(description="Build the M22 semantic coverage generalization report over M21 controls.")
    parser.add_argument("--suite-report", type=Path, required=True)
    parser.add_argument("--adversarial-audit-report", type=Path, default=None)
    parser.add_argument("--m21-control-direct-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["generalization"]))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--min-semantic-delta", type=float, default=0.02)
    parser.add_argument("--max-clean-drop", type=float, default=0.02)
    parser.add_argument("--min-judri-delta", type=float, default=0.70)
    return parser.parse_args(argv)


def run_generalization(args: argparse.Namespace) -> dict[str, Any]:
    registry = M22_REGISTRY["M22"]
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = build_m22_semantic_generalization_payload(
        suite_payload=_read(args.suite_report),
        adversarial_payload=_read(args.adversarial_audit_report),
        control_manifest_payload=_read(args.m21_control_direct_manifest),
        run_id=run_dir.name,
        suite_report_path=args.suite_report,
        adversarial_audit_report_path=args.adversarial_audit_report,
        control_manifest_path=args.m21_control_direct_manifest,
        min_semantic_delta=float(args.min_semantic_delta),
        max_clean_drop=float(args.max_clean_drop),
        min_judri_delta=float(args.min_judri_delta),
    )
    report_path = run_dir / registry["report_names"]["generalization"]
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    print(f"M22 semantic generalization report written to {report_path}")
    print(
        "M22 metrics: "
        f"strict={float(metrics.get('strict_accuracy', 0.0)):.4f} "
        f"semantic={float(metrics.get('semantic_coverage_strict_accuracy', 0.0)):.4f} "
        f"semantic_delta={float(metrics.get('m22_semantic_strict_delta_vs_m21_control', 0.0)):.4f} "
        f"promotion={float(metrics.get('m22_promotion_candidate', 0.0)):.0f}"
    )
    return payload


if __name__ == "__main__":
    run_generalization(parse_args())
