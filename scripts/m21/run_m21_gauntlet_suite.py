from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m21.family import M21_REGISTRY  # noqa: E402
from lojban_evolution.m21.gauntlet import build_m21_gauntlet_payload  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_gauntlet_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_gauntlet_{_timestamp()}"


def _latest_named_manifest(root: Path, report_name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.rglob(report_name), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    parser = argparse.ArgumentParser(description="Adapt M21 dynamic bridi reports into M19-style integrity/kill/order gauntlet surfaces.")
    parser.add_argument("--suite-report", type=Path, default=None)
    parser.add_argument("--actual-bridge-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["gauntlet"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


def run_gauntlet(args: argparse.Namespace) -> dict:
    registry = M21_REGISTRY["M21"]
    suite_report = args.suite_report or _latest_named_manifest(Path(registry["output_roots"]["suite"]), registry["report_names"]["suite"])
    actual_report = args.actual_bridge_report or _latest_named_manifest(Path(registry["output_roots"]["actual_bridge"]), registry["report_names"]["actual_bridge"])
    if suite_report is None:
        raise FileNotFoundError("No M21 suite report found. Pass --suite-report or run scripts/m21/run_m21_dynamic_bridi_suite.py first.")
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = build_m21_gauntlet_payload(
        suite_payload=_load(suite_report),
        actual_payload=_load(actual_report),
        run_id=run_dir.name,
        suite_report_path=suite_report,
        actual_bridge_report_path=actual_report,
    )
    report_path = run_dir / registry["report_names"]["gauntlet"]
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M21 gauntlet report written to {report_path}")
    print(
        "M21 gauntlet metrics: "
        f"purged={payload['metrics'].get('purged_accuracy', 0.0):.4f} "
        f"format={payload['metrics'].get('format_accuracy', 0.0):.4f} "
        f"renamed={payload['metrics'].get('entity_renamed_accuracy', 0.0):.4f} "
        f"judri_delta={payload['metrics'].get('judri_causal_delta', 0.0):.4f}"
    )
    return payload


if __name__ == "__main__":
    run_gauntlet(parse_args())
