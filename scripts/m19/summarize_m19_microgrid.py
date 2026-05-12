from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final or partial M19 stability microgrid artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_microgrid_run(Path(args.run_dir))
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote microgrid summary: {args.output_path}")
    if bool(args.pretty):
        print(render_summary_text(summary))


def summarize_microgrid_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    final_report_path = run_dir / "m19_stability_microgrid_report.json"
    final_report = _read_json(final_report_path) if final_report_path.exists() else {}
    combo_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir()) if run_dir.exists() else []
    combo_rows = []
    for combo_dir in combo_dirs:
        combo_rows.append(_summarize_combo_dir(combo_dir))
    final_rows = final_report.get("grid_rows", []) if isinstance(final_report.get("grid_rows"), list) else []
    merged_rows = _merge_final_rows(combo_rows, final_rows)
    completed = [row for row in merged_rows if row.get("replication_report_exists")]
    incomplete = [row for row in merged_rows if not row.get("replication_report_exists")]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir).replace("\\", "/"),
        "final_report_exists": final_report_path.exists(),
        "final_report_path": str(final_report_path).replace("\\", "/"),
        "headline": final_report.get("headline", {}) if isinstance(final_report.get("headline"), dict) else {},
        "completed_combo_count": len(completed),
        "incomplete_combo_count": len(incomplete),
        "combo_rows": merged_rows,
    }


def render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "M19 Microgrid Summary",
        f"- run_dir: {summary.get('run_dir')}",
        f"- final_report_exists: {summary.get('final_report_exists')}",
        f"- completed_combo_count: {summary.get('completed_combo_count')}",
        f"- incomplete_combo_count: {summary.get('incomplete_combo_count')}",
    ]
    headline = summary.get("headline", {}) if isinstance(summary.get("headline"), dict) else {}
    if headline:
        lines.append(f"- best_mean_accuracy_config: {headline.get('best_mean_accuracy_config')}")
        lines.append(f"- best_mean_accuracy: {headline.get('best_mean_accuracy')}")
        lines.append(f"- promotion_gate_pass_count: {headline.get('promotion_gate_pass_count')}")
    lines.append("")
    lines.append("| combo | ptr | acc | seed29 | avg_tokens | runway_tokens | pointer_loss | pointer_gap | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary.get("combo_rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {combo} | {ptr} | {acc} | {seed29} | {tokens} | {runway} | {ploss} | {pgap} | {status} |".format(
                combo=row.get("combo_slug"),
                ptr=_fmt(row.get("pointer_necessity_weight")),
                acc=_fmt(row.get("mean_accuracy")),
                seed29=_fmt(row.get("seed_29_accuracy")),
                tokens=_fmt(row.get("mean_avg_tokens")),
                runway=_fmt(row.get("mean_avg_runway_tokens")),
                ploss=_fmt(row.get("mean_pointer_necessity_loss")),
                pgap=_fmt(row.get("mean_pointer_necessity_gap")),
                status="complete" if row.get("replication_report_exists") else "pending",
            )
        )
    return "\n".join(lines)


def _summarize_combo_dir(combo_dir: Path) -> dict[str, Any]:
    report_path = combo_dir / "m19_replication_report.json"
    progress_path = _find_progress_report(combo_dir)
    report = _read_json(report_path) if report_path.exists() else {}
    progress = _read_json(progress_path) if progress_path.exists() else {}
    payload = report if report else progress
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    seed_runs = payload.get("seed_runs", []) if isinstance(payload.get("seed_runs"), list) else []
    pointer_losses: list[float] = []
    pointer_gaps: list[float] = []
    pointer_steps: list[float] = []
    for seed_row in seed_runs:
        if not isinstance(seed_row, dict):
            continue
        train_path = seed_row.get("train_report") or seed_row.get("train_report_path")
        if not train_path:
            continue
        train_payload = _read_json(Path(str(train_path)))
        final_metrics = train_payload.get("final_metrics", {}) if isinstance(train_payload.get("final_metrics"), dict) else {}
        _append_float(pointer_losses, final_metrics.get("mean_pointer_necessity_loss"))
        _append_float(pointer_gaps, final_metrics.get("mean_pointer_necessity_gap"))
        _append_float(pointer_steps, final_metrics.get("pointer_necessity_active_steps"))
    return {
        "combo_slug": combo_dir.name,
        "replication_report_exists": report_path.exists(),
        "progress_report_exists": progress_path.exists(),
        "report_path": str(report_path).replace("\\", "/"),
        "progress_report_path": str(progress_path).replace("\\", "/") if progress_path.exists() else None,
        "mean_accuracy": metrics.get("mean_accuracy"),
        "std_accuracy": metrics.get("std_accuracy"),
        "stable_seed_rate": metrics.get("stable_seed_rate"),
        "mean_avg_tokens": metrics.get("mean_avg_tokens"),
        "mean_avg_runway_tokens": metrics.get("mean_avg_runway_tokens"),
        "seed_29_accuracy": _seed_accuracy(seed_runs, 29),
        "pointer_necessity_weight": _config_value(payload, "pointer_necessity_weight"),
        "mean_pointer_necessity_loss": _mean(pointer_losses),
        "mean_pointer_necessity_gap": _mean(pointer_gaps),
        "mean_pointer_necessity_active_steps": _mean(pointer_steps),
    }


def _find_progress_report(combo_dir: Path) -> Path:
    direct = combo_dir / "m19_replication_progress.json"
    if direct.exists():
        return direct
    nested = sorted(combo_dir.glob("*/m19_replication_progress.json"))
    return nested[0] if nested else direct


def _merge_final_rows(partial_rows: list[dict[str, Any]], final_rows: list[Any]) -> list[dict[str, Any]]:
    rows_by_slug = {str(row.get("combo_slug")): dict(row) for row in partial_rows}
    for final_row in final_rows:
        if not isinstance(final_row, dict):
            continue
        slug = str(final_row.get("combo_slug"))
        merged = rows_by_slug.get(slug, {"combo_slug": slug})
        merged.update(final_row)
        merged["replication_report_exists"] = True
        rows_by_slug[slug] = merged
    return [rows_by_slug[key] for key in sorted(rows_by_slug)]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _config_value(payload: dict[str, Any], key: str) -> Any:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    return config.get(key)


def _seed_accuracy(seed_runs: list[Any], seed: int) -> float | None:
    for row in seed_runs:
        if isinstance(row, dict) and int(row.get("seed", -1)) == int(seed):
            value = row.get("overall_accuracy")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _append_float(values: list[float], value: Any) -> None:
    try:
        if value is not None:
            values.append(float(value))
    except (TypeError, ValueError):
        return


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value) if value is not None else ""


if __name__ == "__main__":
    main()
