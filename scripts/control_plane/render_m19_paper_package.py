from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_OUTPUT = REPO_ROOT / "docs" / "history" / "reports" / "M19_PAPER_PACKAGE_LATEST.md"


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Render a paper-style package for the active M19 evidence stack.")
    parser.add_argument("--direct-unified-eval", type=Path, required=True)
    parser.add_argument("--replication-report", type=Path, default=None)
    parser.add_argument("--kill-test-report", type=Path, default=None)
    parser.add_argument("--whole-grid-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["paper_package"]))
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    direct = _read_json(args.direct_unified_eval)
    replication = _read_json(args.replication_report) if args.replication_report else {}
    kill = _read_json(args.kill_test_report) if args.kill_test_report else {}
    whole = _read_json(args.whole_grid_manifest) if args.whole_grid_manifest else {}

    figure_rows = _build_figure_rows(direct, replication, kill)
    table_md = _render_ablation_table(figure_rows)
    paper_md = _render_paper_package(direct, replication, kill, whole, table_md)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_paths": {
            "direct_unified_eval": _repo_string(args.direct_unified_eval),
            "replication_report": _repo_string(args.replication_report) if args.replication_report else None,
            "kill_test_report": _repo_string(args.kill_test_report) if args.kill_test_report else None,
            "whole_grid_manifest": _repo_string(args.whole_grid_manifest) if args.whole_grid_manifest else None,
        },
        "figure_rows": figure_rows,
        "sections": {
            "ablation_table": "m19_ablation_table.md",
            "methodology": "m19_methodology.md",
            "limitations": "m19_limitations.md",
            "appendix": "m19_appendix.md",
            "benchmark_protocol": "m19_benchmark_protocol.md",
            "paper_package": "m19_paper_package.md",
        },
    }

    (run_dir / "m19_ablation_table.md").write_text(table_md, encoding="utf-8")
    (run_dir / "m19_methodology.md").write_text(_render_methodology(direct), encoding="utf-8")
    (run_dir / "m19_limitations.md").write_text(_render_limitations(direct, replication, kill), encoding="utf-8")
    (run_dir / "m19_appendix.md").write_text(_render_appendix(direct, replication, kill), encoding="utf-8")
    (run_dir / "m19_benchmark_protocol.md").write_text(_render_protocol(direct), encoding="utf-8")
    (run_dir / "m19_paper_package.md").write_text(paper_md, encoding="utf-8")
    (run_dir / M19_REGISTRY["M19"]["report_names"]["paper_package"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)
    args.doc_output.write_text(paper_md, encoding="utf-8")
    print(f"Wrote paper package: {run_dir}")


def _build_figure_rows(direct: dict[str, Any], replication: dict[str, Any], kill: dict[str, Any]) -> list[dict[str, Any]]:
    headline = direct.get("headline_metrics", {}) if isinstance(direct.get("headline_metrics"), dict) else {}
    return [
        {
            "regime": "M19.3 mainline",
            "accuracy": headline.get("overall_accuracy"),
            "avg_tokens": headline.get("avg_tokens"),
            "notes": "current direct unified eval headline",
        },
        {
            "regime": "Purged",
            "accuracy": headline.get("purged_accuracy"),
            "avg_tokens": headline.get("purged_avg_tokens"),
            "notes": "overlap-purged benchmark slice",
        },
        {
            "regime": "Replication mean",
            "accuracy": _nested(replication, "metrics", "mean_accuracy"),
            "avg_tokens": _nested(replication, "metrics", "mean_avg_tokens"),
            "notes": "multi-seed mean",
        },
        {
            "regime": "Entity kill",
            "accuracy": _nested(kill, "metrics", "entity_accuracy"),
            "avg_tokens": None,
            "notes": "entity anonymization on purged slice",
        },
        {
            "regime": "Format kill",
            "accuracy": _nested(kill, "metrics", "format_accuracy"),
            "avg_tokens": None,
            "notes": "format flattening on purged slice",
        },
        {
            "regime": "Masked blindfold",
            "accuracy": headline.get("masked_accuracy"),
            "avg_tokens": None,
            "notes": "lexical blindfold carryover from integrity suite",
        },
    ]


def _render_ablation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M19 Ablation Table",
        "",
        "| Regime | Accuracy | Avg Tokens | Notes |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('avg_tokens'))} | {row.get('notes', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_methodology(direct: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Methodology",
            "",
            "M19 is evaluated as a bounded continuous scratchpad runway rather than a claim of general reasoning closure.",
            "The mainline hypothesis is that compact symbiote runway states can retain useful reasoning lift at far lower token cost than explicit natural-language CoT.",
            "All active claims are expected to route through the ledger-backed surfaces: benchmark, audit, integrity, replication, and kill tests.",
            "",
            f"Active track: `{direct.get('track', '')}`",
            "",
        ]
    )


def _render_limitations(direct: dict[str, Any], replication: dict[str, Any], kill: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Limitations",
            "",
            "- M19 does not prove general reasoning solved; it shows a compact runway can retain lift and survive initial anti-shortcut checks.",
            "- Legacy H/J/L families are largely comparison anchors and obligations, not fully rerunnable peers on the exact same benchmark surface.",
            f"- Replication count is `{_nested(replication, 'metrics', 'replication_count')}`; more seeds would still strengthen stability claims.",
            f"- Entity/format kill tests are stronger than lexical masking alone, but they are still perturbation checks rather than theorem-proving guarantees.",
            f"- Current masked blindfold accuracy is `{_fmt(_nested(kill, 'metrics', 'masked_accuracy') or _nested(direct, 'headline_metrics', 'masked_accuracy'))}`.",
            "",
        ]
    )


def _render_appendix(direct: dict[str, Any], replication: dict[str, Any], kill: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Appendix",
            "",
            "## Direct Unified Eval Contracts",
            "",
            *[
                f"- `{row.get('test_id')}`: {row.get('status')} ({row.get('provenance')})"
                for row in direct.get("contract_results", [])
                if isinstance(row, dict)
            ],
            "",
            "## Replication Headlines",
            "",
            f"- mean accuracy: `{_fmt(_nested(replication, 'metrics', 'mean_accuracy'))}`",
            f"- std accuracy: `{_fmt(_nested(replication, 'metrics', 'std_accuracy'))}`",
            "",
            "## Kill-Test Headlines",
            "",
            f"- entity accuracy: `{_fmt(_nested(kill, 'metrics', 'entity_accuracy'))}`",
            f"- format accuracy: `{_fmt(_nested(kill, 'metrics', 'format_accuracy'))}`",
            f"- numeric accuracy: `{_fmt(_nested(kill, 'metrics', 'numeric_accuracy'))}`",
            "",
        ]
    )


def _render_protocol(direct: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable Benchmark Protocol",
            "",
            "1. Train the active M19 mainline cell under the current contract.",
            "2. Benchmark on the shared M14.5 unified surface with BASE, EN-COT, ZH-COT, RANDOM-SHAPE, and SCRATCHPAD-ONLY controls.",
            "3. Run the sanity audit with BASE-NO-SCRATCHPAD, SCRATCHPAD-ONLY, RANDOM, and Q-FORMER.",
            "4. Run integrity on full, purged, overlap, and masked slices.",
            "5. Run multi-seed replications and broader kill tests before promoting a claim.",
            "",
            f"Current direct family: `{direct.get('family_key', '')}` / `{direct.get('track', '')}`",
            "",
        ]
    )


def _render_paper_package(
    direct: dict[str, Any],
    replication: dict[str, Any],
    kill: dict[str, Any],
    whole: dict[str, Any],
    table_md: str,
) -> str:
    stage_count = whole.get("coverage_summary", {}).get("stage_count") if isinstance(whole.get("coverage_summary"), dict) else None
    return "\n".join(
        [
            "# M19 Paper Package",
            "",
            "## Narrative",
            "",
            "M19 currently supports a narrower and more defensible claim than earlier optimistic summaries: a compact symbiote runway can retain substantial lift with strong token efficiency and survive initial anti-shortcut checks.",
            "The current evidence does not justify a claim of general reasoning closure, nor does it erase the negative lessons from earlier bridge families.",
            "",
            "## Ablation Table",
            "",
            table_md,
            "## Methodology",
            "",
            _render_methodology(direct),
            "## Limitations",
            "",
            _render_limitations(direct, replication, kill),
            "## Appendix",
            "",
            _render_appendix(direct, replication, kill),
            "## Benchmark Protocol",
            "",
            _render_protocol(direct),
            "## Whole-Program Context",
            "",
            f"- whole-grid stage count: `{stage_count}`",
            f"- direct unified eval track: `{direct.get('track', '')}`",
            "",
        ]
    )


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _repo_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
