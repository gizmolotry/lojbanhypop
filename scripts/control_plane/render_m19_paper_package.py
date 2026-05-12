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
    parser.add_argument("--dictionary-audit-report", type=Path, default=None)
    parser.add_argument("--surface-contract-report", type=Path, default=None)
    parser.add_argument("--bridge-channel-report", type=Path, default=None)
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
    dictionary = _read_json(args.dictionary_audit_report) if args.dictionary_audit_report else {}
    surface_contract = _read_json(args.surface_contract_report) if args.surface_contract_report else {}
    bridge_channel = _read_json(args.bridge_channel_report) if args.bridge_channel_report else {}
    whole = _read_json(args.whole_grid_manifest) if args.whole_grid_manifest else {}

    figure_rows = _build_figure_rows(direct, replication, kill, surface_contract, bridge_channel)
    table_md = _render_ablation_table(figure_rows)
    verdict = _narrative_verdict(direct, replication, kill, dictionary)
    paper_md = _render_paper_package(direct, replication, kill, dictionary, bridge_channel, whole, table_md, verdict)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_paths": {
            "direct_unified_eval": _repo_string(args.direct_unified_eval),
            "replication_report": _repo_string(args.replication_report) if args.replication_report else None,
            "kill_test_report": _repo_string(args.kill_test_report) if args.kill_test_report else None,
            "dictionary_audit_report": _repo_string(args.dictionary_audit_report) if args.dictionary_audit_report else None,
            "surface_contract_report": _repo_string(args.surface_contract_report) if args.surface_contract_report else None,
            "bridge_channel_report": _repo_string(args.bridge_channel_report) if args.bridge_channel_report else None,
            "whole_grid_manifest": _repo_string(args.whole_grid_manifest) if args.whole_grid_manifest else None,
        },
        "figure_rows": figure_rows,
        "narrative_verdict": verdict,
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
    (run_dir / "m19_appendix.md").write_text(_render_appendix(direct, replication, kill, dictionary), encoding="utf-8")
    (run_dir / "m19_benchmark_protocol.md").write_text(_render_protocol(direct), encoding="utf-8")
    (run_dir / "m19_paper_package.md").write_text(paper_md, encoding="utf-8")
    (run_dir / M19_REGISTRY["M19"]["report_names"]["paper_package"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)
    args.doc_output.write_text(paper_md, encoding="utf-8")
    print(f"Wrote paper package: {run_dir}")


def _build_figure_rows(
    direct: dict[str, Any],
    replication: dict[str, Any],
    kill: dict[str, Any],
    surface_contract: dict[str, Any],
    bridge_channel: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    headline = direct.get("headline_metrics", {}) if isinstance(direct.get("headline_metrics"), dict) else {}
    rows = [
        {
            "regime": "M19.3 mainline",
            "accuracy": headline.get("overall_accuracy"),
            "avg_tokens": headline.get("avg_tokens"),
            "accuracy_per_token": headline.get("accuracy_per_token"),
            "avg_runway_tokens": headline.get("avg_runway_tokens"),
            "accuracy_per_runway_token": headline.get("accuracy_per_runway_token"),
            "token_ratio_vs_en_cot": headline.get("token_ratio_vs_en_cot"),
            "retained_cot_accuracy_per_token": headline.get("compression_adjusted_retention"),
            "notes": "current direct unified eval headline",
        },
        {
            "regime": "Purged",
            "accuracy": headline.get("purged_accuracy"),
            "avg_tokens": headline.get("purged_avg_tokens"),
            "accuracy_per_token": _safe_div(headline.get("purged_accuracy"), headline.get("purged_avg_tokens")),
            "avg_runway_tokens": None,
            "accuracy_per_runway_token": None,
            "token_ratio_vs_en_cot": None,
            "retained_cot_accuracy_per_token": None,
            "notes": "overlap-purged benchmark slice",
        },
        {
            "regime": "Replication mean",
            "accuracy": _nested(replication, "metrics", "mean_accuracy"),
            "avg_tokens": _nested(replication, "metrics", "mean_avg_tokens"),
            "accuracy_per_token": _safe_div(
                _nested(replication, "metrics", "mean_accuracy"),
                _nested(replication, "metrics", "mean_avg_tokens"),
            ),
            "avg_runway_tokens": _nested(replication, "metrics", "mean_avg_runway_tokens"),
            "accuracy_per_runway_token": _safe_div(
                _nested(replication, "metrics", "mean_accuracy"),
                _nested(replication, "metrics", "mean_avg_runway_tokens"),
            ),
            "token_ratio_vs_en_cot": None,
            "retained_cot_accuracy_per_token": None,
            "notes": "multi-seed mean",
        },
        {
            "regime": "Entity kill",
            "accuracy": _nested(kill, "metrics", "entity_accuracy"),
            "avg_tokens": None,
            "accuracy_per_token": None,
            "avg_runway_tokens": None,
            "accuracy_per_runway_token": None,
            "token_ratio_vs_en_cot": None,
            "retained_cot_accuracy_per_token": None,
            "notes": "entity anonymization on purged slice",
        },
        {
            "regime": "Format kill",
            "accuracy": _nested(kill, "metrics", "format_accuracy"),
            "avg_tokens": None,
            "accuracy_per_token": None,
            "avg_runway_tokens": None,
            "accuracy_per_runway_token": None,
            "token_ratio_vs_en_cot": None,
            "retained_cot_accuracy_per_token": None,
            "notes": "format flattening on purged slice",
        },
        {
            "regime": "Masked blindfold",
            "accuracy": headline.get("masked_accuracy"),
            "avg_tokens": None,
            "accuracy_per_token": None,
            "avg_runway_tokens": None,
            "accuracy_per_runway_token": None,
            "token_ratio_vs_en_cot": None,
            "retained_cot_accuracy_per_token": None,
            "notes": "lexical blindfold carryover from integrity suite",
        },
    ]
    if surface_contract:
        rows.append(
            {
                "regime": "May 8 surface contract",
                "accuracy": _nested(surface_contract, "metrics", "mean_accuracy"),
                "avg_tokens": _nested(surface_contract, "metrics", "mean_avg_tokens"),
                "accuracy_per_token": _safe_div(
                    _nested(surface_contract, "metrics", "mean_accuracy"),
                    _nested(surface_contract, "metrics", "mean_avg_tokens"),
                ),
                "avg_runway_tokens": _nested(surface_contract, "metrics", "mean_avg_runway_tokens"),
                "accuracy_per_runway_token": _safe_div(
                    _nested(surface_contract, "metrics", "mean_accuracy"),
                    _nested(surface_contract, "metrics", "mean_avg_runway_tokens"),
                ),
                "token_ratio_vs_en_cot": None,
                "retained_cot_accuracy_per_token": None,
                "notes": "negative robustness result; not promoted",
            }
        )
    if bridge_channel:
        full_row = _channel_mode_row(bridge_channel, "full")
        no_judri = _nested(bridge_channel, "headline", "no_judri_accuracy")
        gismu_only = _nested(bridge_channel, "headline", "gismu_only_accuracy")
        if full_row:
            rows.append(
                {
                    "regime": "Bridge channel full",
                    "accuracy": full_row.get("strict_accuracy"),
                    "avg_tokens": full_row.get("avg_tokens"),
                    "accuracy_per_token": full_row.get("accuracy_per_token"),
                    "avg_runway_tokens": full_row.get("avg_runway_tokens"),
                    "accuracy_per_runway_token": full_row.get("accuracy_per_runway_token"),
                    "token_ratio_vs_en_cot": None,
                    "retained_cot_accuracy_per_token": None,
                    "notes": f"eval-only channel isolation; no_judri={_fmt(no_judri)} gismu_only={_fmt(gismu_only)}",
                }
            )
    return rows


def _render_ablation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M19 Ablation Table",
        "",
        "| Regime | Accuracy | Avg Tokens | Acc/Token | Runway Tokens | Acc/Runway Token | CoT Token Ratio | Retained CoT Acc/Token | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('avg_tokens'))} | {_fmt(row.get('accuracy_per_token'))} | {_fmt(row.get('avg_runway_tokens'))} | {_fmt(row.get('accuracy_per_runway_token'))} | {_fmt(row.get('token_ratio_vs_en_cot'))} | {_fmt(row.get('retained_cot_accuracy_per_token'))} | {row.get('notes', '')} |"
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


def _render_appendix(
    direct: dict[str, Any],
    replication: dict[str, Any],
    kill: dict[str, Any],
    dictionary: dict[str, Any],
    bridge_channel: dict[str, Any] | None = None,
) -> str:
    lines = [
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
            "## Typed Faithfulness Headlines",
            "",
            f"- typed family accuracy: `{_fmt(_dictionary_metric(dictionary, 'typed_family_accuracy'))}`",
            f"- arity violation rate: `{_fmt(_dictionary_metric(dictionary, 'arity_violation_rate'))}`",
            f"- masked pointer zero rate: `{_fmt(_dictionary_metric(dictionary, 'masked_pointer_zero_rate'))}`",
            f"- symbolic trace alignment: `{_fmt(_dictionary_metric(dictionary, 'symbolic_trace_alignment'))}`",
            "",
    ]
    if bridge_channel:
        interp = bridge_channel.get("interpretation", {}) if isinstance(bridge_channel.get("interpretation"), dict) else {}
        lines.extend(
            [
                "## Bridge Channel Isolation",
                "",
                f"- full accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'full_accuracy'))}`",
                f"- scratchpad-only accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'scratchpad_only_accuracy'))}`",
                f"- random-shape accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'random_shape_accuracy'))}`",
                f"- no-gismu accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'no_gismu_accuracy'))}`",
                f"- gismu-only accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'gismu_only_accuracy'))}`",
                f"- no-judri accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'no_judri_accuracy'))}`",
                f"- judri-only accuracy: `{_fmt(_nested(bridge_channel, 'headline', 'judri_only_accuracy'))}`",
                f"- bridge carries answer signal: `{interp.get('bridge_carries_answer_signal')}`",
                f"- predicate-only shortcut warning: `{interp.get('predicate_only_shortcut_warning')}`",
                f"- pointer-only shortcut warning: `{interp.get('pointer_only_shortcut_warning')}`",
                "",
            ]
        )
    return "\n".join(lines)


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
    dictionary: dict[str, Any],
    bridge_channel: dict[str, Any],
    whole: dict[str, Any],
    table_md: str,
    verdict: str,
) -> str:
    stage_count = whole.get("coverage_summary", {}).get("stage_count") if isinstance(whole.get("coverage_summary"), dict) else None
    return "\n".join(
        [
            "# M19 Paper Package",
            "",
            f"- narrative verdict: `{verdict}`",
            "",
            "## Narrative",
            "",
            "M19 currently supports a narrower and more defensible claim than earlier optimistic summaries: a compact symbiote runway can retain substantial lift with strong token efficiency and survive initial anti-shortcut checks.",
            "The current evidence does not justify a claim of general reasoning closure, nor does it erase the negative lessons from earlier bridge families.",
            "The May 8 weak-seed surface-contract run is treated as a negative robustness result: stronger surface consistency did not recover seed stability and should not be promoted.",
            "The bridge-channel isolation suite is now the active diagnostic path: it tests whether answer lift survives removing typed predicate or pointer channels.",
            "The training response is an opt-in pointer-necessity contrast loss: full bridge answer loss must beat a no-judri ablated bridge by a margin, forcing answer lift to depend on judri pointer channels instead of predicate-only shortcuts.",
            "Strict accuracy remains canonical; phrase accuracy and token efficiency are diagnostic context only.",
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
            _render_appendix(direct, replication, kill, dictionary, bridge_channel),
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


def _channel_mode_row(bridge_channel: dict[str, Any], mode: str) -> dict[str, Any] | None:
    rows = bridge_channel.get("mode_rows", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("mode")) == str(mode):
            return row
    return None


def _dictionary_metric(dictionary: dict[str, Any], metric: str) -> Any:
    checkpoints = dictionary.get("checkpoints", [])
    if isinstance(checkpoints, list) and checkpoints:
        first = checkpoints[0]
        if isinstance(first, dict):
            typed = first.get("typed_faithfulness", {})
            if isinstance(typed, dict):
                return typed.get(metric)
    return None


def _narrative_verdict(
    direct: dict[str, Any],
    replication: dict[str, Any],
    kill: dict[str, Any],
    dictionary: dict[str, Any],
) -> str:
    purged = _nested(direct, "headline_metrics", "purged_accuracy")
    mean_acc = _nested(replication, "metrics", "mean_accuracy")
    stable = _nested(replication, "metrics", "stable_seed_rate")
    format_acc = _nested(kill, "metrics", "format_accuracy")
    typed_acc = _dictionary_metric(dictionary, "typed_family_accuracy")
    arity_violation = _dictionary_metric(dictionary, "arity_violation_rate")
    audit = _nested(direct, "headline_metrics", "audit_qformer_accuracy")

    try:
        purged_f = float(purged)
        mean_f = float(mean_acc)
        stable_f = float(stable)
        format_f = float(format_acc)
        typed_f = float(typed_acc)
        arity_f = float(arity_violation)
        audit_f = float(audit)
    except (TypeError, ValueError):
        return "not_promotable"

    if mean_f < 0.05 or audit_f <= 0.0:
        return "not_promotable"
    if typed_f >= 0.5 and purged_f < 0.1:
        return "typed_faithful_but_behaviorally_weak"
    if stable_f >= 0.6 and purged_f >= 0.4 and format_f >= 0.3 and arity_f <= 0.1:
        return "stable_and_competitive"
    if format_f < 0.25 or arity_f > 0.5:
        return "shortcut_risk_unresolved"
    return "promising_but_unstable"


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


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
