from __future__ import annotations

import json
from pathlib import Path
import uuid

from lojban_evolution.direct_unified_eval import (
    build_direct_unified_eval_manifest,
    render_direct_unified_eval_markdown,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scratch_dir() -> Path:
    root = Path("runs") / "test_direct_unified_eval" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_build_direct_unified_eval_manifest_static_m19() -> None:
    tmp_path = _scratch_dir()
    benchmark_path = _write_json(
        tmp_path / "benchmark.json",
        {
            "config": {
                "dynamic_pacing": False,
                "cell_id": "M19.3_8Q_128D_8S",
            },
            "metrics": {
                "strict_accuracy": 0.19,
                "overall_accuracy": 0.19,
                "avg_tokens": 30.32,
                "accuracy_per_token": 0.00626,
                "token_ratio_vs_en_cot": 0.24,
                "lift_vs_zh_cot": 0.19,
                "zh_cot_accuracy": 0.0,
                "zh_cot_avg_tokens": 117.9,
            },
            "results": {
                "M19.3_8Q_128D_8S": {"accuracy": 0.19},
                "EN-COT": {"accuracy": 0.0},
                "ZH-COT": {"accuracy": 0.0},
            },
            "headline": {"overall_accuracy": 0.19},
        },
    )
    audit_path = _write_json(
        tmp_path / "audit.json",
        {
            "headline": {
                "qformer_accuracy": 0.1,
                "random_accuracy": 0.0,
                "lift_vs_base": 0.1,
                "lift_vs_random": 0.1,
            }
        },
    )
    integrity_path = _write_json(
        tmp_path / "integrity.json",
        {
            "metrics": {
                "purged_accuracy": 0.18,
                "overlap_gap": 0.02,
                "masked_accuracy": 0.03,
                "audit_qformer_accuracy": 1.0,
                "audit_lift_vs_random": 1.0,
            },
            "headline": {
                "integrity_status": "pass",
            },
        },
    )
    replication_path = _write_json(
        tmp_path / "replication.json",
        {
            "metrics": {
                "replication_count": 3,
                "mean_accuracy": 0.36,
                "std_accuracy": 0.01,
                "mean_avg_tokens": 31.9,
                "mean_audit_qformer_accuracy": 0.95,
            }
        },
    )
    stability_path = _write_json(
        tmp_path / "stability.json",
        {
            "headline": {
                "configs_tested": 4,
                "best_mean_accuracy": 0.28,
                "best_stable_seed_rate": 0.5,
                "recovered_seed_count": 1,
            },
            "best_configs": {
                "best_balanced": {
                    "combo_slug": "lr_5em05_aug_0p0",
                    "mean_accuracy": 0.28,
                    "stable_seed_rate": 0.5,
                    "mean_audit_qformer_accuracy": 0.95,
                }
            },
        },
    )
    kill_path = _write_json(
        tmp_path / "kill.json",
        {
            "metrics": {
                "purged_accuracy": 0.18,
                "entity_accuracy": 0.17,
                "format_accuracy": 0.18,
                "numeric_accuracy": 0.16,
                "masked_accuracy": 0.03,
            }
        },
    )
    j_anchor = _write_json(tmp_path / "j-5.json", {"metrics": {"accepted_foil_pair_accuracy": 0.77}})
    l_anchor = _write_json(tmp_path / "l_series_summary.json", {"metrics": {"constraint_scope": 0.92}})
    history_path = _write_json(
        tmp_path / "history.json",
        {
            "entries": [
                {
                    "canonical_id": "legacy.j",
                    "normalized_canonical_id": "J",
                    "aliases": ["J"],
                    "lookup_aliases": ["J"],
                    "artifact_roots": [str(j_anchor)],
                },
                {
                    "canonical_id": "legacy.l",
                    "normalized_canonical_id": "L",
                    "aliases": ["L"],
                    "lookup_aliases": ["L"],
                    "artifact_roots": [str(l_anchor)],
                },
            ]
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M19",
        track="M19",
        benchmark_report_path=benchmark_path,
        audit_report_path=audit_path,
        integrity_report_path=integrity_path,
        replication_report_path=replication_path,
        stability_report_path=stability_path,
        kill_test_report_path=kill_path,
        history_manifest_path=history_path,
    )

    assert manifest["family_key"] == "M19"
    assert manifest["track"] == "M19"
    assert len(manifest["contract_results"]) >= 3

    runway = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.runway_efficiency")
    assert runway["status"] == "available"
    assert runway["metrics"]["overall_accuracy"] == 0.19

    guardrails = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.dynamic_pacing_guardrails")
    assert guardrails["status"] == "not_applicable"
    integrity = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.integrity_controls")
    assert integrity["status"] == "available"
    assert integrity["metrics"]["purged_accuracy"] == 0.18
    replication = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.replication_stability")
    assert replication["status"] == "available"
    assert replication["metrics"]["mean_accuracy"] == 0.36
    kill = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.kill_test_suite")
    assert kill["status"] == "available"
    assert kill["metrics"]["entity_accuracy"] == 0.17

    inherited = {row["test_id"] for row in manifest["contract_results"]}
    assert "m14.scratchpad_bleed" in inherited
    assert "m11.native_discriminative_oracle" in inherited

    refs = {row["family"]: row for row in manifest["historical_family_references"]}
    assert refs["J"]["status"] == "resolved"
    assert refs["L"]["status"] == "resolved"

    assert manifest["headline_metrics"]["overall_accuracy"] == 0.19
    assert manifest["headline_metrics"]["audit_qformer_accuracy"] == 0.1
    assert manifest["headline_metrics"]["purged_accuracy"] == 0.18
    assert manifest["headline_metrics"]["mean_accuracy"] == 0.36
    assert manifest["headline_metrics"]["best_mean_accuracy"] == 0.28
    assert manifest["headline_metrics"]["stability_combo_slug"] == "lr_5em05_aug_0p0"
    assert manifest["headline_metrics"]["entity_accuracy"] == 0.17

    rendered = render_direct_unified_eval_markdown(manifest)
    assert "Direct Unified Eval: M19 (M19)" in rendered
    assert "m19.runway_efficiency" in rendered


def test_build_direct_unified_eval_manifest_dynamic_m19() -> None:
    tmp_path = _scratch_dir()
    benchmark_path = _write_json(
        tmp_path / "m19_4_benchmark_report.json",
        {
            "config": {
                "dynamic_pacing": True,
                "cell_id": "M19.4",
            },
            "metrics": {
                "overall_accuracy": 0.31,
                "avg_tokens": 17.0,
                "premature_stop_rate": 0.02,
                "max_cap_hit_rate": 0.0,
                "scratchpad_bleed_rate": 0.01,
                "caa_manifold_entanglement_score": 0.13,
            },
            "dynamic_rollup": {
                "mean_latent_steps": 8.4,
            },
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M19",
        track="M19.4",
        benchmark_report_path=benchmark_path,
        audit_report_path=None,
        integrity_report_path=None,
        history_manifest_path=None,
    )

    guardrails = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.dynamic_pacing_guardrails")
    assert guardrails["status"] == "available"
    assert guardrails["metrics"]["premature_stop_rate"] == 0.02
    assert guardrails["metrics"]["mean_latent_steps"] == 8.4
