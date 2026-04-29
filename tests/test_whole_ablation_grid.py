from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_whole_grid_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "control_plane" / "run_whole_ablation_grid.py"
    spec = importlib.util.spec_from_file_location("run_whole_ablation_grid", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m19_direct_unified_eval_headline_metrics_are_not_clobbered() -> None:
    whole_grid = _load_whole_grid_module()
    whole_grid.DEFAULT_M19_REPLICATION_ROOT = Path("runs") / "missing_replication_for_test"
    whole_grid.DEFAULT_M19_KILL_ROOT = Path("runs") / "missing_kill_for_test"
    payload = {
        "headline_metrics": {
            "overall_accuracy": 0.3775,
            "avg_tokens": 32.0,
            "lift_vs_en_cot": 0.375,
            "lift_vs_random": 0.3575,
            "audit_qformer_accuracy": 1.0,
            "purged_accuracy": 0.373,
            "masked_accuracy": 0.0,
            "replication_mean_accuracy": 0.2,
            "replication_std_accuracy": 0.2,
            "entity_accuracy": 0.0,
            "format_accuracy": 0.2625,
            "numeric_accuracy": 0.3625,
        },
        "metrics": {},
    }

    metrics = whole_grid._special_stage_metrics("M19", payload)

    assert metrics["mainline_overall_accuracy"] == 0.3775
    assert metrics["mainline_avg_tokens"] == 32.0
    assert metrics["mainline_lift_vs_en_cot"] == 0.375
    assert metrics["mainline_lift_vs_random"] == 0.3575
    assert metrics["mainline_audit_qformer_accuracy"] == 1.0
    assert metrics["purged_accuracy"] == 0.373
    assert metrics["replication_mean_accuracy"] == 0.2
    assert metrics["replication_std_accuracy"] == 0.2
    assert metrics["kill_entity_accuracy"] == 0.0
    assert metrics["kill_format_accuracy"] == 0.2625
    assert metrics["kill_numeric_accuracy"] == 0.3625


def test_m19_legacy_metrics_can_supplement_headline_metrics() -> None:
    whole_grid = _load_whole_grid_module()
    whole_grid.DEFAULT_M19_REPLICATION_ROOT = Path("runs") / "missing_replication_for_test"
    whole_grid.DEFAULT_M19_KILL_ROOT = Path("runs") / "missing_kill_for_test"
    payload = {
        "headline_metrics": {
            "overall_accuracy": 0.31,
            "avg_tokens": 17.0,
        },
        "metrics": {
            "premature_stop_rate": 0.02,
            "max_cap_hit_rate": 0.01,
            "caa_manifold_entanglement_score": 0.13,
        },
    }

    metrics = whole_grid._special_stage_metrics("M19", payload)

    assert metrics["mainline_overall_accuracy"] == 0.31
    assert metrics["mainline_avg_tokens"] == 17.0
    assert metrics["mainline_premature_stop_rate"] == 0.02
    assert metrics["mainline_max_cap_hit_rate"] == 0.01
    assert metrics["mainline_caa_entanglement"] == 0.13


def test_m19_direct_contract_updates_stale_spine_policy() -> None:
    whole_grid = _load_whole_grid_module()
    stage = {
        "stage_key": "M19",
        "required_test_contracts": ["m19.runway_efficiency"],
        "comparison_targets": [{"target": "M18"}],
        "historical_comparison_families": ["J"],
    }
    payload = {
        "comparison_contract": {
            "required_test_contract_ids": [
                "m19.runway_efficiency",
                "m19.replication_stability",
                "m19.kill_test_suite",
            ],
            "comparison_targets": [{"target": "M19"}, {"target": "M18"}],
            "historical_comparison_families": ["J", "L"],
        }
    }

    updated = whole_grid._stage_with_direct_contract(stage, payload)

    assert updated["required_test_contracts"] == [
        "m19.runway_efficiency",
        "m19.replication_stability",
        "m19.kill_test_suite",
    ]
    assert [target["target"] for target in updated["comparison_targets"]] == ["M19", "M18"]
    assert updated["historical_comparison_families"] == ["J", "L"]
