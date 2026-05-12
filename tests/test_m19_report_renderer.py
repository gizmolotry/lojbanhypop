from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_renderer_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "control_plane" / "render_m19_paper_package.py"
    spec = importlib.util.spec_from_file_location("render_m19_paper_package", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_renderer = _load_renderer_module()


def test_token_efficiency_fields_render_when_cot_baselines_are_missing() -> None:
    rows = _renderer._build_figure_rows(
        direct={
            "headline_metrics": {
                "overall_accuracy": 0.6,
                "avg_tokens": 30.0,
                "accuracy_per_token": 0.02,
                "avg_runway_tokens": 38.0,
                "accuracy_per_runway_token": 0.015789473684210527,
                "purged_accuracy": 0.5,
                "purged_avg_tokens": 25.0,
                "masked_accuracy": 0.0,
            }
        },
        replication={"metrics": {"mean_accuracy": 0.4, "mean_avg_tokens": 20.0}},
        kill={"metrics": {"entity_accuracy": 0.3, "format_accuracy": 0.2}},
        surface_contract={"metrics": {"mean_accuracy": 0.435, "mean_avg_tokens": 31.32}},
        bridge_channel={
            "headline": {"no_judri_accuracy": 0.70, "gismu_only_accuracy": 0.30},
            "mode_rows": [
                {
                    "mode": "full",
                    "strict_accuracy": 0.75,
                    "avg_tokens": 30.5,
                    "accuracy_per_token": 0.0246,
                    "avg_runway_tokens": 38.5,
                    "accuracy_per_runway_token": 0.0195,
                }
            ],
        },
    )

    table = _renderer._render_ablation_table(rows)

    assert "CoT Token Ratio" in table
    assert "Retained CoT Acc/Token" in table
    assert "Runway Tokens" in table
    assert "| M19.3 mainline | 0.6000 | 30.0000 | 0.0200 | 38.0000 | 0.0158 |  |  |" in table
    assert "| May 8 surface contract | 0.4350 | 31.3200 | 0.0139 |  |  |  |  |" in table
    assert "| Bridge channel full | 0.7500 | 30.5000 | 0.0246 | 38.5000 | 0.0195 |  |  |" in table
