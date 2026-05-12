from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_summary_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "summarize_m19_microgrid.py"
    spec = importlib.util.spec_from_file_location("summarize_m19_microgrid", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_summary = _load_summary_module()


def test_microgrid_summary_reads_partial_pointer_metrics(tmp_path: Path) -> None:
    combo = tmp_path / "lr_5em05_aug_0p5_fmt_0p25_surf_0p0_ptr_0p05"
    seed_dir = combo / "seed_29"
    seed_dir.mkdir(parents=True)
    train_report = seed_dir / "train_report.json"
    train_report.write_text(
        json.dumps(
            {
                "final_metrics": {
                    "mean_pointer_necessity_loss": 0.02,
                    "mean_pointer_necessity_gap": 0.17,
                    "pointer_necessity_active_steps": 6036,
                }
            }
        ),
        encoding="utf-8",
    )
    nested_run = combo / "nested_replication_run"
    nested_run.mkdir()
    progress_report = nested_run / "m19_replication_progress.json"
    progress_report.write_text(
        json.dumps(
            {
                "config": {"pointer_necessity_weight": 0.05},
                "metrics": {"mean_accuracy": 0.4, "mean_avg_tokens": 32.0},
                "seed_runs": [
                    {
                        "seed": 29,
                        "overall_accuracy": 0.41,
                        "train_report": str(train_report),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = _summary.summarize_microgrid_run(tmp_path)

    assert summary["final_report_exists"] is False
    assert summary["completed_combo_count"] == 0
    assert summary["incomplete_combo_count"] == 1
    row = summary["combo_rows"][0]
    assert row["pointer_necessity_weight"] == 0.05
    assert row["progress_report_path"].endswith("nested_replication_run/m19_replication_progress.json")
    assert row["seed_29_accuracy"] == 0.41
    assert row["mean_pointer_necessity_loss"] == 0.02
    assert row["mean_pointer_necessity_gap"] == 0.17
    assert "pointer_loss" in _summary.render_summary_text(summary)


def test_microgrid_summary_merges_final_grid_rows(tmp_path: Path) -> None:
    combo = tmp_path / "combo_a"
    combo.mkdir()
    (combo / "m19_replication_report.json").write_text(
        json.dumps({"metrics": {"mean_accuracy": 0.1}, "seed_runs": []}),
        encoding="utf-8",
    )
    (tmp_path / "m19_stability_microgrid_report.json").write_text(
        json.dumps(
            {
                "headline": {"best_mean_accuracy": 0.6},
                "grid_rows": [
                    {
                        "combo_slug": "combo_a",
                        "pointer_necessity_weight": 0.1,
                        "mean_accuracy": 0.6,
                        "seed_29_accuracy": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = _summary.summarize_microgrid_run(tmp_path)

    assert summary["final_report_exists"] is True
    assert summary["headline"]["best_mean_accuracy"] == 0.6
    assert summary["combo_rows"][0]["mean_accuracy"] == 0.6
    assert summary["combo_rows"][0]["replication_report_exists"] is True
