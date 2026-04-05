from __future__ import annotations

import pytest

from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


def test_series_metadata_shape() -> None:
    meta = series_metadata("J", "invariance_data", "scripts/eval_j_5.py")
    assert meta["series_id"] == "J"
    assert meta["track"] == "invariance_data"
    assert meta["script"].endswith("eval_j_5.py")


def test_allows_expected_roots() -> None:
    assert_output_path_allowed("J", "runs/j_series/20260304_000000/j-5.json")
    assert_output_path_allowed("L", "runs/l_series/20260304_000000/l_series_summary.json")
    assert_output_path_allowed("A-G", "runs/ablation/a_to_g/20260304_000000/ablation_matrix.json")
    assert_output_path_allowed(
        "M",
        "artifacts/runs/telemetry/raw/ablation/hypercube/20260304_000000/ablation_hypercube_report.json",
    )


def test_rejects_cross_series_root() -> None:
    with pytest.raises(ValueError):
        assert_output_path_allowed("J", "runs/l_series/20260304_000000/j-5.json")

    with pytest.raises(ValueError):
        assert_output_path_allowed("L", "runs/j_series/20260304_000000/l_series_summary.json")

    with pytest.raises(ValueError):
        assert_output_path_allowed("M", "runs/ablation/a_to_g/20260304_000000/ablation_matrix.json")

    with pytest.raises(ValueError):
        assert_output_path_allowed("M", "runs/h_series/20260304_000000/run_h_series.json")


def test_validate_series_outputs_rejects_undeclared_root() -> None:
    with pytest.raises(ValueError):
        validate_series_outputs(
            "M",
            ["runs/j_series/test_root"],
            ["runs/j_series/other_root/20260304_000000/run_h_series.json"],
        )


def test_lineage_metadata_shape() -> None:
    lineage = lineage_metadata(
        "train",
        checkpoint_in="runs/l_series/base/checkpoint.pt",
        checkpoint_out="runs/l_series/next/checkpoint.pt",
        dataset_profile="legacy",
        difficulty_tier="all",
    )
    assert lineage["mode"] == "train"
    assert lineage["checkpoint_in"].endswith("checkpoint.pt")
    assert lineage["dataset_profile"] == "legacy"
