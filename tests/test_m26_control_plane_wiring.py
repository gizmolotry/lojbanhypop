from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_program_map_classifies_m26_entries_and_layer() -> None:
    program_map = load_script_module("build_ablation_program_map_m26_test", "scripts/control_plane/build_ablation_program_map.py")

    assert program_map._family_key({"normalized_canonical_id": "M26.A"}) == "M26"
    assert program_map._program_layer("M26") == "end_to_end_lojban_symbiote"

    families: dict[str, dict[str, object]] = {}
    program_map._merge_taxonomy_families(
        families,
        {
            "M26": {
                "architectural_thesis": "end-to-end Lojban symbiote full-organism runtime verification",
            }
        },
    )
    assert families["M26"]["program_layer"] == "end_to_end_lojban_symbiote"
    assert "airflow/dags/m26/lojban_m26_end_to_end_loafman_dag.py" in families["M26"]["dag_paths"]


def test_program_spine_gives_m26_dedicated_layer() -> None:
    program_spine = load_script_module("build_ablation_program_spine_m26_test", "scripts/control_plane/build_ablation_program_spine.py")

    assert program_spine._major_program_layer(25) == "emergent_bridi_grammar"
    assert program_spine._major_program_layer(26) == "end_to_end_lojban_symbiote"
    assert "M26" in program_spine.M_STAGE_ORDER
    assert program_spine.EXTRA_STAGE_DAGS["M26"] == ["airflow/dags/m26/lojban_m26_end_to_end_loafman_dag.py"]


def test_program_spine_uses_taxonomy_contract_shape_for_m26() -> None:
    program_spine = load_script_module("build_ablation_program_spine_m26_contract_test", "scripts/control_plane/build_ablation_program_spine.py")

    stage = program_spine._build_major_stage(
        "M26",
        {"architectural_thesis": "end-to-end full-organism symbiote"},
        [],
        [
            {
                "transition_id": "M25_to_M26",
                "to_major": "M26",
                "selected_upstream": "M25.A",
                "inherits_components": ["loose bridi stream"],
            }
        ],
        {
            "historical_comparison_families": ["M25"],
            "required_test_contracts": ["m26.full_hidden_state_bridge_organism"],
            "explicit_compare_entries": ["M25.A", "M26.A"],
        },
    )

    assert stage["required_test_contracts"] == ["m26.full_hidden_state_bridge_organism"]
    assert stage["historical_comparison_families"] == ["M25"]
    assert [target["target"] for target in stage["comparison_targets"]] == ["M25.A", "M26.A"]
    assert stage["selected_upstream"] == "M25.A"


def test_control_plane_stage_tracks_test_matrix_and_whole_grid_consumers() -> None:
    program_map = load_script_module("build_ablation_program_map_control_plane_test", "scripts/control_plane/build_ablation_program_map.py")
    program_spine = load_script_module("build_ablation_program_spine_control_plane_test", "scripts/control_plane/build_ablation_program_spine.py")

    history_row = next(row for row in program_map._ordered_family_rows({}) if row["family_key"] == "History")
    control_stage = program_spine._build_control_plane_stage(
        Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/test/ablation_history_manifest.json")
    )

    assert "scripts/control_plane/run_ablation_test_matrix.py" in history_row["script_paths"]
    assert "scripts/control_plane/run_whole_ablation_grid.py" in history_row["script_paths"]
    assert "airflow/dags/control_plane/lojban_ablation_test_matrix_dag.py" in history_row["dag_paths"]
    assert "airflow/dags/control_plane/lojban_whole_ablation_grid_dag.py" in history_row["dag_paths"]
    assert "scripts/control_plane/run_ablation_test_matrix.py" in control_stage["scripts"]
    assert "scripts/control_plane/run_whole_ablation_grid.py" in control_stage["scripts"]
    assert "airflow/dags/control_plane/lojban_ablation_test_matrix_dag.py" in control_stage["dags"]
    assert "airflow/dags/control_plane/lojban_whole_ablation_grid_dag.py" in control_stage["dags"]


def test_master_spine_exposes_m25_and_m26_child_dags() -> None:
    text = (REPO_ROOT / "airflow/dags/control_plane/lojban_ablation_master_spine_dag.py").read_text(encoding="utf-8")

    assert '"stage_key": "M25"' in text
    assert '"lojban_m25_emergent_bridi"' in text
    assert '"stage_key": "M26"' in text
    assert '"lojban_m26_end_to_end_loafman"' in text
    assert '"lojban_ablation_test_matrix"' in text


def test_whole_grid_dag_can_pin_ablation_test_matrix_manifest() -> None:
    text = (REPO_ROOT / "airflow/dags/control_plane/lojban_whole_ablation_grid_dag.py").read_text(encoding="utf-8")

    assert '"ablation_test_matrix_manifest": ""' in text
    assert "--ablation-test-matrix-manifest" in text
    assert "ablation_test_matrix_manifest" in text


def test_script_surface_registry_tracks_m26_paths() -> None:
    payload = json.loads((REPO_ROOT / "configs/script_surface_registry.json").read_text(encoding="utf-8"))

    assert "M26" in payload["series_order"]
    question = payload["series"]["M26"]["question"]
    assert question.startswith("End-to-end Lojban symbiote")
    assert "full-organism" in question or "spinal-cord" in question
    rules = payload["path_rules"]
    assert any(rule.get("series") == "M26" and "src/lojban_evolution/m26/**" in rule.get("globs", []) for rule in rules)
    assert any(rule.get("series") == "M26" and "scripts/m26/*.py" in rule.get("globs", []) for rule in rules)
