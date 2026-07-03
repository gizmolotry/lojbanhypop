from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_program_map_classifies_m27_entries_and_layer() -> None:
    program_map = load_script_module("build_ablation_program_map_m27_test", "scripts/control_plane/build_ablation_program_map.py")

    assert program_map._family_key({"normalized_canonical_id": "M27.A"}) == "M27"
    assert program_map._program_layer("M27") == "coconut_bridi_runtime"

    families: dict[str, dict[str, object]] = {}
    program_map._merge_taxonomy_families(
        families,
        {"M27": {"architectural_thesis": "Coconut-Bridi recurrent runtime verification"}},
    )
    assert families["M27"]["program_layer"] == "coconut_bridi_runtime"
    assert "airflow/dags/m27/lojban_m27_coconut_bridi_runtime_dag.py" in families["M27"]["dag_paths"]


def test_program_spine_gives_m27_dedicated_layer() -> None:
    program_spine = load_script_module("build_ablation_program_spine_m27_test", "scripts/control_plane/build_ablation_program_spine.py")

    assert program_spine._major_program_layer(26) == "end_to_end_lojban_symbiote"
    assert program_spine._major_program_layer(27) == "coconut_bridi_runtime"
    assert "M27" in program_spine.M_STAGE_ORDER
    assert program_spine.EXTRA_STAGE_DAGS["M27"] == ["airflow/dags/m27/lojban_m27_coconut_bridi_runtime_dag.py"]


def test_program_spine_uses_taxonomy_contract_shape_for_m27() -> None:
    program_spine = load_script_module("build_ablation_program_spine_m27_contract_test", "scripts/control_plane/build_ablation_program_spine.py")

    stage = program_spine._build_major_stage(
        "M27",
        {"architectural_thesis": "Coconut-Bridi runtime"},
        [],
        [
            {
                "transition_id": "M26_to_M27",
                "to_major": "M27",
                "selected_upstream": "M26.A",
                "inherits_components": ["M26 bridge"],
            }
        ],
        {
            "historical_comparison_families": ["M26"],
            "required_test_contracts": ["m27.coconut_bridi_runtime"],
            "explicit_compare_entries": ["M26.A", "M27.A"],
        },
    )

    assert stage["required_test_contracts"] == ["m27.coconut_bridi_runtime"]
    assert stage["historical_comparison_families"] == ["M26"]
    assert [target["target"] for target in stage["comparison_targets"]] == ["M26.A", "M27.A"]
    assert stage["selected_upstream"] == "M26.A"


def test_master_spine_exposes_m27_child_dag() -> None:
    text = (REPO_ROOT / "airflow/dags/control_plane/lojban_ablation_master_spine_dag.py").read_text(encoding="utf-8")

    assert '"stage_key": "M27"' in text
    assert '"lojban_m27_coconut_bridi_runtime"' in text


def test_script_surface_registry_tracks_m27_paths() -> None:
    payload = json.loads((REPO_ROOT / "configs/script_surface_registry.json").read_text(encoding="utf-8"))

    assert "M27" in payload["series_order"]
    question = payload["series"]["M27"]["question"]
    assert "Coconut-Bridi runtime" in question or "Autoregressive Coconut-Bridi runtime" in question
    rules = payload["path_rules"]
    assert any(rule.get("series") == "M27" and "src/lojban_evolution/m27/**" in rule.get("globs", []) for rule in rules)
    assert any(rule.get("series") == "M27" and "scripts/m27/*.py" in rule.get("globs", []) for rule in rules)


def test_ablation_test_matrix_tracks_m27_tests() -> None:
    payload = json.loads((REPO_ROOT / "configs/ablation_test_matrix.json").read_text(encoding="utf-8"))
    group = next(row for row in payload["test_groups"] if row["group_id"] == "m27.coconut_bridi_runtime")

    assert group["series"] == "M27"
    assert {"smoke", "architecture", "full"} <= set(group["lanes"])
    assert "tests/test_m27_coconut_bridi_runtime.py" in group["pytest_paths"]
    assert "tests/test_m27_cli_smoke.py" in group["pytest_paths"]
