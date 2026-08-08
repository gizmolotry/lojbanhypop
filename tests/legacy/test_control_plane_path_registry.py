from __future__ import annotations

import ast
from pathlib import Path

from lojban_evolution.control_plane.path_registry import (
    REPO_ROOT,
    canonical_repo_path,
    canonicalize_manifest_paths,
    canonicalize_path_list,
    latest_history_manifest,
    repo_relative,
)


def test_legacy_script_aliases_canonicalize_to_current_series_paths() -> None:
    assert canonical_repo_path("scripts/run_ablation_history_backfill.py") == (
        "scripts/control_plane/run_ablation_history_backfill.py"
    )
    assert canonical_repo_path("scripts/run_m18_controller_family.py") == "scripts/m18/run_m18_controller_family.py"
    assert canonical_repo_path("scripts/run_m21_dynamic_bridi_suite.py") == (
        "scripts/m21/run_m21_dynamic_bridi_suite.py"
    )
    assert canonical_repo_path("scripts/run_experiment.py") == "scripts/legacy/run_experiment.py"
    assert canonical_repo_path("./scripts/run_m18_controller_family.py") == "scripts/m18/run_m18_controller_family.py"
    assert canonical_repo_path(".github/workflows/build.yml") == ".github/workflows/build.yml"


def test_path_list_and_manifest_canonicalization_are_stable() -> None:
    assert canonicalize_path_list(
        [
            "scripts/run_m21_dynamic_bridi_suite.py",
            "scripts/m21/run_m21_dynamic_bridi_suite.py",
            "",
        ]
    ) == ["scripts/m21/run_m21_dynamic_bridi_suite.py"]

    manifest = {
        "scripts": ["scripts/run_m20_dictionary_first_suite.py"],
        "dags": ["airflow/dags/lojban_m21_dynamic_bridi_dag.py"],
    }

    canonicalize_manifest_paths(manifest)

    assert manifest == {
        "scripts": ["scripts/m20/run_m20_dictionary_first_suite.py"],
        "dags": ["airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py"],
    }


def test_repo_relative_and_latest_history_manifest_error_contract(tmp_path: Path) -> None:
    assert repo_relative(REPO_ROOT / "docs" / "PROJECT_INDEX.md") == "docs/PROJECT_INDEX.md"

    missing = tmp_path / "history"
    try:
        latest_history_manifest(missing)
    except FileNotFoundError as exc:
        assert "ablation_history_manifest.json" in str(exc)
    else:
        raise AssertionError("latest_history_manifest should fail clearly for an empty root")


def test_airflow_run_repo_script_targets_canonicalize_to_existing_files() -> None:
    missing: list[tuple[str, str, str]] = []
    for dag_path in (REPO_ROOT / "airflow" / "dags").rglob("*.py"):
        tree = ast.parse(dag_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name != "run_repo_script":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            raw_path = node.args[0].value
            canonical = canonical_repo_path(raw_path)
            if not (REPO_ROOT / canonical).exists():
                missing.append((dag_path.relative_to(REPO_ROOT).as_posix(), raw_path, canonical))

    assert missing == []
