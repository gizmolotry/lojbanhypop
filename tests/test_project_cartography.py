from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from lojban_evolution.control_plane.series_registry import classify_surface_path, known_series, series_order


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cartography_module():
    module_path = REPO_ROOT / "scripts" / "control_plane" / "build_project_cartography.py"
    spec = importlib.util.spec_from_file_location("build_project_cartography", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_series_registry_classifies_branch_series_surfaces() -> None:
    cases = {
        "scripts/legacy/run_legacy_ablation_grid.py": "A-G",
        "scripts/legacy/run_true_coconut_h_series.py": "H",
        "scripts/legacy/train_h5_persistent_vq_advisor.py": "H5",
        "scripts/legacy/eval_j_5.py": "J",
        "scripts/legacy/train_l_series_mvs.py": "L",
        "scripts/control_plane/run_whole_ablation_grid.py": "control_plane",
        "scripts/m21/run_m21_dynamic_bridi_suite.py": "M21",
        "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py": "M21",
        "src/lojban_evolution/m21/bridi.py": "M21",
        "scripts/m22/run_m22_semantic_generalization.py": "M22",
        "airflow/dags/m22/lojban_m22_semantic_generalization_dag.py": "M22",
        "src/lojban_evolution/m22/generalization.py": "M22",
        "scripts/m23/run_m23_relevance_suite.py": "M23",
        "airflow/dags/m23/lojban_m23_relevance_router_dag.py": "M23",
        "src/lojban_evolution/m23/relevance.py": "M23",
        "src/lojban_evolution/m24/compression.py": "M24",
        "scripts/m24/run_m24_substrate_compression_suite.py": "M24",
        "airflow/dags/m24/lojban_m24_substrate_compression_dag.py": "M24",
        "src/lojban_evolution/m25/emergent_bridi.py": "M25",
        "scripts/m25/run_m25_emergent_bridi_suite.py": "M25",
        "airflow/dags/m25/lojban_m25_emergent_bridi_dag.py": "M25",
    }

    for path, expected_series in cases.items():
        assert classify_surface_path(path).series == expected_series


def test_series_registry_preserves_a_to_m24_ordering() -> None:
    order = series_order()

    assert order[:5] == ["A-G", "H", "H5", "J", "L"]
    assert "M22" in order
    assert order.index("M21") < order.index("M22")
    assert "M23" in order
    assert order.index("M22") < order.index("M23")
    assert "M24" in order
    assert order.index("M23") < order.index("M24")
    assert "M25" in order
    assert order.index("M24") < order.index("M25")
    assert {"A-G", "H", "H5", "J", "L", "M19", "M20", "M21", "M22", "M23", "M24", "M25"}.issubset(known_series())


def test_cartography_infers_m24_and_m25_standard_family_paths() -> None:
    cartography = _load_cartography_module()

    cases = {
        "src/lojban_evolution/m24/compression.py": "M24",
        "scripts/m24/run_m24_substrate_compression_suite.py": "M24",
        "airflow/dags/m24/lojban_m24_substrate_compression_dag.py": "M24",
        "src/lojban_evolution/m25/emergent_bridi.py": "M25",
        "scripts/m25/run_m25_emergent_bridi_suite.py": "M25",
        "airflow/dags/m25/lojban_m25_emergent_bridi_dag.py": "M25",
    }

    for path, expected_series in cases.items():
        assert cartography._infer_family(path) == expected_series


def test_cartography_has_no_unclassified_runnable_surfaces() -> None:
    cartography = _load_cartography_module()

    entries = cartography.build_entries()
    runnable = [entry for entry in entries if entry.kind in {"script", "dag"}]

    assert runnable
    assert [entry.path for entry in runnable if entry.family == "unclassified"] == []


def test_cartography_keeps_generic_legacy_bucket_small_and_explicit() -> None:
    cartography = _load_cartography_module()

    entries = cartography.build_entries()
    generic_legacy = sorted(
        entry.path
        for entry in entries
        if entry.kind in {"script", "dag"} and entry.family == "legacy"
    )

    assert generic_legacy == [
        "scripts/legacy/eval_with_lms.py",
        "scripts/legacy/tasks.ps1",
    ]


def test_cartography_does_not_treat_local_readmes_as_scripts() -> None:
    cartography = _load_cartography_module()

    entries = cartography.build_entries()
    runnable_readmes = [
        entry.path
        for entry in entries
        if entry.path.endswith("README.md") and entry.kind in {"script", "dag"}
    ]

    assert runnable_readmes == []


def test_root_workspace_artifacts_are_not_tracked_runtime_surfaces() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    tracked = result.stdout.splitlines()
    forbidden_prefixes = (
        "RESULTS_",
        "ROOT_",
        "M21_",
        "m21_",
    )

    offenders = [
        path
        for path in tracked
        if "/" not in path and path.startswith(forbidden_prefixes) and path != "PROJECT_INDEX.md"
    ]

    assert offenders == []
