from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_rel_path: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_rel_path), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_run_experiment_help() -> None:
    out = _run_help("scripts/legacy/run_experiment.py")
    assert "usage:" in out.lower()
    assert "--iterations" in out


def test_run_phase_ablation_help() -> None:
    out = _run_help("scripts/control_plane/pipeline_eval_manifold.py")
    assert "usage:" in out.lower()
    assert "--input-artifact" in out
    assert "--output" in out


def test_build_mixed_dataset_help() -> None:
    out = _run_help("scripts/data/build_mixed_curriculum_dataset.py")
    assert "usage:" in out.lower()
    assert "--output" in out


def test_run_direct_unified_eval_help() -> None:
    out = _run_help("scripts/control_plane/run_direct_unified_eval.py")
    assert "usage:" in out.lower()
    assert "--family" in out
    assert "--execute-m19-direct" in out


def test_run_m19_integrity_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_integrity_suite.py")
    assert "usage:" in out.lower()
    assert "--train-data-path" in out
    assert "--bridge-path" in out


def test_run_m19_replication_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_replication_suite.py")
    assert "usage:" in out.lower()
    assert "--seed-list" in out
    assert "--eval-data-path" in out
    assert "--checkpoint-selection-policy" in out
    assert "--query-repulsion-weight" in out


def test_run_m19_stability_microgrid_help() -> None:
    out = _run_help("scripts/m19/run_m19_stability_microgrid.py")
    assert "usage:" in out.lower()
    assert "--learning-rate-list" in out
    assert "--augmentation-prob-list" in out
    assert "--format-augmentation-prob-list" in out


def test_run_m19_kill_test_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_kill_test_suite.py")
    assert "usage:" in out.lower()
    assert "--train-data-path" in out
    assert "--bridge-path" in out


def test_run_m19_dictionary_audit_help() -> None:
    out = _run_help("scripts/m19/run_m19_dictionary_audit.py")
    assert "usage:" in out.lower()
    assert "--bridge-spec" in out
    assert "--dataset-path" in out
    assert "--typed-slot-layout" in out


def test_run_m19_typed_physics_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_typed_physics_suite.py")
    assert "usage:" in out.lower()
    assert "--track" in out
    assert "--typed-slot-layout" in out


def test_run_m19_gumbel_and_hyperbolic_suite_help() -> None:
    gumbel = _run_help("scripts/m19/run_m19_gumbel_faithfulness_suite.py")
    hyper = _run_help("scripts/m19/run_m19_hyperbolic_faithfulness_suite.py")
    assert "--epochs" in gumbel
    assert "--poincare-curvature" in hyper
