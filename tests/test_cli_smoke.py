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
    out = _run_help("scripts/run_experiment.py")
    assert "usage:" in out.lower()
    assert "--iterations" in out


def test_run_phase_ablation_help() -> None:
    out = _run_help("scripts/run_phase_ablation.py")
    assert "usage:" in out.lower()
    assert "--dataset-size" in out


def test_build_mixed_dataset_help() -> None:
    out = _run_help("scripts/build_mixed_curriculum_dataset.py")
    assert "usage:" in out.lower()
    assert "--output" in out
