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


def test_m20_train_help() -> None:
    out = _run_help("scripts/m20/train_m20_dictionary.py")
    assert "usage:" in out.lower()
    assert "--codebook-size" in out
    assert "--quotient-invariance-weight" in out
    assert "--brivi-lock-weight" in out


def test_m20_dictionary_first_suite_help() -> None:
    out = _run_help("scripts/m20/run_m20_dictionary_first_suite.py")
    assert "usage:" in out.lower()
    assert "--cell-list" in out
    assert "--seed-list" in out
    assert "--stable-threshold" in out


def test_m20_induction_and_lock_suite_help() -> None:
    induction = _run_help("scripts/m20/run_m20_predicate_induction.py")
    locks = _run_help("scripts/m20/run_m20_lock_suite.py")
    assert "--dataset-size" in induction
    assert "--induction-report" in locks
    assert "--train-report" in locks
