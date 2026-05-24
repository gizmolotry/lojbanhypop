from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _help(script: str) -> str:
    result = subprocess.run([sys.executable, str(REPO_ROOT / script), "--help"], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return result.stdout


def test_m23_train_cli_help() -> None:
    out = _help("scripts/m23/train_m23_relevance_router.py")
    assert "--relevance-rank-weight" in out
    assert "--use-relevance-router" in out


def test_m23_suite_cli_help() -> None:
    out = _help("scripts/m23/run_m23_relevance_suite.py")
    assert "--cell-list" in out
    assert "--seed-list" in out
