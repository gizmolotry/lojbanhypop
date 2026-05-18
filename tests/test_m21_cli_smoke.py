from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m21_cli_help_surfaces() -> None:
    scripts = [
        "scripts/m21/train_m21_dynamic_bridi.py",
        "scripts/m21/run_m21_dynamic_bridi_suite.py",
        "scripts/m21/run_m21_synthetic_assay_suite.py",
        "scripts/m21/run_m21_actual_bridge_suite.py",
        "scripts/m21/run_m21_lock_suite.py",
        "scripts/m21/run_m21_pointer_necessity_microgrid.py",
        "scripts/m21/run_m21_gauntlet_suite.py",
        "scripts/m21/run_m21_adversarial_audit.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
