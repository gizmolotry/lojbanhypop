from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "m21"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_m21_lock_suite import _statuses  # noqa: E402


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


def test_m21_adversarial_augmented_lock_requires_training_exposure() -> None:
    metrics = {
        "judri_bridge_gate_enabled": 1.0,
        "judri_bridge_gate_active_mean": 0.6,
        "judri_causal_delta": 0.8,
    }

    assert _statuses(metrics)["judri_gated_bridge"] is True
    assert _statuses(metrics)["adversarial_augmented_judri_gated_bridge"] is False
    metrics["adversarial_train_fraction"] = 0.25
    assert _statuses(metrics)["adversarial_augmented_judri_gated_bridge"] is True
