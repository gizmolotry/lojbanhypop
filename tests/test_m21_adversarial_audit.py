from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "m21" / "run_m21_adversarial_audit.py"
    spec = importlib.util.spec_from_file_location("run_m21_adversarial_audit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m21_adversarial_audit_summarizes_semantic_isolation_deltas() -> None:
    audit = _load_audit_module()
    rows = [
        _row("H", "heldout_paraphrase,clausal_permutation", 0.30, 0.25, 0.20),
        _row("I", "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train", 0.50, 0.40, 0.45),
        _row("J", "heldout_paraphrase,clausal_permutation,lexical_shift_train", 0.44, 0.35, 0.38),
        _row("K", "heldout_paraphrase,clausal_permutation,role_binding_train", 0.41, 0.34, 0.36),
        _row("L", "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train", 0.52, 0.42, 0.46, fraction=0.50),
    ]

    summary = audit._summarize(rows)

    assert summary["semantic_isolation_cell_count"] == 5.0
    assert summary["semantic_coverage_lexical_shift_effect_strict_accuracy_delta"] == pytest.approx(0.14)
    assert summary["semantic_coverage_role_binding_effect_strict_accuracy_delta"] == pytest.approx(0.11)
    assert summary["semantic_coverage_combined_effect_strict_accuracy_delta"] == pytest.approx(0.20)
    assert summary["semantic_coverage_fraction_effect_strict_accuracy_delta"] == pytest.approx(0.02)
    assert summary["semantic_coverage_lexical_shift_effect_worst_surface_accuracy_delta"] == pytest.approx(0.10)
    assert summary["semantic_coverage_role_binding_effect_judri_causal_delta_delta"] == pytest.approx(0.16)


def _row(
    cell: str,
    surfaces: str,
    strict: float,
    worst: float,
    judri_delta: float,
    *,
    fraction: float = 0.25,
) -> dict[str, object]:
    return {
        "cell_key": cell,
        "config": {
            "adversarial_train_fraction": fraction,
            "adversarial_train_surfaces": surfaces,
        },
        "metrics": {
            "adversarial_strict_accuracy": strict,
            "adversarial_worst_surface_accuracy": worst,
            "adversarial_judri_causal_delta": judri_delta,
            "adversarial_oov_token_rate": 0.1,
        },
    }
