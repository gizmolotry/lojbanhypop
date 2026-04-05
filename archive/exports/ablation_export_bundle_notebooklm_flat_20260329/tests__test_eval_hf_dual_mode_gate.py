from __future__ import annotations

from conftest import load_script_module


def test_dual_mode_gate_lift_detects_over_specialization():
    mod = load_script_module("eval_hf_dual_mode_gate", "scripts/eval_hf_dual_mode_gate.py")

    final_lift = -0.094
    symbolic_lift = 0.979
    threshold = 0.05

    passed_final = final_lift > threshold
    passed_symbolic = symbolic_lift > threshold
    gate = mod.gate_pass(
        mean_final_lift=final_lift,
        mean_symbolic_lift=symbolic_lift,
        min_final_lift=threshold,
        min_symbolic_lift=threshold,
    )

    assert gate is False
    assert passed_symbolic and not passed_final, (
        "Expected specialization signature: symbolic passes while final-answer fails."
    )
