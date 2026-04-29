from __future__ import annotations

from copy import deepcopy
from typing import Any


M18_FAMILY_VERSION = "1.0"


M18_FAMILY_REGISTRY: dict[str, dict[str, Any]] = {
    "M18": {
        "family": "controller_family",
        "implementation_label": "two_pass_salience_controller",
        "runner_script": "scripts/m18/run_m18_controller_family.py",
        "dag": "airflow/dags/lojban_m18_controller_family_dag.py",
        "output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family",
        "report_name": "m18_family_report.json",
        "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
        "thesis": "extract salient host-model structure, induce a small relation graph, then bias a second decoder pass instead of injecting a raw foreign state directly into answer rollout.",
        "architecture": {
            "stage_1": "pass_one_hidden_state_tap",
            "stage_2": "salience_selection",
            "stage_3": "graph_induction_and_bias_compilation",
            "stage_4": "biased_second_pass_generation",
        },
        "evaluation_surfaces": [
            "sapir_whorf_audit",
            "harmonized_audit",
            "hybrid_cot_audit",
        ],
    }
}


def m18_family_spec(track: str = "M18") -> dict[str, Any]:
    return deepcopy(M18_FAMILY_REGISTRY[track])
