from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M22_FAMILY_VERSION = "0.1"


M22_REGISTRY: dict[str, dict[str, Any]] = {
    "M22": {
        "family": "semantic_coverage_generalization",
        "implementation_label": "m21_dynamic_bridi_semantic_generalization_gate",
        "runner_scripts": {
            "generalization": "scripts/m22/run_m22_semantic_generalization.py",
        },
        "dags": {
            "generalization": "airflow/dags/m22/lojban_m22_semantic_generalization_dag.py",
        },
        "output_roots": {
            "generalization": "artifacts/runs/telemetry/raw/ablation/hypercube/m22_semantic_generalization",
        },
        "report_names": {
            "generalization": "m22_semantic_generalization_report.json",
        },
        "dataset_defaults": {
            "profile": "m21_dynamic_bridi_semantic_generalization_v1",
            "train_size": 6000,
            "eval_size": 6000,
        },
        "thesis": (
            "hold the M21 dynamic bridi architecture fixed and test whether semantic coverage "
            "generalizes beyond synthetic templates without sacrificing clean accuracy or judri causality."
        ),
        "architecture": {
            "stage_1": "reuse_m21_dynamic_bridi_qformer",
            "stage_2": "expand_allowed_semantic_training_surfaces",
            "stage_3": "audit_reserved_and_cross-template semantic surfaces",
            "stage_4": "compare_against_m21_control_direct_eval",
            "stage_5": "promote_only_if_generalization_improves_without_clean_or_judri_regression",
        },
        "parameter_axes": [
            "semantic_surface_family",
            "adversarial_train_fraction",
            "reserved_audit_surfaces",
            "m21_control_manifest",
            "promotion_tolerance",
        ],
        "comparison_targets": [
            "M21",
            "M21.1.H",
            "M21.1.I",
            "M21.1.J",
            "M21.1.K",
            "M21.1.L",
            "M21.1.M",
            "M21.1.N",
            "M21.1.O",
        ],
    }
}


def m22_track_spec(track: str = "M22") -> dict[str, Any]:
    return deepcopy(M22_REGISTRY[track])


def m22_default_output_root(kind: str) -> Path:
    return Path(M22_REGISTRY["M22"]["output_roots"][kind])
