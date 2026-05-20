from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M21_FAMILY_VERSION = "0.3"
M21_DEFAULT_MAX_FRAMES = 6
M21_DEFAULT_MAX_CMAVO_PER_FRAME = 3
M21_DEFAULT_MAX_PLACES = 5
M21_DEFAULT_MAX_ENTITIES = 8


def m21_cell_id(cell_key: str) -> str:
    return f"M21.1.{str(cell_key).strip().upper()}"


M21_DYNAMIC_BRIDI_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m21_cell_id("A"),
        "lock": "dynamic_frame_count",
        "label": "Dynamic frame baseline",
        "variant": {"trace_weight": 1.0, "frame_necessity_weight": 0.25, "mdl_weight": 0.01},
    },
    {
        "cell_key": "B",
        "cell_id": m21_cell_id("B"),
        "lock": "cmavo_causality",
        "label": "Cmavo-causal pressure",
        "variant": {"trace_weight": 1.25, "frame_necessity_weight": 0.75, "mdl_weight": 0.01},
    },
    {
        "cell_key": "C",
        "cell_id": m21_cell_id("C"),
        "lock": "judri_binding_causality",
        "label": "Judri/brivi lock pressure",
        "variant": {"brivi_lock_weight": 2.0, "frame_necessity_weight": 0.75, "mdl_weight": 0.01},
    },
    {
        "cell_key": "D",
        "cell_id": m21_cell_id("D"),
        "lock": "frame_necessity",
        "label": "Frame necessity/dropout pressure",
        "variant": {"frame_necessity_weight": 1.25, "necessity_margin": 0.08, "mdl_weight": 0.01},
    },
    {
        "cell_key": "E",
        "cell_id": m21_cell_id("E"),
        "lock": "actual_bridge_transfer",
        "label": "Actual bridge adapter",
        "variant": {"answer_weight": 1.5, "frame_necessity_weight": 0.75, "mdl_weight": 0.005},
    },
    {
        "cell_key": "F",
        "cell_id": m21_cell_id("F"),
        "lock": "full_dynamic_bridi",
        "label": "Full dynamic bridi model",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
        },
    },
    {
        "cell_key": "G",
        "cell_id": m21_cell_id("G"),
        "lock": "judri_gated_bridge",
        "label": "Judri-gated bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
        },
    },
    {
        "cell_key": "H",
        "cell_id": m21_cell_id("H"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Adversarial-augmented judri-gated bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation",
        },
    },
    {
        "cell_key": "I",
        "cell_id": m21_cell_id("I"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Semantic-coverage adversarial judri-gated bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train",
        },
    },
    {
        "cell_key": "J",
        "cell_id": m21_cell_id("J"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Lexical-shift semantic coverage bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,lexical_shift_train",
        },
    },
    {
        "cell_key": "K",
        "cell_id": m21_cell_id("K"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Role-binding semantic coverage bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,role_binding_train",
        },
    },
    {
        "cell_key": "L",
        "cell_id": m21_cell_id("L"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "High-fraction semantic coverage bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.5,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train",
        },
    },

    {
        "cell_key": "M",
        "cell_id": m21_cell_id("M"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Role-binding curriculum bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train",
        },
    },
    {
        "cell_key": "N",
        "cell_id": m21_cell_id("N"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Swap-focused role-binding curriculum bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,role_binding_swap_train",
        },
    },
    {
        "cell_key": "O",
        "cell_id": m21_cell_id("O"),
        "lock": "adversarial_augmented_judri_gated_bridge",
        "label": "Higher-fraction role-binding curriculum bridge",
        "variant": {
            "trace_weight": 1.25,
            "answer_weight": 1.25,
            "counterfactual_weight": 1.25,
            "brivi_lock_weight": 1.5,
            "frame_necessity_weight": 1.0,
            "mdl_weight": 0.01,
            "judri_bridge_gate": True,
            "judri_bridge_gate_temperature": 1.0,
            "adversarial_train_fraction": 0.35,
            "adversarial_train_surfaces": "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train",
        },
    },
]


M21_REGISTRY: dict[str, dict[str, Any]] = {
    "M21": {
        "family": "dynamic_bridi_qformer",
        "implementation_label": "variable_length_gismu_cmavo_judri_trace",
        "runner_scripts": {
            "train": "scripts/m21/train_m21_dynamic_bridi.py",
            "synthetic_assay": "scripts/m21/run_m21_synthetic_assay_suite.py",
            "actual_bridge": "scripts/m21/run_m21_actual_bridge_suite.py",
            "lock_suite": "scripts/m21/run_m21_lock_suite.py",
            "suite": "scripts/m21/run_m21_dynamic_bridi_suite.py",
            "pointer_microgrid": "scripts/m21/run_m21_pointer_necessity_microgrid.py",
            "gauntlet": "scripts/m21/run_m21_gauntlet_suite.py",
            "adversarial_audit": "scripts/m21/run_m21_adversarial_audit.py",
        },
        "dags": {
            "train": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "synthetic_assay": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "actual_bridge": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "lock_suite": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "suite": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "pointer_microgrid": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "gauntlet": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
            "adversarial_audit": "airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py",
        },
        "output_roots": {
            "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_dynamic_bridi_train",
            "synthetic_assay": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_synthetic_assay_suite",
            "actual_bridge": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite",
            "lock_suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_lock_suite",
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_dynamic_bridi_suite",
            "pointer_microgrid": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_pointer_necessity_microgrid",
            "gauntlet": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_gauntlet_suite",
            "adversarial_audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_adversarial_audit",
        },
        "report_names": {
            "train": "m21_dynamic_bridi_train_report.json",
            "synthetic_assay": "m21_synthetic_assay_report.json",
            "actual_bridge": "m21_actual_bridge_report.json",
            "lock_suite": "m21_lock_suite_report.json",
            "suite": "m21_dynamic_bridi_suite_report.json",
            "pointer_microgrid": "m21_pointer_necessity_microgrid_report.json",
            "gauntlet": "m21_gauntlet_report.json",
            "adversarial_audit": "m21_adversarial_audit_report.json",
        },
        "dataset_defaults": {
            "profile": "dynamic_bridi_synthetic_minimal_pairs_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "replace fixed typed slots with dynamic Lojbanic bridi frames: the model must learn "
            "the number of gismu predicate frames, cmavo modifiers, and judri/entity-place bindings, "
            "then prove those traces are causal."
        ),
        "architecture": {
            "stage_1": "dynamic_bridi_trace_synthetic_generator",
            "stage_2": "qformer_frame_queries_with_active_and_stop_heads",
            "stage_3": "gismu_cmavo_judri_supervised_reconstruction",
            "stage_4": "frame_necessity_ablation_contrast",
            "stage_5": "minimal_actual_bridge_adapter",
        },
        "parameter_axes": [
            "max_frames",
            "trace_weight",
            "counterfactual_weight",
            "brivi_lock_weight",
            "frame_necessity_weight",
            "pointer_necessity_weight",
            "judri_bridge_gate",
            "judri_bridge_gate_temperature",
            "adversarial_train_fraction",
            "adversarial_train_surfaces",
            "geometry_mode",
            "poincare_curvature",
            "mdl_weight",
            "seed",
        ],
        "comparison_targets": [
            "M20",
            "M19.31",
            "M21.1.A",
            "M21.1.F",
            "M21.1.G",
            "M21.1.H",
            "M21.1.I",
            "M21.1.J",
            "M21.1.K",
            "M21.1.L",
            "M21.1.M",
            "M21.1.N",
            "M21.1.O",
        ],
        "default_grid": deepcopy(M21_DYNAMIC_BRIDI_GRID),
    }
}


def m21_track_spec(track: str = "M21") -> dict[str, Any]:
    return deepcopy(M21_REGISTRY[track])


def m21_default_output_root(kind: str) -> Path:
    return Path(M21_REGISTRY["M21"]["output_roots"][kind])


def m21_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M21_DYNAMIC_BRIDI_GRID)
