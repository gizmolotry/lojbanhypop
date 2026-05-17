from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M20_FAMILY_VERSION = "0.1"
M20_DEFAULT_CODEBOOK_SIZE = 2000
M20_DEFAULT_MAX_ARITY = 3


def m20_cell_id(cell_key: str) -> str:
    return f"M20.1.{str(cell_key).strip().upper()}"


M20_DICTIONARY_FIRST_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m20_cell_id("A"),
        "lock": "dictionary_first_pretraining",
        "label": "Dictionary-first pretraining",
        "variant": {"factor_weight": 1.0, "dictionary_commitment_weight": 1.5, "quotient_invariance_weight": 1.0, "brivi_lock_weight": 1.0},
    },
    {
        "cell_key": "B",
        "cell_id": m20_cell_id("B"),
        "lock": "factorized_predicate_dictionary",
        "label": "Factorized predicate dictionary",
        "variant": {"factor_weight": 1.75, "dictionary_commitment_weight": 0.75, "quotient_invariance_weight": 1.0, "brivi_lock_weight": 1.0},
    },
    {
        "cell_key": "C",
        "cell_id": m20_cell_id("C"),
        "lock": "counterfactual_quotient_dictionary",
        "label": "Counterfactual quotient dictionary",
        "variant": {"factor_weight": 1.0, "dictionary_commitment_weight": 0.75, "quotient_invariance_weight": 4.0, "brivi_lock_weight": 1.0},
    },
    {
        "cell_key": "D",
        "cell_id": m20_cell_id("D"),
        "lock": "brivi_locked_predicate_formation",
        "label": "Brivi-locked predicate formation",
        "variant": {"factor_weight": 1.0, "dictionary_commitment_weight": 0.75, "quotient_invariance_weight": 1.0, "brivi_lock_weight": 3.0},
    },
    {
        "cell_key": "E",
        "cell_id": m20_cell_id("E"),
        "lock": "synthetic_world_pretraining",
        "label": "Synthetic world pretraining",
        "variant": {"factor_weight": 1.25, "dictionary_commitment_weight": 1.0, "quotient_invariance_weight": 2.0, "brivi_lock_weight": 1.25},
    },
    {
        "cell_key": "F",
        "cell_id": m20_cell_id("F"),
        "lock": "soft_dictionary_before_hard_dictionary",
        "label": "Soft-to-hard dictionary annealing",
        "variant": {
            "factor_weight": 1.0,
            "dictionary_commitment_weight": 0.75,
            "quotient_invariance_weight": 2.0,
            "brivi_lock_weight": 1.0,
            "temperature_start": 2.0,
            "temperature_end": 0.15,
        },
    },
]


M20_REGISTRY: dict[str, dict[str, Any]] = {
    "M20": {
        "family": "dictionary_first_predicate_induction",
        "implementation_label": "factorized_soft_dictionary_brivi_lock",
        "runner_scripts": {
            "train": "scripts/m20/train_m20_dictionary.py",
            "predicate_induction": "scripts/m20/run_m20_predicate_induction.py",
            "lock_suite": "scripts/m20/run_m20_lock_suite.py",
            "suite": "scripts/m20/run_m20_dictionary_first_suite.py",
        },
        "dags": {
            "train": "airflow/dags/m20/lojban_m20_dictionary_first_dag.py",
            "predicate_induction": "airflow/dags/m20/lojban_m20_dictionary_first_dag.py",
            "lock_suite": "airflow/dags/m20/lojban_m20_dictionary_first_dag.py",
            "suite": "airflow/dags/m20/lojban_m20_dictionary_first_dag.py",
        },
        "output_roots": {
            "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_train",
            "predicate_induction": "artifacts/runs/telemetry/raw/ablation/hypercube/m20_predicate_induction",
            "lock_suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m20_lock_suite",
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite",
        },
        "report_names": {
            "train": "m20_dictionary_train_report.json",
            "predicate_induction": "m20_predicate_induction_report.json",
            "lock_suite": "m20_lock_suite_report.json",
            "suite": "m20_dictionary_first_suite_report.json",
        },
        "dataset_defaults": {
            "profile": "synthetic_world_predicate_minimal_pairs_v1",
            "train_size": 2400,
            "eval_size": 600,
        },
        "thesis": (
            "pretrain a Lojban-inspired predicate dictionary before bridge work: soft dictionary "
            "tokens must predict factorized causal properties, remain invariant under entity swaps, "
            "and stay silent unless grounded by brivi arguments."
        ),
        "architecture": {
            "stage_1": "synthetic_world_predicate_minimal_pairs",
            "stage_2": "soft_dictionary_mixture_with_temperature_anneal",
            "stage_3": "factorized_predicate_heads_domain_polarity_relation_arity_roles",
            "stage_4": "counterfactual_quotient_invariance_over_entity_swaps",
            "stage_5": "brivi_gate_suppresses_ungrounded_predicates",
        },
        "parameter_axes": [
            "codebook_size",
            "temperature_start",
            "temperature_end",
            "quotient_invariance_weight",
            "brivi_lock_weight",
            "factor_weight",
            "seed",
        ],
        "comparison_targets": [
            "M19.31",
            "M19.dictionary_audit",
            "random_code_dictionary",
        ],
        "default_grid": deepcopy(M20_DICTIONARY_FIRST_GRID),
    }
}


def m20_track_spec(track: str = "M20") -> dict[str, Any]:
    return deepcopy(M20_REGISTRY[track])


def m20_default_output_root(kind: str) -> Path:
    return Path(M20_REGISTRY["M20"]["output_roots"][kind])


def m20_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M20_DICTIONARY_FIRST_GRID)
