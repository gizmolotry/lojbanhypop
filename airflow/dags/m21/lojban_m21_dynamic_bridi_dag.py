from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "seed_list": "23,29",
    "cell_list": "A,B,C,D,E,F,G",
    "train_size": 6000,
    "eval_size": 1500,
    "epochs": 16,
    "batch_size": 128,
    "learning_rate": 0.002,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "max_frames": 6,
    "max_cmavo_per_frame": 3,
    "max_places": 5,
    "max_entities": 8,
    "trace_weight": 1.0,
    "answer_weight": 1.0,
    "counterfactual_weight": 1.0,
    "brivi_lock_weight": 1.0,
    "frame_necessity_weight": 0.5,
    "pointer_necessity_weight": 0.0,
    "pointer_necessity_margin": 0.05,
    "hyperbolic_topology_weight": 0.0,
    "mdl_weight": 0.01,
    "necessity_margin": 0.04,
    "geometry_mode": "euclidean",
    "poincare_curvature": 1.0,
    "poincare_max_norm": 0.99,
    "riemannian_gradient_scale": True,
    "judri_bridge_gate": False,
    "judri_bridge_gate_temperature": 1.0,
    "pointer_microgrid_weights": "0.0,0.25,0.5,1.0,2.0",
    "stable_threshold": 0.70,
    "device": "cpu",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_dynamic_bridi_suite",
    "pointer_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_pointer_necessity_microgrid",
    "gauntlet_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_gauntlet_suite",
    "run_id": "",
}


def _run_m21_dynamic_bridi(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--seed-list",
        str(cfg.get("seed_list", "23,29")),
        "--cell-list",
        str(cfg.get("cell_list", "A,B,C,D,E,F")),
        "--train-size",
        str(int(cfg.get("train_size", 6000))),
        "--eval-size",
        str(int(cfg.get("eval_size", 1500))),
        "--epochs",
        str(int(cfg.get("epochs", 16))),
        "--batch-size",
        str(int(cfg.get("batch_size", 128))),
        "--learning-rate",
        str(float(cfg.get("learning_rate", 0.002))),
        "--embedding-dim",
        str(int(cfg.get("embedding_dim", 64))),
        "--hidden-dim",
        str(int(cfg.get("hidden_dim", 128))),
        "--max-frames",
        str(int(cfg.get("max_frames", 6))),
        "--max-cmavo-per-frame",
        str(int(cfg.get("max_cmavo_per_frame", 3))),
        "--max-places",
        str(int(cfg.get("max_places", 5))),
        "--max-entities",
        str(int(cfg.get("max_entities", 8))),
        "--trace-weight",
        str(float(cfg.get("trace_weight", 1.0))),
        "--answer-weight",
        str(float(cfg.get("answer_weight", 1.0))),
        "--counterfactual-weight",
        str(float(cfg.get("counterfactual_weight", 1.0))),
        "--brivi-lock-weight",
        str(float(cfg.get("brivi_lock_weight", 1.0))),
        "--frame-necessity-weight",
        str(float(cfg.get("frame_necessity_weight", 0.5))),
        "--pointer-necessity-weight",
        str(float(cfg.get("pointer_necessity_weight", 0.0))),
        "--pointer-necessity-margin",
        str(float(cfg.get("pointer_necessity_margin", 0.05))),
        "--hyperbolic-topology-weight",
        str(float(cfg.get("hyperbolic_topology_weight", 0.0))),
        "--mdl-weight",
        str(float(cfg.get("mdl_weight", 0.01))),
        "--necessity-margin",
        str(float(cfg.get("necessity_margin", 0.04))),
        "--geometry-mode",
        str(cfg.get("geometry_mode", "euclidean")),
        "--poincare-curvature",
        str(float(cfg.get("poincare_curvature", 1.0))),
        "--poincare-max-norm",
        str(float(cfg.get("poincare_max_norm", 0.99))),
        "--riemannian-gradient-scale" if bool(cfg.get("riemannian_gradient_scale", True)) else "--no-riemannian-gradient-scale",
        "--judri-bridge-gate" if bool(cfg.get("judri_bridge_gate", False)) else "--no-judri-bridge-gate",
        "--judri-bridge-gate-temperature",
        str(float(cfg.get("judri_bridge_gate_temperature", 1.0))),
        "--stable-threshold",
        str(float(cfg.get("stable_threshold", 0.70))),
        "--device",
        str(cfg.get("device", "cpu")),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    run_repo_script("scripts/m21/run_m21_dynamic_bridi_suite.py", args)


def _run_m21_pointer_microgrid(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("pointer_output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--weights",
        str(cfg.get("pointer_microgrid_weights", "0.0,0.25,0.5,1.0,2.0")),
        "--seed-list",
        str(cfg.get("seed_list", "23,29")),
        "--train-size",
        str(int(cfg.get("train_size", 6000))),
        "--eval-size",
        str(int(cfg.get("eval_size", 1500))),
        "--epochs",
        str(int(cfg.get("epochs", 16))),
        "--batch-size",
        str(int(cfg.get("batch_size", 128))),
        "--learning-rate",
        str(float(cfg.get("learning_rate", 0.002))),
        "--pointer-necessity-margin",
        str(float(cfg.get("pointer_necessity_margin", 0.05))),
        "--frame-necessity-weight",
        str(float(cfg.get("frame_necessity_weight", 0.5))),
        "--brivi-lock-weight",
        str(float(cfg.get("brivi_lock_weight", 1.0))),
        "--trace-weight",
        str(float(cfg.get("trace_weight", 1.0))),
        "--answer-weight",
        str(float(cfg.get("answer_weight", 1.0))),
        "--counterfactual-weight",
        str(float(cfg.get("counterfactual_weight", 1.0))),
        "--mdl-weight",
        str(float(cfg.get("mdl_weight", 0.01))),
        "--geometry-mode",
        str(cfg.get("geometry_mode", "euclidean")),
        "--poincare-curvature",
        str(float(cfg.get("poincare_curvature", 1.0))),
        "--poincare-max-norm",
        str(float(cfg.get("poincare_max_norm", 0.99))),
        "--hyperbolic-topology-weight",
        str(float(cfg.get("hyperbolic_topology_weight", 0.0))),
        "--judri-bridge-gate" if bool(cfg.get("judri_bridge_gate", False)) else "--no-judri-bridge-gate",
        "--judri-bridge-gate-temperature",
        str(float(cfg.get("judri_bridge_gate_temperature", 1.0))),
        "--device",
        str(cfg.get("device", "cpu")),
        "--output-root",
        output_dir,
        "--run-id",
        f"{run_id}_pointer",
    ]
    run_repo_script("scripts/m21/run_m21_pointer_necessity_microgrid.py", args)


def _run_m21_gauntlet(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("gauntlet_output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--output-root",
        output_dir,
        "--run-id",
        f"{run_id}_gauntlet",
    ]
    run_repo_script("scripts/m21/run_m21_gauntlet_suite.py", args)


with DAG(
    dag_id="lojban_m21_dynamic_bridi",
    description="M21 dynamic Lojbanic bridi Q-former retraining suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m21", "bridi"],
    params={
        "seed_list": Param("23,29", type="string", minLength=1),
        "cell_list": Param("A,B,C,D,E,F,G", type="string", minLength=1),
        "train_size": Param(6000, type="integer", minimum=1),
        "eval_size": Param(1500, type="integer", minimum=1),
        "epochs": Param(16, type="integer", minimum=1),
        "batch_size": Param(128, type="integer", minimum=1),
        "learning_rate": Param(0.002, type="number", minimum=0.0),
        "embedding_dim": Param(64, type="integer", minimum=1),
        "hidden_dim": Param(128, type="integer", minimum=1),
        "max_frames": Param(6, type="integer", minimum=1),
        "max_cmavo_per_frame": Param(3, type="integer", minimum=1),
        "max_places": Param(5, type="integer", minimum=1),
        "max_entities": Param(8, type="integer", minimum=1),
        "trace_weight": Param(1.0, type="number", minimum=0.0),
        "answer_weight": Param(1.0, type="number", minimum=0.0),
        "counterfactual_weight": Param(1.0, type="number", minimum=0.0),
        "brivi_lock_weight": Param(1.0, type="number", minimum=0.0),
        "frame_necessity_weight": Param(0.5, type="number", minimum=0.0),
        "pointer_necessity_weight": Param(0.0, type="number", minimum=0.0),
        "pointer_necessity_margin": Param(0.05, type="number", minimum=0.0),
        "hyperbolic_topology_weight": Param(0.0, type="number", minimum=0.0),
        "mdl_weight": Param(0.01, type="number", minimum=0.0),
        "necessity_margin": Param(0.04, type="number", minimum=0.0),
        "geometry_mode": Param("euclidean", type="string", minLength=1),
        "poincare_curvature": Param(1.0, type="number", minimum=0.0),
        "poincare_max_norm": Param(0.99, type="number", minimum=0.0),
        "riemannian_gradient_scale": Param(True, type="boolean"),
        "judri_bridge_gate": Param(False, type="boolean"),
        "judri_bridge_gate_temperature": Param(1.0, type="number", minimum=0.0),
        "pointer_microgrid_weights": Param("0.0,0.25,0.5,1.0,2.0", type="string", minLength=1),
        "stable_threshold": Param(0.70, type="number", minimum=0.0),
        "device": Param("cpu", type="string", minLength=1),
        "output_dir": Param(
            "artifacts/runs/telemetry/raw/ablation/hypercube/m21_dynamic_bridi_suite",
            type="string",
            minLength=1,
        ),
        "pointer_output_dir": Param(
            "artifacts/runs/telemetry/raw/ablation/hypercube/m21_pointer_necessity_microgrid",
            type="string",
            minLength=1,
        ),
        "gauntlet_output_dir": Param(
            "artifacts/runs/telemetry/raw/ablation/hypercube/m21_gauntlet_suite",
            type="string",
            minLength=1,
        ),
        "run_id": Param("", type="string"),
    },
) as dag:
    suite = PythonOperator(task_id="run_m21_dynamic_bridi", python_callable=_run_m21_dynamic_bridi)
    pointer = PythonOperator(task_id="run_m21_pointer_necessity_microgrid", python_callable=_run_m21_pointer_microgrid)
    gauntlet = PythonOperator(task_id="run_m21_gauntlet", python_callable=_run_m21_gauntlet)
    [suite, pointer] >> gauntlet
