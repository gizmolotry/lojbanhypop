from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "output_dir": "artifacts/runs/models/frozen_manifolds",
    "run_id": "",
    "train_steps": 200,
    "rho": 0.2,
    "init_lambda": 0.0,
    "max_lambda": 100.0,
    "tier_a_lock_eps": 0.02,
    "tier_a_lock_window": 16,
}


def _run_l_series(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    if not str(cfg["base_model"]).strip():
        raise ValueError("base_model is required")
    if not str(cfg["adapter"]).strip():
        raise ValueError("adapter is required")

    output_dir = validate_output_partition(str(cfg["output_dir"]), "models/frozen_manifolds")
    if output_dir.startswith("s3://"):
        raise ValueError("L-series trainer writes local artifacts only. Use local output_dir and sync to S3 downstream.")

    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    run_repo_script(
        "scripts/train_l_series_mvs.py",
        [
            "--base-model",
            str(cfg["base_model"]),
            "--adapter",
            str(cfg["adapter"]),
            "--output-root",
            f"{output_dir.rstrip('/')}/{run_id}",
            "--train-steps",
            str(int(cfg["train_steps"])),
            "--rho",
            str(float(cfg["rho"])),
            "--init-lambda",
            str(float(cfg["init_lambda"])),
            "--max-lambda",
            str(float(cfg["max_lambda"])),
            "--tier-a-lock-eps",
            str(float(cfg["tier_a_lock_eps"])),
            "--tier-a-lock-window",
            str(int(cfg["tier_a_lock_window"])),
        ],
    )


with DAG(
    dag_id="lojban_l_series_train",
    description="L-Series MVS trainer with Lexicographic Augmented Lagrangian constraints.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "l-series", "train"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "output_dir": Param("artifacts/runs/models/frozen_manifolds", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "train_steps": Param(200, type="integer", minimum=1),
        "rho": Param(0.2, type="number", minimum=0),
        "init_lambda": Param(0.0, type="number", minimum=0),
        "max_lambda": Param(100.0, type="number", minimum=0),
        "tier_a_lock_eps": Param(0.02, type="number", minimum=0),
        "tier_a_lock_window": Param(16, type="integer", minimum=1),
    },
) as dag:
    PythonOperator(
        task_id="run_l_series",
        python_callable=_run_l_series,
    )
