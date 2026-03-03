from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, validate_output_dir


DEFAULTS = {
    "dataset_size": 1000,
    "seed": 7,
    "iterations": 6,
    "max_accept": 3,
    "top_k": 10,
    "min_support": 30,
    "output_dir": "artifacts/runs",
}


def _run_phase_ablation(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    config = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_dir(str(config["output_dir"]))
    args = [
        "--dataset-size",
        str(config["dataset_size"]),
        "--seed",
        str(config["seed"]),
        "--iterations",
        str(config["iterations"]),
        "--max-accept",
        str(config["max_accept"]),
        "--top-k",
        str(config["top_k"]),
        "--min-support",
        str(config["min_support"]),
        "--output-dir",
        output_dir,
    ]
    run_repo_script("scripts/run_phase_ablation.py", args)


with DAG(
    dag_id="lojban_phase_ablation",
    description="Wrapper DAG for scripts/run_phase_ablation.py",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation"],
    params={
        "dataset_size": Param(1000, type="integer", minimum=10),
        "seed": Param(7, type="integer", minimum=0),
        "iterations": Param(6, type="integer", minimum=1),
        "max_accept": Param(3, type="integer", minimum=1),
        "top_k": Param(10, type="integer", minimum=1),
        "min_support": Param(30, type="integer", minimum=1),
        "output_dir": Param("artifacts/runs", type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_phase_ablation",
        python_callable=_run_phase_ablation,
    )
