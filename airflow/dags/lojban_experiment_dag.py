from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, validate_output_dir


DEFAULTS = {
    "iterations": 6,
    "seed": 7,
    "dataset_size": 1000,
    "max_accept": 3,
    "output_dir": "artifacts/runs",
}


def _run_experiment(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    config = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_dir(str(config["output_dir"]))
    args = [
        "--iterations",
        str(config["iterations"]),
        "--seed",
        str(config["seed"]),
        "--dataset-size",
        str(config["dataset_size"]),
        "--max-accept",
        str(config["max_accept"]),
        "--output-dir",
        output_dir,
    ]
    run_repo_script("scripts/run_experiment.py", args)


with DAG(
    dag_id="lojban_identity_experiment",
    description="Wrapper DAG for scripts/run_experiment.py",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "experiment"],
    params={
        "iterations": Param(6, type="integer", minimum=1),
        "seed": Param(7, type="integer", minimum=0),
        "dataset_size": Param(1000, type="integer", minimum=10),
        "max_accept": Param(3, type="integer", minimum=1),
        "output_dir": Param("artifacts/runs", type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_identity_experiment",
        python_callable=_run_experiment,
    )
