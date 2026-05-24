from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m23_relevance_suite",
    "run_id": "",
    "seed_list": "23,29,31,37,41,43",
    "cell_list": "A,B",
    "train_size": 6000,
    "eval_size": 1500,
    "epochs": 16,
    "batch_size": 128,
    "device": "cuda",
}


def _run_m23_relevance_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    run_repo_script(
        "scripts/m23/run_m23_relevance_suite.py",
        [
            "--output-root", output_dir,
            "--run-id", run_id,
            "--seed-list", str(cfg.get("seed_list")),
            "--cell-list", str(cfg.get("cell_list")),
            "--train-size", str(int(cfg.get("train_size", 6000))),
            "--eval-size", str(int(cfg.get("eval_size", 1500))),
            "--epochs", str(int(cfg.get("epochs", 16))),
            "--batch-size", str(int(cfg.get("batch_size", 128))),
            "--device", str(cfg.get("device")),
        ],
    )


with DAG(
    dag_id="lojban_m23_relevance_router",
    description="M23 causal relevance router suite over dynamic bridi frames.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m23", "relevance-router"],
    params={
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "seed_list": Param(DEFAULTS["seed_list"], type="string", minLength=1),
        "cell_list": Param(DEFAULTS["cell_list"], type="string", minLength=1),
        "train_size": Param(DEFAULTS["train_size"], type="integer", minimum=1),
        "eval_size": Param(DEFAULTS["eval_size"], type="integer", minimum=1),
        "epochs": Param(DEFAULTS["epochs"], type="integer", minimum=1),
        "batch_size": Param(DEFAULTS["batch_size"], type="integer", minimum=1),
        "device": Param(DEFAULTS["device"], type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_m23_relevance_suite",
        python_callable=_run_m23_relevance_suite,
    )
