from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill",
    "run_id": "",
    "include_git": False,
}


def _run_history_backfill(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    include_git = bool(cfg.get("include_git", False))

    args = [
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if include_git:
        args.append("--include-git")

    run_repo_script("scripts/run_ablation_history_backfill.py", args)


with DAG(
    dag_id="lojban_ablation_history_backfill",
    description="Canonical backfill of every tracked ablation family into one unified history ledger and normalized row export.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "history", "ledger", "dag"],
    params={
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "include_git": Param(False, type="boolean"),
    },
) as dag:
    PythonOperator(
        task_id="run_ablation_history_backfill",
        python_callable=_run_history_backfill,
    )
