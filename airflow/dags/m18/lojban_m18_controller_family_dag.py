from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "data_path": "",
    "epochs": 1,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family",
    "model_output_dir": "artifacts/models/m18/frontier",
    "skip_train": False,
    "run_id": "",
}


def _run_m18(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    data_path = str(cfg.get("data_path", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not data_path:
        raise ValueError("data_path is required")

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model",
        base_model,
        "--data-path",
        data_path,
        "--epochs",
        str(int(cfg.get("epochs", 1))),
        "--output-root",
        output_dir,
        "--model-output-root",
        str(cfg.get("model_output_dir", "artifacts/models/m18/frontier")),
        "--run-id",
        run_id,
    ]
    if bool(cfg.get("skip_train", False)):
        args.append("--skip-train")

    run_repo_script("scripts/run_m18_controller_family.py", args)


with DAG(
    dag_id="lojban_m18_controller_family",
    description="M18 controller family: salience, interpreter, joint controller, and audit surfaces.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m18", "controller", "frontier"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "data_path": Param("", type="string", minLength=1),
        "epochs": Param(1, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family", type="string", minLength=1),
        "model_output_dir": Param("artifacts/models/m18/frontier", type="string", minLength=1),
        "skip_train": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m18_controller_family",
        python_callable=_run_m18,
    )
