from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_input_artifact,
    validate_output_partition,
)


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw",
    "input_artifact": "",
    "run_id": "",
    "logical_accuracy": 0.6,
    "ce_loss_final": None,
    "cross_attention_gain": None,
}


def _compose_output_artifact(output_dir: str, run_id: str) -> str:
    filename = f"{run_id}_manifold_eval.json"
    if output_dir.startswith("s3://"):
        return f"{output_dir.rstrip('/')}/{filename}"
    return str(Path(output_dir) / filename)


def _run_eval_manifold(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    config = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(config["output_dir"]), "telemetry/raw")
    input_artifact = validate_input_artifact(str(config["input_artifact"]), "models/frozen_manifolds")
    run_id = sanitize_run_id(str(config.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_artifact = _compose_output_artifact(output_dir, run_id)
    args = [
        "--input-artifact",
        input_artifact,
        "--output",
        output_artifact,
        "--run-id",
        run_id,
        "--logical-accuracy",
        str(config["logical_accuracy"]),
    ]
    if config.get("ce_loss_final") is not None:
        args.extend(["--ce-loss-final", str(config["ce_loss_final"])])
    if config.get("cross_attention_gain") is not None:
        args.extend(["--cross-attention-gain", str(config["cross_attention_gain"])])
    run_repo_script("scripts/pipeline_eval_manifold.py", args)


with DAG(
    dag_id="lojban_eval_manifold",
    description="Wrapper DAG for scripts/pipeline_eval_manifold.py",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "pipeline", "eval"],
    params={
        "output_dir": Param("artifacts/runs/telemetry/raw", type="string", minLength=1),
        "input_artifact": Param("", type="string"),
        "run_id": Param("", type="string"),
        "logical_accuracy": Param(0.6, type="number", minimum=0, maximum=1),
        "ce_loss_final": Param(None, type=["null", "number"]),
        "cross_attention_gain": Param(None, type=["null", "number"]),
    },
) as dag:
    PythonOperator(
        task_id="run_eval_manifold",
        python_callable=_run_eval_manifold,
    )
