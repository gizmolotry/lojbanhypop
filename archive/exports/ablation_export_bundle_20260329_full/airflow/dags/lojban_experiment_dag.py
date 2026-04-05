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
    validate_distribution_json_path,
    validate_output_partition,
)


DEFAULTS = {
    "output_dir": "artifacts/runs/models/frozen_manifolds",
    "run_id": "",
    "arity_violation_rate": 0.05,
    "surgery_trigger_count": 1,
    "ce_loss_final": 1.0,
    "cross_attention_gain": 0.1,
    "logical_accuracy": 0.6,
    "variable_token_distribution_json": "docs/variable_token_distribution_1995.example.json",
}


def _compose_output_artifact(output_dir: str, run_id: str) -> str:
    filename = f"{run_id}_grounded_reasoner_train.json"
    if output_dir.startswith("s3://"):
        return f"{output_dir.rstrip('/')}/{filename}"
    return str(Path(output_dir) / filename)


def _run_train_grounded_reasoner(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    config = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(config["output_dir"]), "models/frozen_manifolds")
    run_id = sanitize_run_id(str(config.get("run_id") or getattr(dag_run, "run_id", "manual")))
    variable_token_distribution_json = validate_distribution_json_path(str(config["variable_token_distribution_json"]))
    output_artifact = _compose_output_artifact(output_dir, run_id)
    args = [
        "--run-id",
        run_id,
        "--output",
        output_artifact,
        "--arity-violation-rate",
        str(config["arity_violation_rate"]),
        "--surgery-trigger-count",
        str(config["surgery_trigger_count"]),
        "--ce-loss-final",
        str(config["ce_loss_final"]),
        "--cross-attention-gain",
        str(config["cross_attention_gain"]),
        "--logical-accuracy",
        str(config["logical_accuracy"]),
        "--variable-token-distribution-json",
        variable_token_distribution_json,
    ]
    run_repo_script("scripts/pipeline_train_grounded_reasoner.py", args)


with DAG(
    dag_id="lojban_train_grounded_reasoner",
    description="Wrapper DAG for scripts/pipeline_train_grounded_reasoner.py",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "pipeline", "train"],
    params={
        "output_dir": Param("artifacts/runs/models/frozen_manifolds", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "arity_violation_rate": Param(0.05, type="number", minimum=0),
        "surgery_trigger_count": Param(1, type="integer", minimum=0),
        "ce_loss_final": Param(1.0, type="number"),
        "cross_attention_gain": Param(0.1, type="number"),
        "logical_accuracy": Param(0.6, type="number", minimum=0, maximum=1),
        "variable_token_distribution_json": Param(
            "docs/variable_token_distribution_1995.example.json",
            type="string",
            minLength=1,
        ),
    },
) as dag:
    PythonOperator(
        task_id="run_train_grounded_reasoner",
        python_callable=_run_train_grounded_reasoner,
    )
