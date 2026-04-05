from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_baseline_manifest_path,
    validate_output_partition,
)


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "checkpoint": "",
    "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
    "dataset_size": 128,
    "train_steps": 160,
    "lr": 0.03,
    "margin": 0.5,
    "seed": 42,
    "relation_vocab": 5,
    "max_slots": 8,
    "max_logic_new_tokens": 48,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding",
    "local_files_only": False,
    "run_id": "",
}


def _run_m4_2(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    checkpoint = str(cfg.get("checkpoint", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")
    if not checkpoint:
        raise ValueError("checkpoint is required")
    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())
    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model", base_model,
        "--adapter", adapter,
        "--checkpoint", checkpoint,
        "--baseline-manifest", baseline_manifest,
        "--dataset-size", str(int(cfg.get("dataset_size", 128))),
        "--train-steps", str(int(cfg.get("train_steps", 160))),
        "--lr", str(float(cfg.get("lr", 0.03))),
        "--margin", str(float(cfg.get("margin", 0.5))),
        "--seed", str(int(cfg.get("seed", 42))),
        "--relation-vocab", str(int(cfg.get("relation_vocab", 5))),
        "--max-slots", str(int(cfg.get("max_slots", 8))),
        "--max-logic-new-tokens", str(int(cfg.get("max_logic_new_tokens", 48))),
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")
    run_repo_script("scripts/run_m4_2_predicate_grounding.py", args)


with DAG(
    dag_id="lojban_m4_2_predicate_grounding",
    description="M4.2 predicate grounding on System 1 relation head with post-grounding semantic probe.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m4-2", "grounding", "semantic"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "dataset_size": Param(128, type="integer", minimum=2),
        "train_steps": Param(160, type="integer", minimum=1),
        "lr": Param(0.03, type="number", minimum=1e-6),
        "margin": Param(0.5, type="number", minimum=0.0),
        "seed": Param(42, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "max_slots": Param(8, type="integer", minimum=1),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m4_2_predicate_grounding",
        python_callable=_run_m4_2,
    )
