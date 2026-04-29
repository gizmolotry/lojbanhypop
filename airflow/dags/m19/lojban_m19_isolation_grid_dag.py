from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "data_path": "artifacts/datasets/m19_mixed_curriculum_v1.jsonl",
    "eval_data_path": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
    "audit_data_path": "artifacts/datasets/sanity_check_v1.jsonl",
    "eval_size": 50,
    "epochs": 1,
    "learning_rate": 1e-4,
    "seed": 19,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_isolation_grid",
    "model_output_dir": "artifacts/models/m19/grid",
    "use_existing_checkpoints": True,
    "force_retrain": False,
    "include_random_control": True,
    "replication_runs": 2,
    "local_files_only": False,
    "run_id": "",
}


def _run_m19_grid(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model",
        str(cfg.get("base_model", DEFAULTS["base_model"])),
        "--data-path",
        str(cfg.get("data_path", DEFAULTS["data_path"])),
        "--eval-data-path",
        str(cfg.get("eval_data_path", DEFAULTS["eval_data_path"])),
        "--audit-data-path",
        str(cfg.get("audit_data_path", DEFAULTS["audit_data_path"])),
        "--eval-size",
        str(int(cfg.get("eval_size", 50))),
        "--epochs",
        str(int(cfg.get("epochs", 1))),
        "--learning-rate",
        str(float(cfg.get("learning_rate", 1e-4))),
        "--seed",
        str(int(cfg.get("seed", 19))),
        "--output-root",
        output_dir,
        "--model-output-root",
        str(cfg.get("model_output_dir", DEFAULTS["model_output_dir"])),
        "--replication-runs",
        str(int(cfg.get("replication_runs", 2))),
        "--run-id",
        run_id,
    ]
    if bool(cfg.get("use_existing_checkpoints", True)):
        args.append("--use-existing-checkpoints")
    else:
        args.append("--no-use-existing-checkpoints")
    if bool(cfg.get("force_retrain", False)):
        args.append("--force-retrain")
    else:
        args.append("--no-force-retrain")
    if bool(cfg.get("include_random_control", True)):
        args.append("--include-random-control")
    else:
        args.append("--no-include-random-control")
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m19_isolation_grid.py", args)


with DAG(
    dag_id="lojban_m19_isolation_grid",
    description="Structured M19 isolation grid with one report per cell plus replication checks.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "grid", "benchmark", "replication"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "data_path": Param("artifacts/datasets/m19_mixed_curriculum_v1.jsonl", type="string", minLength=1),
        "eval_data_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "audit_data_path": Param("artifacts/datasets/sanity_check_v1.jsonl", type="string", minLength=1),
        "eval_size": Param(50, type="integer", minimum=1),
        "epochs": Param(1, type="integer", minimum=1),
        "learning_rate": Param(1e-4, type="number", minimum=1e-8),
        "seed": Param(19, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_isolation_grid", type="string", minLength=1),
        "model_output_dir": Param("artifacts/models/m19/grid", type="string", minLength=1),
        "use_existing_checkpoints": Param(True, type="boolean"),
        "force_retrain": Param(False, type="boolean"),
        "include_random_control": Param(True, type="boolean"),
        "replication_runs": Param(2, type="integer", minimum=0),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m19_isolation_grid",
        python_callable=_run_m19_grid,
    )
