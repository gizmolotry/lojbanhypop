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
    "seed": 42,
    "num_queries": 8,
    "bottleneck_dim": 64,
    "scratchpad_length": 8,
    "random_scale": 0.05,
    "include_random_control": True,
    "include_replications": True,
    "replication_runs": 2,
    "use_existing_checkpoints": False,
    "force_retrain": False,
    "local_files_only": False,
    "mainline_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_suite",
    "grid_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_isolation_grid",
    "mainline_model_output_dir": "artifacts/models/m19/mainline_suite",
    "grid_model_output_dir": "artifacts/models/m19/grid",
    "run_id": "",
}


def _run_m19_mainline(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("mainline_output_dir", "")), "telemetry/raw")
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
        str(int(cfg.get("eval_size", DEFAULTS["eval_size"]))),
        "--epochs",
        str(int(cfg.get("epochs", DEFAULTS["epochs"]))),
        "--seed",
        str(int(cfg.get("seed", DEFAULTS["seed"]))),
        "--num-queries",
        str(int(cfg.get("num_queries", DEFAULTS["num_queries"]))),
        "--bottleneck-dim",
        str(int(cfg.get("bottleneck_dim", DEFAULTS["bottleneck_dim"]))),
        "--scratchpad-length",
        str(int(cfg.get("scratchpad_length", DEFAULTS["scratchpad_length"]))),
        "--random-scale",
        str(float(cfg.get("random_scale", DEFAULTS["random_scale"]))),
        "--output-root",
        output_dir,
        "--model-output-root",
        str(cfg.get("mainline_model_output_dir", DEFAULTS["mainline_model_output_dir"])),
        "--run-id",
        run_id,
    ]
    args.append("--include-random-control" if bool(cfg.get("include_random_control", True)) else "--no-include-random-control")
    run_repo_script("scripts/run_m19_mainline_suite.py", args)


def _run_m19_grid(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("grid_output_dir", "")), "telemetry/raw")
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
        str(int(cfg.get("eval_size", DEFAULTS["eval_size"]))),
        "--epochs",
        str(int(cfg.get("epochs", DEFAULTS["epochs"]))),
        "--learning-rate",
        str(float(cfg.get("learning_rate", DEFAULTS["learning_rate"]))),
        "--seed",
        str(int(cfg.get("seed", DEFAULTS["seed"]))),
        "--random-scale",
        str(float(cfg.get("random_scale", DEFAULTS["random_scale"]))),
        "--output-root",
        output_dir,
        "--model-output-root",
        str(cfg.get("grid_model_output_dir", DEFAULTS["grid_model_output_dir"])),
        "--replication-runs",
        str(int(cfg.get("replication_runs", DEFAULTS["replication_runs"]))),
        "--run-id",
        run_id,
    ]
    args.append("--include-random-control" if bool(cfg.get("include_random_control", True)) else "--no-include-random-control")
    args.append("--include-replications" if bool(cfg.get("include_replications", True)) else "--no-include-replications")
    args.append("--use-existing-checkpoints" if bool(cfg.get("use_existing_checkpoints", False)) else "--no-use-existing-checkpoints")
    args.append("--force-retrain" if bool(cfg.get("force_retrain", False)) else "--no-force-retrain")
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")
    run_repo_script("scripts/run_m19_isolation_grid.py", args)


with DAG(
    dag_id="lojban_m19_family",
    description="Canonical M19 family DAG: mainline suite plus isolation-grid follow-up under one Airflow surface.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "frontier", "family"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "data_path": Param("artifacts/datasets/m19_mixed_curriculum_v1.jsonl", type="string", minLength=1),
        "eval_data_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "audit_data_path": Param("artifacts/datasets/sanity_check_v1.jsonl", type="string", minLength=1),
        "eval_size": Param(50, type="integer", minimum=1),
        "epochs": Param(1, type="integer", minimum=1),
        "learning_rate": Param(1e-4, type="number", minimum=1e-8),
        "seed": Param(42, type="integer", minimum=0),
        "num_queries": Param(8, type="integer", minimum=1),
        "bottleneck_dim": Param(64, type="integer", minimum=1),
        "scratchpad_length": Param(8, type="integer", minimum=1),
        "random_scale": Param(0.05, type="number", minimum=0.0),
        "include_random_control": Param(True, type="boolean"),
        "include_replications": Param(True, type="boolean"),
        "replication_runs": Param(2, type="integer", minimum=0),
        "use_existing_checkpoints": Param(False, type="boolean"),
        "force_retrain": Param(False, type="boolean"),
        "local_files_only": Param(False, type="boolean"),
        "mainline_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_suite", type="string", minLength=1),
        "grid_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_isolation_grid", type="string", minLength=1),
        "mainline_model_output_dir": Param("artifacts/models/m19/mainline_suite", type="string", minLength=1),
        "grid_model_output_dir": Param("artifacts/models/m19/grid", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    mainline = PythonOperator(
        task_id="run_m19_mainline_suite",
        python_callable=_run_m19_mainline,
    )
    grid = PythonOperator(
        task_id="run_m19_isolation_grid",
        python_callable=_run_m19_grid,
    )

    mainline >> grid
