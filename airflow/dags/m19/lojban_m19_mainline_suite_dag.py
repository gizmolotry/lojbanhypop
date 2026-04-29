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
    "seed": 42,
    "track": "M19",
    "dynamic_pacing": False,
    "num_queries": 8,
    "bottleneck_dim": 64,
    "scratchpad_length": 8,
    "min_latent_steps": 4,
    "max_latent_steps": 64,
    "random_scale": 0.05,
    "include_random_control": True,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_suite",
    "model_output_dir": "artifacts/models/m19/mainline_suite",
    "static_bridge_path": "artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt",
    "static_cell_id": "M19.3_8Q_128D_8S",
    "static_num_queries": 8,
    "static_bottleneck_dim": 128,
    "static_scratchpad_length": 8,
    "run_id": "",
}


def _run_m19(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model",
        str(cfg.get("base_model")),
        "--data-path",
        str(cfg.get("data_path")),
        "--eval-data-path",
        str(cfg.get("eval_data_path")),
        "--audit-data-path",
        str(cfg.get("audit_data_path")),
        "--eval-size",
        str(int(cfg.get("eval_size", 50))),
        "--epochs",
        str(int(cfg.get("epochs", 1))),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--track",
        str(cfg.get("track", "M19")),
        "--num-queries",
        str(int(cfg.get("num_queries", 8))),
        "--bottleneck-dim",
        str(int(cfg.get("bottleneck_dim", 64))),
        "--scratchpad-length",
        str(int(cfg.get("scratchpad_length", 8))),
        "--min-latent-steps",
        str(int(cfg.get("min_latent_steps", 4))),
        "--max-latent-steps",
        str(int(cfg.get("max_latent_steps", 64))),
        "--random-scale",
        str(float(cfg.get("random_scale", 0.05))),
        "--output-root",
        output_dir,
        "--model-output-root",
        str(cfg.get("model_output_dir", "artifacts/models/m19/mainline_suite")),
        "--static-cell-id",
        str(cfg.get("static_cell_id", "M19.3_8Q_128D_8S")),
        "--static-num-queries",
        str(int(cfg.get("static_num_queries", 8))),
        "--static-bottleneck-dim",
        str(int(cfg.get("static_bottleneck_dim", 128))),
        "--static-scratchpad-length",
        str(int(cfg.get("static_scratchpad_length", 8))),
        "--run-id",
        run_id,
    ]
    if str(cfg.get("static_bridge_path", "")).strip():
        args.extend(["--static-bridge-path", str(cfg.get("static_bridge_path"))])
    if bool(cfg.get("dynamic_pacing", False)):
        args.append("--dynamic-pacing")
    args.append("--include-random-control" if bool(cfg.get("include_random_control", True)) else "--no-include-random-control")
    run_repo_script("scripts/m19/run_m19_mainline_suite.py", args)


with DAG(
    dag_id="lojban_m19_mainline_suite",
    description="M19 mainline neuro-symbolic runway suite with train, audit, and godtier benchmark reports.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "frontier", "mainline", "scratchpad"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "data_path": Param("artifacts/datasets/m19_mixed_curriculum_v1.jsonl", type="string", minLength=1),
        "eval_data_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "audit_data_path": Param("artifacts/datasets/sanity_check_v1.jsonl", type="string", minLength=1),
        "eval_size": Param(50, type="integer", minimum=1),
        "epochs": Param(1, type="integer", minimum=0),
        "seed": Param(42, type="integer", minimum=0),
        "track": Param("M19", type="string", minLength=1),
        "dynamic_pacing": Param(False, type="boolean"),
        "num_queries": Param(8, type="integer", minimum=1),
        "bottleneck_dim": Param(64, type="integer", minimum=1),
        "scratchpad_length": Param(8, type="integer", minimum=1),
        "min_latent_steps": Param(4, type="integer", minimum=1),
        "max_latent_steps": Param(64, type="integer", minimum=1),
        "random_scale": Param(0.05, type="number", minimum=0.0),
        "include_random_control": Param(True, type="boolean"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_suite", type="string", minLength=1),
        "model_output_dir": Param("artifacts/models/m19/mainline_suite", type="string", minLength=1),
        "static_bridge_path": Param("artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt", type="string"),
        "static_cell_id": Param("M19.3_8Q_128D_8S", type="string", minLength=1),
        "static_num_queries": Param(8, type="integer", minimum=1),
        "static_bottleneck_dim": Param(128, type="integer", minimum=1),
        "static_scratchpad_length": Param(8, type="integer", minimum=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m19_mainline_suite",
        python_callable=_run_m19,
    )
