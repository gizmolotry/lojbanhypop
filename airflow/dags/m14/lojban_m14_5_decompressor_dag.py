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
    "base_model": "archive/results/m9/active/RESULTS_M9_SYNCED/synced_model",
    "adapter": "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5",
    "checkpoint": "RESULTS_M9_PHASE3/m11_s1_v2_geometry.pt",
    "train_steps": 0,
    "eval_size": 20,
    "lr": 1e-4,
    "layer_index": 12,
    "scratchpad_alpha": 1.0,
    "seed_token": "<loj_seed>",
    "runway_token": "<loj_i>",
    "max_runway_length": 10,
    "seed": 42,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m14_5_decompressor",
    "local_files_only": True,
    "run_id": "",
}


def _run_m14_5(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    checkpoint = str(cfg.get("checkpoint", "")).strip()
    
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model", base_model,
        "--adapter", adapter,
        "--checkpoint", checkpoint,
        "--eval-size", str(int(cfg.get("eval_size", 20))),
        "--layer-index", str(int(cfg.get("layer_index", 12))),
        "--scratchpad-alpha", str(float(cfg.get("scratchpad_alpha", 1.0))),
        "--seed-token", str(cfg.get("seed_token", "<loj_seed>")),
        "--max-runway-length", str(int(cfg.get("max_runway_length", 10))),
        "--seed", str(int(cfg.get("seed", 42))),
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if bool(cfg.get("local_files_only", True)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m14_5_decompressor.py", args)


with DAG(
    dag_id="lojban_m14_5_decompressor",
    description="M14.5 Continuous Decompressor: validate if continuous seeds can causally constrain discrete runways.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m14.5", "decompressor", "unspooling"],
    params={
        "base_model": Param(DEFAULTS["base_model"], type="string"),
        "adapter": Param(DEFAULTS["adapter"], type="string"),
        "checkpoint": Param(DEFAULTS["checkpoint"], type="string"),
        "eval_size": Param(20, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "scratchpad_alpha": Param(1.0, type="number", minimum=0.0),
        "seed_token": Param("<loj_seed>", type="string"),
        "max_runway_length": Param(10, type="integer", minimum=1),
        "seed": Param(42, type="integer", minimum=0),
        "output_dir": Param(DEFAULTS["output_dir"], type="string"),
        "local_files_only": Param(True, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m14_5_decompressor",
        python_callable=_run_m14_5,
    )
