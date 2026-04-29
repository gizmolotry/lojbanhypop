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
    "forge_ckpt": "archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt",
    "adapter_ckpt": "archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m11_native_adapter.pt",
    "head_ckpt": "archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m11_native_head.pt",
    "train_steps": 0,
    "lr": 1e-4,
    "num_samples": 100,
    "disable_adapter": False,
    "skip_train": True,
    "port": 5555,
    "forge_startup_timeout": 45.0,
    "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m11_discriminative_suite",
    "local_files_only": False,
    "run_id": "",
}


def _run_m11_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())
    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model", str(cfg.get("base_model", "")).strip(),
        "--forge-ckpt", str(cfg.get("forge_ckpt", "")).strip(),
        "--adapter-ckpt", str(cfg.get("adapter_ckpt", "")).strip(),
        "--head-ckpt", str(cfg.get("head_ckpt", "")).strip(),
        "--train-steps", str(int(cfg.get("train_steps", 0))),
        "--lr", str(float(cfg.get("lr", 1e-4))),
        "--num-samples", str(int(cfg.get("num_samples", 100))),
        "--port", str(int(cfg.get("port", 5555))),
        "--forge-startup-timeout", str(float(cfg.get("forge_startup_timeout", 45.0))),
        "--baseline-manifest", baseline_manifest,
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if bool(cfg.get("disable_adapter", False)):
        args.append("--disable-adapter")
    if bool(cfg.get("skip_train", True)):
        args.append("--skip-train")
    else:
        args.append("--no-skip-train")
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m11_discriminative_suite.py", args)


with DAG(
    dag_id="lojban_m11_discriminative_suite",
    description="M11 discriminative bridge suite with forge orchestration and unified-ledger manifest output.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m11", "discriminative", "audit", "bridge"],
    params={
        "base_model": Param("archive/results/m9/active/RESULTS_M9_SYNCED/synced_model", type="string", minLength=1),
        "forge_ckpt": Param("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt", type="string", minLength=1),
        "adapter_ckpt": Param("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m11_native_adapter.pt", type="string", minLength=1),
        "head_ckpt": Param("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m11_native_head.pt", type="string", minLength=1),
        "train_steps": Param(0, type="integer", minimum=0),
        "lr": Param(1e-4, type="number", minimum=1e-8),
        "num_samples": Param(100, type="integer", minimum=1),
        "disable_adapter": Param(False, type="boolean"),
        "skip_train": Param(True, type="boolean"),
        "port": Param(5555, type="integer", minimum=1),
        "forge_startup_timeout": Param(45.0, type="number", minimum=1.0),
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m11_discriminative_suite", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m11_discriminative_suite",
        python_callable=_run_m11_suite,
    )
