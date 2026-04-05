from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "train_steps": 4,
    "semantic_dataset_size": 48,
    "winograd_pack_size": 64,
    "seed": 7,
    "dataset_profile": "semantic_bench_v1",
    "difficulty_tier": "all",
    "layer_index": 12,
    "max_chain_steps": 4,
    "l_output_dir": "runs/l_series/m5_2_autoregressive_chain",
    "report_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain",
    "local_files_only": False,
}


def _run_m5_2(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")

    report_output_dir = validate_output_partition(str(cfg.get("report_output_dir", "")).strip(), "telemetry/raw")

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--train-steps",
        str(int(cfg.get("train_steps", 4))),
        "--semantic-dataset-size",
        str(int(cfg.get("semantic_dataset_size", 48))),
        "--winograd-pack-size",
        str(int(cfg.get("winograd_pack_size", 64))),
        "--seed",
        str(int(cfg.get("seed", 7))),
        "--dataset-profile",
        str(cfg.get("dataset_profile", "semantic_bench_v1")),
        "--difficulty-tier",
        str(cfg.get("difficulty_tier", "all")),
        "--layer-index",
        str(int(cfg.get("layer_index", 12))),
        "--max-chain-steps",
        str(int(cfg.get("max_chain_steps", 4))),
        "--l-output-root",
        str(cfg.get("l_output_dir", DEFAULTS["l_output_dir"])),
        "--report-output-root",
        report_output_dir,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m5_2_autoregressive_chain.py", args)


with DAG(
    dag_id="lojban_m5_2_autoregressive_chain",
    description="M5.2 bounded autoregressive matrix-chain isolated test.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m5-2", "autoregressive-chain"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "train_steps": Param(4, type="integer", minimum=1),
        "semantic_dataset_size": Param(48, type="integer", minimum=8),
        "winograd_pack_size": Param(64, type="integer", minimum=8),
        "seed": Param(7, type="integer", minimum=0),
        "dataset_profile": Param("semantic_bench_v1", type="string", minLength=1),
        "difficulty_tier": Param("all", type="string", minLength=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "max_chain_steps": Param(4, type="integer", minimum=2),
        "l_output_dir": Param("runs/l_series/m5_2_autoregressive_chain", type="string", minLength=1),
        "report_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
    },
) as dag:
    PythonOperator(
        task_id="run_m5_2_autoregressive_chain",
        python_callable=_run_m5_2,
    )
