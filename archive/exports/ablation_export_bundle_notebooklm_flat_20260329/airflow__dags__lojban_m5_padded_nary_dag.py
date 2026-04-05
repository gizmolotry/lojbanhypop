from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "train_steps": 24,
    "semantic_dataset_size": 180,
    "winograd_pack_size": 240,
    "seed": 7,
    "dataset_profile": "semantic_bench_v1",
    "difficulty_tier": "all",
    "layer_index": 12,
    "l_output_dir": "runs/l_series/m5_padded_nary",
    "report_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary",
    "local_files_only": False,
    "run_id": "",
}


def _run_m5_padded_nary(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")

    l_output_dir = str(cfg.get("l_output_dir", "")).strip()
    if not l_output_dir:
        raise ValueError("l_output_dir is required")
    report_output_dir = validate_output_partition(str(cfg.get("report_output_dir", "")).strip(), "telemetry/raw")
    _ = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--train-steps",
        str(int(cfg.get("train_steps", 24))),
        "--semantic-dataset-size",
        str(int(cfg.get("semantic_dataset_size", 180))),
        "--winograd-pack-size",
        str(int(cfg.get("winograd_pack_size", 240))),
        "--seed",
        str(int(cfg.get("seed", 7))),
        "--dataset-profile",
        str(cfg.get("dataset_profile", "semantic_bench_v1")),
        "--difficulty-tier",
        str(cfg.get("difficulty_tier", "all")),
        "--layer-index",
        str(int(cfg.get("layer_index", 12))),
        "--l-output-root",
        l_output_dir,
        "--report-output-root",
        report_output_dir,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m5_padded_nary_family.py", args)


with DAG(
    dag_id="lojban_m5_padded_nary",
    description="M5 padded n-ary auto-formalization family with GRL, counterfactual invariance, CPC, and PAD-masked graph bias.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m5", "padded-nary", "ablation"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "train_steps": Param(24, type="integer", minimum=1),
        "semantic_dataset_size": Param(180, type="integer", minimum=16),
        "winograd_pack_size": Param(240, type="integer", minimum=32),
        "seed": Param(7, type="integer", minimum=0),
        "dataset_profile": Param("semantic_bench_v1", type="string", minLength=1),
        "difficulty_tier": Param("all", type="string", minLength=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "l_output_dir": Param("runs/l_series/m5_padded_nary", type="string", minLength=1),
        "report_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m5_padded_nary_family",
        python_callable=_run_m5_padded_nary,
    )
