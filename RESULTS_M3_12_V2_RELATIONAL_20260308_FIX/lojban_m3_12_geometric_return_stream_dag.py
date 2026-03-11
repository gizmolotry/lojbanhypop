from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "checkpoint": "",
    "train_steps": 200,
    "eval_size": 100,
    "lr": 3e-4,
    "bottleneck_dim": 32,
    "num_queries": 8,
    "max_logic_new_tokens": 48,
    "max_final_new_tokens": 16,
    "layer_index": 12,
    "relation_vocab": 5,
    "var_min_id": 5,
    "seed": 42,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_12_geometric_return_stream",
    "local_files_only": False,
    "run_id": "",
}


def _run_m3_12(**context: object) -> None:
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

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_root = f"{output_dir.rstrip('/')}/{run_id}"

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--checkpoint",
        checkpoint,
        "--train-steps",
        str(int(cfg.get("train_steps", 200))),
        "--eval-size",
        str(int(cfg.get("eval_size", 100))),
        "--lr",
        str(float(cfg.get("lr", 3e-4))),
        "--bottleneck-dim",
        str(int(cfg.get("bottleneck_dim", 32))),
        "--num-queries",
        str(int(cfg.get("num_queries", 8))),
        "--max-logic-new-tokens",
        str(int(cfg.get("max_logic_new_tokens", 48))),
        "--max-final-new-tokens",
        str(int(cfg.get("max_final_new_tokens", 16))),
        "--layer-index",
        str(int(cfg.get("layer_index", 12))),
        "--relation-vocab",
        str(int(cfg.get("relation_vocab", 5))),
        "--var-min-id",
        str(int(cfg.get("var_min_id", 5))),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--output-root",
        output_root,
        "--run-id",
        run_id,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m3_12_geometric_return_stream.py", args)


with DAG(
    dag_id="lojban_m3_12_geometric_return_stream",
    description="M3.12 geometric return-stream ablation (A/B/C) for Winograd-focused refinement.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-12", "return-stream", "ablation"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "train_steps": Param(200, type="integer", minimum=0),
        "eval_size": Param(100, type="integer", minimum=1),
        "lr": Param(3e-4, type="number", minimum=1e-8),
        "bottleneck_dim": Param(32, type="integer", minimum=1),
        "num_queries": Param(8, type="integer", minimum=1),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "max_final_new_tokens": Param(16, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "var_min_id": Param(5, type="integer", minimum=1),
        "seed": Param(42, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_12_geometric_return_stream", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_12_geometric_return_stream",
        python_callable=_run_m3_12,
    )

