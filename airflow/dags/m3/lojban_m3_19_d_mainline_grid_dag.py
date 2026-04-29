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
    "pack_size": 512,
    "pack_seed": 19,
    "train_steps": 8,
    "eval_size": 16,
    "lr": 3e-4,
    "bottleneck_dim": 64,
    "num_return_tokens": 3,
    "max_logic_new_tokens": 48,
    "layer_index": 12,
    "relation_vocab": 5,
    "var_min_id": 5,
    "answer_weight": 1.0,
    "margin": 0.2,
    "return_norm_weight": 0.01,
    "continuation_target_max_tokens": 5,
    "residual_guard_weight": 5.0,
    "hybrid_token_weight": 0.2,
    "hybrid_residual_weight": 1.0,
    "hybrid_token_scale": 0.15,
    "continuation_eval_tokens": 5,
    "seed": 42,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid",
    "local_files_only": False,
    "run_id": "",
}


def _run_m3_19(**context: object) -> None:
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
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model", base_model,
        "--adapter", adapter,
        "--checkpoint", checkpoint,
        "--baseline-manifest", baseline_manifest,
        "--pack-size", str(int(cfg.get("pack_size", 512))),
        "--pack-seed", str(int(cfg.get("pack_seed", 19))),
        "--train-steps", str(int(cfg.get("train_steps", 8))),
        "--eval-size", str(int(cfg.get("eval_size", 16))),
        "--lr", str(float(cfg.get("lr", 3e-4))),
        "--bottleneck-dim", str(int(cfg.get("bottleneck_dim", 64))),
        "--num-return-tokens", str(int(cfg.get("num_return_tokens", 3))),
        "--max-logic-new-tokens", str(int(cfg.get("max_logic_new_tokens", 48))),
        "--layer-index", str(int(cfg.get("layer_index", 12))),
        "--relation-vocab", str(int(cfg.get("relation_vocab", 5))),
        "--var-min-id", str(int(cfg.get("var_min_id", 5))),
        "--answer-weight", str(float(cfg.get("answer_weight", 1.0))),
        "--margin", str(float(cfg.get("margin", 0.2))),
        "--return-norm-weight", str(float(cfg.get("return_norm_weight", 0.01))),
        "--continuation-target-max-tokens", str(int(cfg.get("continuation_target_max_tokens", 5))),
        "--residual-guard-weight", str(float(cfg.get("residual_guard_weight", 5.0))),
        "--hybrid-token-weight", str(float(cfg.get("hybrid_token_weight", 0.2))),
        "--hybrid-residual-weight", str(float(cfg.get("hybrid_residual_weight", 1.0))),
        "--hybrid-token-scale", str(float(cfg.get("hybrid_token_scale", 0.15))),
        "--continuation-eval-tokens", str(int(cfg.get("continuation_eval_tokens", 5))),
        "--seed", str(int(cfg.get("seed", 42))),
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m3_19_d_mainline_grid.py", args)


with DAG(
    dag_id="lojban_m3_19_d_mainline_grid",
    description="M3.19 D-mainline grid over rich resumption supervision and residual guardrail thresholds.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-19", "reentry", "d-mainline", "grid"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "pack_size": Param(512, type="integer", minimum=8),
        "pack_seed": Param(19, type="integer", minimum=0),
        "train_steps": Param(8, type="integer", minimum=0),
        "eval_size": Param(16, type="integer", minimum=2),
        "lr": Param(3e-4, type="number", minimum=1e-8),
        "bottleneck_dim": Param(64, type="integer", minimum=1),
        "num_return_tokens": Param(3, type="integer", minimum=1),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "var_min_id": Param(5, type="integer", minimum=1),
        "answer_weight": Param(1.0, type="number", minimum=0.0),
        "margin": Param(0.2, type="number", minimum=0.0),
        "return_norm_weight": Param(0.01, type="number", minimum=0.0),
        "continuation_target_max_tokens": Param(5, type="integer", minimum=1),
        "residual_guard_weight": Param(5.0, type="number", minimum=0.0),
        "hybrid_token_weight": Param(0.2, type="number", minimum=0.0),
        "hybrid_residual_weight": Param(1.0, type="number", minimum=0.0),
        "hybrid_token_scale": Param(0.15, type="number", minimum=0.0),
        "continuation_eval_tokens": Param(5, type="integer", minimum=1),
        "seed": Param(42, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_19_d_mainline_grid",
        python_callable=_run_m3_19,
    )
