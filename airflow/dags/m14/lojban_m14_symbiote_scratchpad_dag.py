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
    "train_steps": 8,
    "eval_size": 16,
    "lr": 3e-4,
    "bottleneck_dim": 64,
    "max_logic_new_tokens": 48,
    "layer_index": 12,
    "relation_vocab": 5,
    "var_min_id": 5,
    "answer_weight": 1.0,
    "margin": 0.2,
    "return_norm_weight": 0.01,
    "continuation_target_max_tokens": 5,
    "continuation_eval_tokens": 5,
    "scratchpad_token": "<symbiote>",
    "scratchpad_length": 4,
    "scratchpad_alpha": 1.0,
    "b_guard_threshold": 0.01,
    "c_guard_threshold": 0.05,
    "d_guard_threshold": 0.10,
    "residual_guard_weight": 5.0,
    "seed": 42,
    "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
    "upstream_m3_18_report": "",
    "upstream_m3_19_report": "",
    "upstream_m11_manifest": "",
    "pack_jsonl": "",
    "strict_balance": True,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m14_symbiote_scratchpad",
    "local_files_only": False,
    "run_id": "",
}


def _run_m14(**context: object) -> None:
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
        "--train-steps", str(int(cfg.get("train_steps", 8))),
        "--eval-size", str(int(cfg.get("eval_size", 16))),
        "--lr", str(float(cfg.get("lr", 3e-4))),
        "--bottleneck-dim", str(int(cfg.get("bottleneck_dim", 64))),
        "--max-logic-new-tokens", str(int(cfg.get("max_logic_new_tokens", 48))),
        "--layer-index", str(int(cfg.get("layer_index", 12))),
        "--relation-vocab", str(int(cfg.get("relation_vocab", 5))),
        "--var-min-id", str(int(cfg.get("var_min_id", 5))),
        "--answer-weight", str(float(cfg.get("answer_weight", 1.0))),
        "--margin", str(float(cfg.get("margin", 0.2))),
        "--return-norm-weight", str(float(cfg.get("return_norm_weight", 0.01))),
        "--continuation-target-max-tokens", str(int(cfg.get("continuation_target_max_tokens", 5))),
        "--continuation-eval-tokens", str(int(cfg.get("continuation_eval_tokens", 5))),
        "--scratchpad-token", str(cfg.get("scratchpad_token", "<symbiote>")),
        "--scratchpad-length", str(int(cfg.get("scratchpad_length", 4))),
        "--scratchpad-alpha", str(float(cfg.get("scratchpad_alpha", 1.0))),
        "--b-guard-threshold", str(float(cfg.get("b_guard_threshold", 0.01))),
        "--c-guard-threshold", str(float(cfg.get("c_guard_threshold", 0.05))),
        "--d-guard-threshold", str(float(cfg.get("d_guard_threshold", 0.10))),
        "--residual-guard-weight", str(float(cfg.get("residual_guard_weight", 5.0))),
        "--seed", str(int(cfg.get("seed", 42))),
        "--baseline-manifest", baseline_manifest,
        "--output-root", output_dir,
        "--run-id", run_id,
    ]
    if bool(cfg.get("strict_balance", True)):
        args.append("--strict-balance")
    else:
        args.append("--no-strict-balance")
    for key, flag in (
        ("upstream_m3_18_report", "--upstream-m3-18-report"),
        ("upstream_m3_19_report", "--upstream-m3-19-report"),
        ("upstream_m11_manifest", "--upstream-m11-manifest"),
        ("pack_jsonl", "--pack-jsonl"),
    ):
        value = str(cfg.get(key, "")).strip()
        if value:
            args.extend([flag, value])
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m14_symbiote_scratchpad.py", args)


with DAG(
    dag_id="lojban_m14_symbiote_scratchpad",
    description="M14 symbiote scratchpad: inject continuous advisor residuals into bounded scratchpad token positions before English continuation.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m14", "scratchpad", "symbiote", "reentry"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "train_steps": Param(8, type="integer", minimum=0),
        "eval_size": Param(16, type="integer", minimum=2),
        "lr": Param(3e-4, type="number", minimum=1e-8),
        "bottleneck_dim": Param(64, type="integer", minimum=1),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "var_min_id": Param(5, type="integer", minimum=1),
        "answer_weight": Param(1.0, type="number", minimum=0.0),
        "margin": Param(0.2, type="number", minimum=0.0),
        "return_norm_weight": Param(0.01, type="number", minimum=0.0),
        "continuation_target_max_tokens": Param(5, type="integer", minimum=1),
        "continuation_eval_tokens": Param(5, type="integer", minimum=1),
        "scratchpad_token": Param("<symbiote>", type="string", minLength=1),
        "scratchpad_length": Param(4, type="integer", minimum=1),
        "scratchpad_alpha": Param(1.0, type="number", minimum=0.0),
        "b_guard_threshold": Param(0.01, type="number", minimum=0.0),
        "c_guard_threshold": Param(0.05, type="number", minimum=0.0),
        "d_guard_threshold": Param(0.10, type="number", minimum=0.0),
        "residual_guard_weight": Param(5.0, type="number", minimum=0.0),
        "seed": Param(42, type="integer", minimum=0),
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "upstream_m3_18_report": Param("", type="string"),
        "upstream_m3_19_report": Param("", type="string"),
        "upstream_m11_manifest": Param("", type="string"),
        "pack_jsonl": Param("", type="string"),
        "strict_balance": Param(True, type="boolean"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m14_symbiote_scratchpad", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m14_symbiote_scratchpad",
        python_callable=_run_m14,
    )
