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
    "train_steps": 60,
    "eval_size": 64,
    "lr": 3e-4,
    "max_logic_new_tokens": 48,
    "layer_index": 12,
    "relation_vocab": 5,
    "var_min_id": 5,
    "max_nodes": 12,
    "answer_weight": 1.0,
    "margin": 0.2,
    "anti_collapse_weight": 1.0,
    "collapse_entropy_floor_ratio": 0.85,
    "collapse_top1_margin": 0.10,
    "collapse_top1_weight": 1.0,
    "collapse_kl_weight": 0.5,
    "bridge_train_gate_cap": 0.08,
    "runtime_gate_cap": 0.03,
    "runtime_cue_norm_cap": 1.0,
    "runtime_enable_min_acc_gain": 0.02,
    "seed": 42,
    "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
    "pack_jsonl": "",
    "strict_balance": True,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing",
    "local_files_only": False,
    "run_id": "",
}


def _run_m3_15d(**context: object) -> None:
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

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--checkpoint",
        checkpoint,
        "--train-steps",
        str(int(cfg.get("train_steps", 60))),
        "--eval-size",
        str(int(cfg.get("eval_size", 64))),
        "--lr",
        str(float(cfg.get("lr", 3e-4))),
        "--max-logic-new-tokens",
        str(int(cfg.get("max_logic_new_tokens", 48))),
        "--layer-index",
        str(int(cfg.get("layer_index", 12))),
        "--relation-vocab",
        str(int(cfg.get("relation_vocab", 5))),
        "--var-min-id",
        str(int(cfg.get("var_min_id", 5))),
        "--max-nodes",
        str(int(cfg.get("max_nodes", 12))),
        "--answer-weight",
        str(float(cfg.get("answer_weight", 1.0))),
        "--margin",
        str(float(cfg.get("margin", 0.2))),
        "--anti-collapse-weight",
        str(float(cfg.get("anti_collapse_weight", 1.0))),
        "--collapse-entropy-floor-ratio",
        str(float(cfg.get("collapse_entropy_floor_ratio", 0.85))),
        "--collapse-top1-margin",
        str(float(cfg.get("collapse_top1_margin", 0.10))),
        "--collapse-top1-weight",
        str(float(cfg.get("collapse_top1_weight", 1.0))),
        "--collapse-kl-weight",
        str(float(cfg.get("collapse_kl_weight", 0.5))),
        "--bridge-train-gate-cap",
        str(float(cfg.get("bridge_train_gate_cap", 0.08))),
        "--runtime-gate-cap",
        str(float(cfg.get("runtime_gate_cap", 0.03))),
        "--runtime-cue-norm-cap",
        str(float(cfg.get("runtime_cue_norm_cap", 1.0))),
        "--runtime-enable-min-acc-gain",
        str(float(cfg.get("runtime_enable_min_acc_gain", 0.02))),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--baseline-manifest",
        baseline_manifest,
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    if bool(cfg.get("strict_balance", True)):
        args.append("--strict-balance")
    else:
        args.append("--no-strict-balance")
    pack_jsonl = str(cfg.get("pack_jsonl", "")).strip()
    if pack_jsonl:
        args.extend(["--pack-jsonl", pack_jsonl])
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m3_15d_answer_path_forcing.py", args)


with DAG(
    dag_id="lojban_m3_15d_answer_path_forcing",
    description="M3.15d answer-path forcing ablation: direct-drive margin, blindfolded answer path, and candidate pointer head.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-15d", "answer-path", "ablation"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "train_steps": Param(60, type="integer", minimum=0),
        "eval_size": Param(64, type="integer", minimum=2),
        "lr": Param(3e-4, type="number", minimum=1e-8),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "var_min_id": Param(5, type="integer", minimum=1),
        "max_nodes": Param(12, type="integer", minimum=2),
        "answer_weight": Param(1.0, type="number", minimum=0.0),
        "margin": Param(0.2, type="number", minimum=0.0),
        "anti_collapse_weight": Param(1.0, type="number", minimum=0.0),
        "collapse_entropy_floor_ratio": Param(0.85, type="number", minimum=0.0),
        "collapse_top1_margin": Param(0.10, type="number", minimum=0.0),
        "collapse_top1_weight": Param(1.0, type="number", minimum=0.0),
        "collapse_kl_weight": Param(0.5, type="number", minimum=0.0),
        "bridge_train_gate_cap": Param(0.08, type="number", minimum=0.0),
        "runtime_gate_cap": Param(0.03, type="number", minimum=0.0),
        "runtime_cue_norm_cap": Param(1.0, type="number", minimum=0.0),
        "runtime_enable_min_acc_gain": Param(0.02, type="number", minimum=0.0),
        "seed": Param(42, type="integer", minimum=0),
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "pack_jsonl": Param("", type="string"),
        "strict_balance": Param(True, type="boolean"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_15d_answer_path_forcing",
        python_callable=_run_m3_15d,
    )
