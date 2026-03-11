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
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "dataset_size_per_bucket": 80,
    "buckets": "legacy,easy,medium,hard",
    "seed": 7,
    "max_logic_new_tokens": 48,
    "max_final_new_tokens": 16,
    "layer_index": 12,
    "relation_vocab": 5,
    "var_min_id": 5,
    "primitive_candidates": "",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy",
    "local_files_only": False,
    "run_id": "",
}


def _run_m3_10(**context: object) -> None:
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
    output_root = f"{output_dir.rstrip('/')}/{run_id}"

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--checkpoint",
        checkpoint,
        "--baseline-manifest",
        baseline_manifest,
        "--dataset-size-per-bucket",
        str(int(cfg.get("dataset_size_per_bucket", 80))),
        "--buckets",
        str(cfg.get("buckets", "legacy,easy,medium,hard")),
        "--seed",
        str(int(cfg.get("seed", 7))),
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
        "--output-root",
        output_root,
        "--run-id",
        run_id,
    ]
    primitive_candidates = str(cfg.get("primitive_candidates", "")).strip()
    if primitive_candidates:
        args.extend(["--primitive-candidates", primitive_candidates])
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m3_10_ood_accuracy_probe.py", args)


with DAG(
    dag_id="lojban_m3_10_ood_accuracy_probe",
    description="M3.10 OOD accuracy closure probe (evaluation-only, frozen M_BASE).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-10", "ood", "accuracy", "eval"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "baseline_manifest": Param("docs/baselines/m_series_baseline_manifest.json", type="string", minLength=1),
        "dataset_size_per_bucket": Param(80, type="integer", minimum=1),
        "buckets": Param("legacy,easy,medium,hard", type="string", minLength=1),
        "seed": Param(7, type="integer", minimum=0),
        "max_logic_new_tokens": Param(48, type="integer", minimum=1),
        "max_final_new_tokens": Param(16, type="integer", minimum=1),
        "layer_index": Param(12, type="integer", minimum=0),
        "relation_vocab": Param(5, type="integer", minimum=1),
        "var_min_id": Param(5, type="integer", minimum=1),
        "primitive_candidates": Param("", type="string"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_10_ood_accuracy_probe",
        python_callable=_run_m3_10,
    )
