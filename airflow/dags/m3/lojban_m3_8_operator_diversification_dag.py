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
    validate_output_dir,
    validate_output_partition,
)


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "train_steps": 120,
    "dataset_size": 1000,
    "dataset_profile": "diverse_v2",
    "difficulty_buckets": "legacy,easy,medium,hard",
    "seed": 7,
    "local_files_only": False,
    "l_output_root": "runs/l_series/m3_8_diversification",
    "report_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_8_diversification",
    "arity_enforcement_mode": "crystallization",
    "dynamic_arity_signatures": False,
    "operator_arity_json": "",
    "default_relation_arity": 2,
    "shadow_margin": 0.10,
    "stage0_steps": 30,
    "stage1_steps": 120,
    "diversification_weight": 0.05,
    "diversification_domain_overlap_target": 0.45,
    "diversification_top1_penalty": 0.25,
    "diversification_cluster_centroids": 3,
    "diversification_cluster_margin": 0.80,
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "run_id": "",
}


def _run_m3_8(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")
    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())
    l_output_root = validate_output_dir(str(cfg["l_output_root"]))
    report_output_dir = validate_output_partition(str(cfg["report_output_dir"]), "telemetry/raw")
    if l_output_root.startswith("s3://") or report_output_dir.startswith("s3://"):
        raise ValueError("M3.8 runner writes local artifacts only. Use local output roots and sync to S3 downstream.")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    report_output_root = f"{report_output_dir.rstrip('/')}/{run_id}"

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--train-steps",
        str(int(cfg.get("train_steps", 120))),
        "--dataset-size",
        str(int(cfg.get("dataset_size", 1000))),
        "--dataset-profile",
        str(cfg.get("dataset_profile", "diverse_v2")),
        "--difficulty-buckets",
        str(cfg.get("difficulty_buckets", "legacy,easy,medium,hard")),
        "--seed",
        str(int(cfg.get("seed", 7))),
        "--l-output-root",
        l_output_root,
        "--report-output-root",
        report_output_root,
        "--arity-enforcement-mode",
        str(cfg.get("arity_enforcement_mode", "crystallization")),
        "--default-relation-arity",
        str(int(cfg.get("default_relation_arity", 2))),
        "--shadow-margin",
        str(float(cfg.get("shadow_margin", 0.10))),
        "--stage0-steps",
        str(int(cfg.get("stage0_steps", 30))),
        "--stage1-steps",
        str(int(cfg.get("stage1_steps", 120))),
        "--diversification-weight",
        str(float(cfg.get("diversification_weight", 0.05))),
        "--diversification-domain-overlap-target",
        str(float(cfg.get("diversification_domain_overlap_target", 0.45))),
        "--diversification-top1-penalty",
        str(float(cfg.get("diversification_top1_penalty", 0.25))),
        "--diversification-cluster-centroids",
        str(int(cfg.get("diversification_cluster_centroids", 3))),
        "--diversification-cluster-margin",
        str(float(cfg.get("diversification_cluster_margin", 0.80))),
        "--baseline-manifest",
        baseline_manifest,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")
    if bool(cfg.get("dynamic_arity_signatures", False)):
        args.append("--dynamic-arity-signatures")
    operator_arity_json = str(cfg.get("operator_arity_json", "")).strip()
    if operator_arity_json:
        args.extend(["--operator-arity-json", operator_arity_json])

    run_repo_script("scripts/run_m3_8_operator_diversification.py", args)


with DAG(
    dag_id="lojban_m3_8_operator_diversification",
    description="M3.8 operator diversification pressure family (entropy, domain reuse, family clustering).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-8", "diversification", "ablation"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "train_steps": Param(120, type="integer", minimum=1),
        "dataset_size": Param(1000, type="integer", minimum=1),
        "dataset_profile": Param("diverse_v2", type="string", minLength=1),
        "difficulty_buckets": Param("legacy,easy,medium,hard", type="string", minLength=1),
        "seed": Param(7, type="integer", minimum=0),
        "local_files_only": Param(False, type="boolean"),
        "l_output_root": Param("runs/l_series/m3_8_diversification", type="string", minLength=1),
        "report_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_8_diversification", type="string", minLength=1),
        "arity_enforcement_mode": Param("crystallization", enum=["legacy_strict", "registry_strict", "crystallization"]),
        "dynamic_arity_signatures": Param(False, type="boolean"),
        "operator_arity_json": Param("", type="string"),
        "default_relation_arity": Param(2, type="integer", minimum=1),
        "shadow_margin": Param(0.10, type="number"),
        "stage0_steps": Param(30, type="integer", minimum=0),
        "stage1_steps": Param(120, type="integer", minimum=1),
        "diversification_weight": Param(0.05, type="number"),
        "diversification_domain_overlap_target": Param(0.45, type="number"),
        "diversification_top1_penalty": Param(0.25, type="number"),
        "diversification_cluster_centroids": Param(3, type="integer", minimum=1),
        "diversification_cluster_margin": Param(0.80, type="number"),
        "baseline_manifest": Param("docs/baselines/m_series_baseline_manifest.json", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_8_operator_diversification",
        python_callable=_run_m3_8,
    )
