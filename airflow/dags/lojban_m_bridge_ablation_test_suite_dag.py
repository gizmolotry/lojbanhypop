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
    "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite",
    "run_id": "",
    "history_manifest": "",
    "m3_15d_report": "",
    "m3_16_report": "",
    "m3_17_report": "",
    "m3_18_report": "",
    "m3_19_report": "",
    "m14_report": "",
    "m11_manifest": "",
    "m11_bridge_audit": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_bridge_audit.json",
    "m11_floor_lock": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_floor_lock.json",
    "m11_publication_metrics": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_publication_metrics.json",
}


def _run_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--baseline-manifest", baseline_manifest,
        "--output-root", output_dir,
        "--run-id", run_id,
    ]

    for key, flag in (
        ("history_manifest", "--history-manifest"),
        ("m3_15d_report", "--m3-15d-report"),
        ("m3_16_report", "--m3-16-report"),
        ("m3_17_report", "--m3-17-report"),
        ("m3_18_report", "--m3-18-report"),
        ("m3_19_report", "--m3-19-report"),
        ("m14_report", "--m14-report"),
        ("m11_manifest", "--m11-manifest"),
        ("m11_bridge_audit", "--m11-bridge-audit"),
        ("m11_floor_lock", "--m11-floor-lock"),
        ("m11_publication_metrics", "--m11-publication-metrics"),
    ):
        value = str(cfg.get(key, "")).strip()
        if value:
            args.extend([flag, value])

    run_repo_script("scripts/run_m_bridge_ablation_test_suite.py", args)


with DAG(
    dag_id="lojban_m_bridge_ablation_test_suite",
    description="Unified bridge-series test suite that normalizes classic M3 bridge tracks, re-entry tracks, M14 scratchpad runs, and recent M11 discriminative results into one ledgerized diagnosis artifact.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "bridge", "ablation", "suite", "ledger"],
    params={
        "baseline_manifest": Param("docs/baselines/m_series_bridge_baseline_manifest.json", type="string", minLength=1),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "history_manifest": Param("", type="string"),
        "m3_15d_report": Param("", type="string"),
        "m3_16_report": Param("", type="string"),
        "m3_17_report": Param("", type="string"),
        "m3_18_report": Param("", type="string"),
        "m3_19_report": Param("", type="string"),
        "m14_report": Param("", type="string"),
        "m11_manifest": Param("", type="string"),
        "m11_bridge_audit": Param("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_bridge_audit.json", type="string", minLength=1),
        "m11_floor_lock": Param("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_floor_lock.json", type="string", minLength=1),
        "m11_publication_metrics": Param("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_publication_metrics.json", type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_m_bridge_ablation_test_suite",
        python_callable=_run_suite,
    )
