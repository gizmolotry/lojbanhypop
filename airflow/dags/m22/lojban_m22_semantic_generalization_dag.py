from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "suite_report": "artifacts/runs/telemetry/raw/ablation/hypercube/m21_dynamic_bridi_suite/m21_role_curriculum_mno_48e_20260518/m21_dynamic_bridi_suite_report.json",
    "adversarial_audit_report": "",
    "m21_control_direct_manifest": "artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m21_semantic_hijkl_isolation_direct_fixed_20260518/direct_unified_eval_manifest.json",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m22_semantic_generalization",
    "run_id": "",
    "min_semantic_delta": 0.02,
    "max_clean_drop": 0.02,
    "min_judri_delta": 0.70,
}


def _run_m22_semantic_generalization(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--suite-report",
        str(cfg.get("suite_report")),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
        "--min-semantic-delta",
        str(float(cfg.get("min_semantic_delta", 0.02))),
        "--max-clean-drop",
        str(float(cfg.get("max_clean_drop", 0.02))),
        "--min-judri-delta",
        str(float(cfg.get("min_judri_delta", 0.70))),
    ]
    if str(cfg.get("adversarial_audit_report", "")).strip():
        args.extend(["--adversarial-audit-report", str(cfg.get("adversarial_audit_report"))])
    if str(cfg.get("m21_control_direct_manifest", "")).strip():
        args.extend(["--m21-control-direct-manifest", str(cfg.get("m21_control_direct_manifest"))])
    run_repo_script("scripts/m22/run_m22_semantic_generalization.py", args)


with DAG(
    dag_id="lojban_m22_semantic_generalization",
    description="M22 semantic coverage generalization gate over fixed M21 dynamic bridi controls.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m22", "semantic-generalization"],
    params={
        "suite_report": Param(DEFAULTS["suite_report"], type="string", minLength=1),
        "adversarial_audit_report": Param("", type="string"),
        "m21_control_direct_manifest": Param(DEFAULTS["m21_control_direct_manifest"], type="string"),
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "min_semantic_delta": Param(0.02, type="number", minimum=0.0),
        "max_clean_drop": Param(0.02, type="number", minimum=0.0),
        "min_judri_delta": Param(0.70, type="number", minimum=0.0),
    },
) as dag:
    PythonOperator(
        task_id="run_m22_semantic_generalization",
        python_callable=_run_m22_semantic_generalization,
    )
