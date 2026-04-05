from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_input_artifact,
    validate_output_partition,
)


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw",
    "run_id": "",
    "h5_ood_artifact": "",
    "j2_variants_per_graph": 1000,
    "j4_per_operator": 256,
    "j5_sample_count": 256,
    "j5_novelty_threshold": 0.30,
}

J3_SOURCE_SCRIPT = "scripts/train_h5_persistent_vq_advisor.py"


def _repo_root() -> Path:
    allow_overrides = os.environ.get("LOJBAN_ALLOW_ENV_OVERRIDES", "0") == "1"
    override = os.environ.get("LOJBAN_REPO_ROOT")
    if allow_overrides and override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def _compose_output(output_dir: str, run_id: str, file_name: str) -> str:
    if output_dir.startswith("s3://"):
        return f"{output_dir.rstrip('/')}/{run_id}_{file_name}"
    return str(Path(output_dir) / f"{run_id}_{file_name}")


def _normalize_path(path_value: str) -> str:
    return str(path_value).replace("\\", "/").strip()


def _require_local_file(path_value: str, label: str) -> Path:
    normalized = _normalize_path(path_value)
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    if normalized.startswith("s3://"):
        raise ValueError(f"{label} must be a local file path, not S3: {path_value}")
    path = Path(path_value)
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_json_artifact(path_value: str, stage_label: str, expected_run_id: str) -> dict[str, Any]:
    artifact_path = _require_local_file(path_value, f"{stage_label} artifact")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{stage_label} artifact is not valid JSON: {artifact_path}") from exc

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{stage_label} artifact must include a summary object: {artifact_path}")
    actual_run_id = str(summary.get("run_id", "")).strip()
    if actual_run_id != expected_run_id:
        raise ValueError(
            f"{stage_label} artifact summary.run_id must be '{expected_run_id}', got '{actual_run_id or '<missing>'}'"
        )
    return payload


def _require_metric_keys(payload: dict[str, Any], stage_label: str, required_keys: tuple[str, ...]) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{stage_label} artifact must include metrics")
    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise ValueError(f"{stage_label} artifact metrics missing keys: {missing}")


def _validate_dataset_sidecar(payload: dict[str, Any], stage_label: str, expected_dataset_path: str) -> None:
    summary = payload["summary"]
    actual_dataset_path = _normalize_path(str(summary.get("dataset_output", "")))
    expected = _normalize_path(expected_dataset_path)
    if actual_dataset_path != expected:
        raise ValueError(
            f"{stage_label} artifact summary.dataset_output must be '{expected_dataset_path}', got '{actual_dataset_path}'"
        )
    _require_local_file(expected_dataset_path, f"{stage_label} dataset_output")


def _resolve_j_series_contract(context: dict[str, object]) -> dict[str, object]:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    if output_dir.startswith("s3://"):
        raise ValueError(
            "lojban_j_series_invariance currently writes local artifacts only. "
            "Use a local telemetry/raw path and sync to S3 downstream."
        )

    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    j2_variants_per_graph = int(cfg["j2_variants_per_graph"])
    j4_per_operator = int(cfg["j4_per_operator"])
    j5_sample_count = int(cfg["j5_sample_count"])
    j5_novelty_threshold = float(cfg["j5_novelty_threshold"])

    if j2_variants_per_graph < 1:
        raise ValueError("j2_variants_per_graph must be >= 1")
    if j4_per_operator < 1:
        raise ValueError("j4_per_operator must be >= 1")
    if j5_sample_count < 1:
        raise ValueError("j5_sample_count must be >= 1")
    if not 0.0 <= j5_novelty_threshold <= 1.0:
        raise ValueError("j5_novelty_threshold must be within [0, 1]")

    h5_ood_artifact = str(cfg.get("h5_ood_artifact") or "").strip()
    if h5_ood_artifact:
        h5_ood_artifact = validate_input_artifact(h5_ood_artifact, "telemetry/raw")
        _require_local_file(h5_ood_artifact, "h5_ood_artifact")

    return {
        "output_dir": output_dir,
        "run_id": run_id,
        "h5_ood_artifact": h5_ood_artifact,
        "j2_variants_per_graph": j2_variants_per_graph,
        "j4_per_operator": j4_per_operator,
        "j5_sample_count": j5_sample_count,
        "j5_novelty_threshold": j5_novelty_threshold,
        "j1_artifact": _compose_output(output_dir, run_id, "j1_graph_target.json"),
        "j2_artifact": _compose_output(output_dir, run_id, "j2_paraphrase_explosion.json"),
        "j3_artifact": _compose_output(output_dir, run_id, "j3_stopgrad_gate.json"),
        "j4_artifact": _compose_output(output_dir, run_id, "j4_operator_curriculum.json"),
        "j4_dataset_output": _compose_output(output_dir, run_id, "j4_operator_curriculum.jsonl"),
        "j5_artifact": _compose_output(output_dir, run_id, "j5_adversarial_synthesis.json"),
        "j5_dataset_output": _compose_output(output_dir, run_id, "j5_adversarial_synthesis.jsonl"),
    }


def _validate_j_series_contract(**context: object) -> None:
    _resolve_j_series_contract(context)
    _require_local_file(J3_SOURCE_SCRIPT, "J-3 source_script")


def _run_j1_graph_target(**context: object) -> None:
    contract = _resolve_j_series_contract(context)
    j1_args = ["--output", str(contract["j1_artifact"])]
    h5_ood_artifact = str(contract["h5_ood_artifact"])
    if h5_ood_artifact:
        j1_args.extend(["--input-artifact", h5_ood_artifact])
    run_repo_script("scripts/eval_j_1.py", j1_args)

    payload = _load_json_artifact(str(contract["j1_artifact"]), "J-1", "J-1")
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("J-1 artifact must include a non-empty graphs list")
    _require_metric_keys(payload, "J-1", ("graph_count", "schema_valid_rate"))


def _run_j2_paraphrase_explosion(**context: object) -> None:
    contract = _resolve_j_series_contract(context)
    upstream = _load_json_artifact(str(contract["j1_artifact"]), "J-1", "J-1")
    graphs = upstream.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("J-2 requires a non-empty J-1 graphs artifact")

    run_repo_script(
        "scripts/eval_j_2.py",
        [
            "--j1-artifact",
            str(contract["j1_artifact"]),
            "--variants-per-graph",
            str(contract["j2_variants_per_graph"]),
            "--output",
            str(contract["j2_artifact"]),
        ],
    )

    payload = _load_json_artifact(str(contract["j2_artifact"]), "J-2", "J-2")
    _require_metric_keys(payload, "J-2", ("variant_count", "invariance_rate"))


def _run_j3_stopgrad_gate(**context: object) -> None:
    contract = _resolve_j_series_contract(context)
    _require_local_file(J3_SOURCE_SCRIPT, "J-3 source_script")

    run_repo_script(
        "scripts/eval_j_3.py",
        [
            "--source-script",
            J3_SOURCE_SCRIPT,
            "--output",
            str(contract["j3_artifact"]),
        ],
    )

    payload = _load_json_artifact(str(contract["j3_artifact"]), "J-3", "J-3")
    _require_metric_keys(payload, "J-3", ("stopgrad_contract_pass",))


def _run_j4_operator_curriculum(**context: object) -> None:
    contract = _resolve_j_series_contract(context)

    run_repo_script(
        "scripts/eval_j_4.py",
        [
            "--per-operator",
            str(contract["j4_per_operator"]),
            "--output",
            str(contract["j4_artifact"]),
            "--dataset-output",
            str(contract["j4_dataset_output"]),
        ],
    )

    payload = _load_json_artifact(str(contract["j4_artifact"]), "J-4", "J-4")
    _require_metric_keys(payload, "J-4", ("sample_count", "operator_count"))
    _validate_dataset_sidecar(payload, "J-4", str(contract["j4_dataset_output"]))


def _run_j5_adversarial_synthesis(**context: object) -> None:
    contract = _resolve_j_series_contract(context)

    run_repo_script(
        "scripts/eval_j_5.py",
        [
            "--sample-count",
            str(contract["j5_sample_count"]),
            "--novelty-threshold",
            str(contract["j5_novelty_threshold"]),
            "--output",
            str(contract["j5_artifact"]),
            "--dataset-output",
            str(contract["j5_dataset_output"]),
        ],
    )

    payload = _load_json_artifact(str(contract["j5_artifact"]), "J-5", "J-5")
    _require_metric_keys(payload, "J-5", ("accepted_count", "generator_accept_rate", "accepted_foil_pair_accuracy", "accept_rate_by_depth"))
    _validate_dataset_sidecar(payload, "J-5", str(contract["j5_dataset_output"]))


with DAG(
    dag_id="lojban_j_series_invariance",
    description="Phase 7 J-series invariance DAG with explicit J1-J5 task boundaries and artifact checks.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "phase7", "j-series", "invariance"],
    params={
        "output_dir": Param("artifacts/runs/telemetry/raw", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "h5_ood_artifact": Param("", type="string"),
        "j2_variants_per_graph": Param(1000, type="integer", minimum=1),
        "j4_per_operator": Param(256, type="integer", minimum=1),
        "j5_sample_count": Param(256, type="integer", minimum=1),
        "j5_novelty_threshold": Param(0.30, type="number", minimum=0, maximum=1),
    },
) as dag:
    validate_j_series_contract = PythonOperator(
        task_id="validate_j_series_contract",
        python_callable=_validate_j_series_contract,
    )
    run_j1_graph_target = PythonOperator(
        task_id="run_j1_graph_target",
        python_callable=_run_j1_graph_target,
    )
    run_j2_paraphrase_explosion = PythonOperator(
        task_id="run_j2_paraphrase_explosion",
        python_callable=_run_j2_paraphrase_explosion,
    )
    run_j3_stopgrad_gate = PythonOperator(
        task_id="run_j3_stopgrad_gate",
        python_callable=_run_j3_stopgrad_gate,
    )
    run_j4_operator_curriculum = PythonOperator(
        task_id="run_j4_operator_curriculum",
        python_callable=_run_j4_operator_curriculum,
    )
    run_j5_adversarial_synthesis = PythonOperator(
        task_id="run_j5_adversarial_synthesis",
        python_callable=_run_j5_adversarial_synthesis,
    )

    validate_j_series_contract >> run_j1_graph_target >> run_j2_paraphrase_explosion
    validate_j_series_contract >> run_j3_stopgrad_gate
    validate_j_series_contract >> run_j4_operator_curriculum
    validate_j_series_contract >> run_j5_adversarial_synthesis
