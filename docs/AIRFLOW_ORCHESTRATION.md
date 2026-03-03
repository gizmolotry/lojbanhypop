# Airflow Orchestration

Airflow DAGs in this repository are thin wrappers around canonical pipeline scripts only:

- `airflow/dags/lojban_experiment_dag.py` -> `scripts/pipeline_train_grounded_reasoner.py`
- `airflow/dags/lojban_phase_ablation_dag.py` -> `scripts/pipeline_eval_manifold.py`

No training/eval business logic is implemented in DAG code.

## Contract-First Artifact Flow

This orchestration follows an artifact-contract-first model:

1. Train pipeline writes frozen manifold artifacts.
2. Eval pipeline reads model artifacts and writes raw telemetry artifacts.
3. Downstream jobs consume artifacts by partition contract, not by ad-hoc paths.

Canonical S3 partitions:

- `models/frozen_manifolds` (train outputs)
- `telemetry/raw` (eval outputs)

DAG runtime validation enforces that `output_dir` is both:

- valid per shared Airflow safety checks (`LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS`, `LOJBAN_ALLOWED_S3_PREFIXES`)
- targeted to the expected partition for that DAG

## Prerequisites

1. Install optional orchestration dependencies:

```bash
pip install -e ".[orchestration]"
```

2. Initialize Airflow metadata DB (example for local SQLite):

```bash
airflow db init
```

3. Ensure DAG files are visible to Airflow:

- Option A: run Airflow with `D:\lojbanhypop\airflow\dags` as the DAG folder.
- Option B: copy/symlink `airflow/dags/*.py` into your existing `$AIRFLOW_HOME/dags`.

## Environment Variables

- `LOJBAN_ALLOW_ENV_OVERRIDES` (optional, default `0`): set to `1` to allow the two overrides below.
- `LOJBAN_REPO_ROOT` (optional): absolute path to repository root (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `LOJBAN_PYTHON_BIN` (optional): Python executable used for script execution (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `PYTHONPATH` is managed automatically by DAG utility to include `<repo>/src`.
- `LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS` (optional, default `artifacts/runs,runs`): comma-separated allowlist for local `output_dir`.
- `LOJBAN_ALLOWED_S3_PREFIXES` (optional): comma-separated allowlist for S3 prefixes (for example `s3://my-bucket/lojban-prod`).
- `LOJBAN_ALLOW_ABSOLUTE_OUTPUT_DIR` (optional, default `0`): set to `1` to allow absolute local output paths.
- `LOJBAN_ALLOW_ABSOLUTE_INPUT_ARTIFACT` (optional, default `0`): set to `1` to allow absolute local `input_artifact` paths.

## DAGs

### `lojban_train_grounded_reasoner`

- Schedule: `@daily`
- Task: `run_train_grounded_reasoner`
- Script: `scripts/pipeline_train_grounded_reasoner.py`
- Required partition: `models/frozen_manifolds`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/models/frozen_manifolds`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `arity_violation_rate` (float, default `0.05`)
- `surgery_trigger_count` (int, default `1`)
- `ce_loss_final` (float, default `1.0`)
- `cross_attention_gain` (float, default `0.1`)
- `logical_accuracy` (float in `[0,1]`, default `0.6`)
- `variable_token_distribution_json` (str path, default `docs/variable_token_distribution_1995.example.json`)

Wrapper behavior:

- Validates `output_dir` and required partition (`models/frozen_manifolds`).
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Validates `variable_token_distribution_json` as a local `docs/*.json` path.
- Composes script `--output` as `<output_dir>/<run_id>_grounded_reasoner_train.json`.

Trigger example:

```bash
airflow dags trigger lojban_train_grounded_reasoner \
  --conf '{
    "output_dir": "s3://my-bucket/lojban-prod/models/frozen_manifolds/2026-03-03",
    "run_id": "train_2026_03_03",
    "arity_violation_rate": 0.04,
    "surgery_trigger_count": 3,
    "ce_loss_final": 0.92,
    "cross_attention_gain": 0.18,
    "logical_accuracy": 0.81,
    "variable_token_distribution_json": "docs/variable_token_distribution_1995.example.json"
  }'
```

### `lojban_eval_manifold`

- Schedule: manual (`schedule=None`)
- Task: `run_eval_manifold`
- Script: `scripts/pipeline_eval_manifold.py`
- Required partition: `telemetry/raw`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/telemetry/raw`)
- `input_artifact` (str path, required)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `logical_accuracy` (float in `[0,1]`, default `0.6`)
- `ce_loss_final` (float, optional)
- `cross_attention_gain` (float, optional)

Wrapper behavior:

- Validates `output_dir` and required partition (`telemetry/raw`).
- Validates `input_artifact` is under `models/frozen_manifolds`.
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Composes script `--output` as `<output_dir>/<run_id>_manifold_eval.json`.

Trigger example:

```bash
airflow dags trigger lojban_eval_manifold \
  --conf '{
    "output_dir": "s3://my-bucket/lojban-prod/telemetry/raw/2026-03-03",
    "input_artifact": "s3://my-bucket/lojban-prod/models/frozen_manifolds/2026-03-03/train_2026_03_03_grounded_reasoner_train.json",
    "run_id": "eval_2026_03_03",
    "logical_accuracy": 0.79,
    "ce_loss_final": 0.95,
    "cross_attention_gain": 0.16
  }'
```

## Operations Notes

- Keep `max_active_runs=1` per DAG to avoid overlapping heavyweight runs.
- For production workers, keep `LOJBAN_ALLOW_ENV_OVERRIDES=0` unless explicit path overrides are required.
- Keep pipeline compatibility at the script contract boundary (`--output` artifact + `output_dir` partitioning) to simplify upgrades.
