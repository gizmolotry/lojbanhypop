# Airflow Orchestration

This repository now includes lightweight Airflow DAG wrappers around existing experiment scripts:

- `airflow/dags/lojban_experiment_dag.py` -> `scripts/run_experiment.py`
- `airflow/dags/lojban_phase_ablation_dag.py` -> `scripts/run_phase_ablation.py`

The DAGs call existing script entrypoints only; no experiment logic is duplicated in Airflow.

## Prerequisites

1. Install optional orchestration dependencies:

```bash
pip install -e ".[orchestration]"
```

2. Initialize Airflow metadata DB (example for local SQLite):

```bash
airflow db init
```

3. Ensure the DAG files are visible to Airflow:
- Option A: run Airflow with `D:\lojbanhypop\airflow\dags` as the DAG folder.
- Option B: copy/symlink `airflow/dags/*.py` into your existing `$AIRFLOW_HOME/dags`.

## Environment Variables

- `LOJBAN_ALLOW_ENV_OVERRIDES` (optional, default `0`): set to `1` to allow the two overrides below.
- `LOJBAN_REPO_ROOT` (optional): absolute path to repository root (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `LOJBAN_PYTHON_BIN` (optional): Python executable used for script execution (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `PYTHONPATH` is managed automatically by the DAG utility to include `<repo>/src`.
- `LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS` (optional, default `artifacts/runs,runs`): comma-separated allowlist for local `output_dir`.
- `LOJBAN_ALLOWED_S3_PREFIXES` (optional): comma-separated allowlist for S3 output prefixes (for example `s3://my-bucket/lojban-prod`).
- `LOJBAN_ALLOW_ABSOLUTE_OUTPUT_DIR` (optional, default `0`): set to `1` to allow absolute local output paths.

## DAGs

### `lojban_identity_experiment`

- Schedule: `@daily`
- Task: `run_identity_experiment`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `iterations` (int, default `6`)
- `seed` (int, default `7`)
- `dataset_size` (int, default `1000`)
- `max_accept` (int, default `3`)
- `output_dir` (str, default `artifacts/runs`)

Trigger example:

```bash
airflow dags trigger lojban_identity_experiment \
  --conf '{"iterations": 4, "dataset_size": 800, "output_dir": "artifacts/runs"}'
```

### `lojban_phase_ablation`

- Schedule: manual (`schedule=None`)
- Task: `run_phase_ablation`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `dataset_size` (int, default `1000`)
- `seed` (int, default `7`)
- `iterations` (int, default `6`)
- `max_accept` (int, default `3`)
- `top_k` (int, default `10`)
- `min_support` (int, default `30`)
- `output_dir` (str, default `artifacts/runs`)

Trigger example:

```bash
airflow dags trigger lojban_phase_ablation \
  --conf '{"dataset_size": 1200, "iterations": 8, "top_k": 12}'
```

## Outputs

Outputs are unchanged because the DAGs are wrappers:

- `scripts/run_experiment.py` writes run directories under `artifacts/runs/` with `history.json`, `summary.md`, `run_manifest.json`.
- `scripts/run_phase_ablation.py` writes quarantine + ablation artifacts under `artifacts/runs/` including `ablation.json`, `summary.md`, and `run_manifest.json`.

## Operations Notes

- Keep `max_active_runs=1` per DAG to avoid overlapping heavyweight runs.
- For production workers, keep `LOJBAN_ALLOW_ENV_OVERRIDES=0` unless you need explicit path overrides.
- Use standard Airflow retries/alerts at the task level if your environment requires stricter failure handling.
