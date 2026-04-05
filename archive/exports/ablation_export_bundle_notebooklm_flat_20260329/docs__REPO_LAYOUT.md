# Repo Layout

The repository is organized by intent, not by historical accident.

## Top Level

- `src/`: library code and reusable runtime modules
- `scripts/`: runnable entrypoints and experiment utilities
- `tests/`: tests and validation harnesses
- `airflow/`: orchestration DAGs and Airflow helpers
- `docs/`: stable human-facing docs, specs, ledgers, and policy
- `artifacts/`: canonical generated telemetry outputs that are intentionally kept under a stable contract
- `archive/`: preserved historical outputs, datasets, snapshots, and legacy reports
- `configs/`: tracked machine-readable configuration and taxonomy manifests

## What Does Not Belong At Root

Do not create new top-level:

- `RESULTS_*`
- `ROOT_*`
- one-off report markdown/json files
- zipped snapshot drops

Use these locations instead:

- historical outputs: `archive/results/<family>/<date_or_window>/<legacy_name>/`
- historical datasets: `archive/datasets/<family_or_topic>/`
- snapshots: `archive/snapshots/`
- stable docs and reports: `docs/history/`, `docs/specs/`, or `docs/ledger/`

## Active Path Policy

- Active canonical telemetry stays under `artifacts/`
- Preserved historical runtime output goes under `archive/`
- Scratch output should be ignored unless deliberately promoted into `artifacts/` or `archive/`

## Script And DAG Navigation

- Family-specific scripts should prefer `scripts/<family>/` when a family already has a home
- History and ledger utilities should live together logically and reference canonical manifests instead of ad hoc root files
- DAGs should mirror the same family boundaries used by experiment series and lineage docs
