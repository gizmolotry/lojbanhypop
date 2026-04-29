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

Local-only runtime state may still appear under an ignored `config/` directory on individual machines.
That path is not part of the canonical repo layout and should never be used for tracked experiment metadata.
The top-level `runs/` directory is preserved as a legacy runtime contract because a large amount of historical code and documentation still references it explicitly.
New canonical tracked outputs should still go to `artifacts/` or `archive/`, not to `runs/`.

## What Does Not Belong At Root

Do not create new top-level:

- `RESULTS_*`
- `ROOT_*`
- one-off report markdown/json files
- zipped snapshot drops

Use these locations instead:

- historical outputs: `archive/results/<family>/<date_or_window>/<legacy_name>/`
- historical datasets: `archive/datasets/<family_or_topic>/`
- text export bundles and notebook-ingest drops: `archive/exports/<target>/`
- snapshots: `archive/snapshots/`
- stable docs and reports: `docs/history/`, `docs/specs/`, or `docs/ledger/`

## Active Path Policy

- Active canonical telemetry stays under `artifacts/`
- Preserved historical runtime output goes under `archive/`
- Flat text export mirrors belong under `archive/exports/`, not at repo root
- `runs/` exists only as a legacy execution/output contract for older scripts and reproductions
- Scratch output should be ignored unless deliberately promoted into `artifacts/` or `archive/`

## Script And DAG Navigation

- Family-specific scripts should live under `scripts/<family>/`
- Control-plane builders and history refresh tools should live under `scripts/control_plane/`
- Legacy one-off executables should live under `scripts/legacy/`
- Control-plane utilities should stay grouped and reference canonical manifests instead of ad hoc root files
- DAGs should mirror the same family boundaries used by experiment series and lineage docs
- Root `scripts/` should contain only shared helper/build/train surfaces that are still intentionally cross-family
