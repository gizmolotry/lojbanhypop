# DAG Surface

This directory contains thin Airflow wrappers around canonical scripts.

## Navigation

- `control_plane/`: master spine, backfill, reporting, and orchestration DAGs
- `legacy/`: older letter-series DAGs preserved for lineage and rerun access
- `m3/`, `m4/`, `m5/`, `m11/`, `m14/`, `m18/`, `m19/`, `m_bridge/`: normalized family DAG buckets
- root `lojban_airflow_utils.py`: shared DAG helper surface

## Design Rule

DAG code should stay orchestration-only:

- validate runtime config
- call canonical scripts
- write outputs under the declared contract roots

Business logic belongs in `scripts/` and `src/`, not in DAG tasks.

## Cleanup Status

The DAG surface is now physically grouped by role and family instead of living as one flat directory.
Future cleanup should focus on consistency inside those buckets, not on re-flattening the DAG tree.
