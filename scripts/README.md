# Scripts Surface

This directory is the runnable command surface of the repo.

## Primary Buckets

- `scripts/control_plane/`: lineage, backfill, rendering, and orchestration-facing control-plane builders
- `scripts/data/`: dataset builders, pack generators, and corpus reconstruction helpers
- `scripts/legacy/`: older executable surfaces that remain useful but are not the forward organizational model
- `scripts/training/`: cross-family training entrypoints and local training wrappers
- `scripts/m3/`, `scripts/m4/`, `scripts/m5/`, `scripts/m14/`, `scripts/m18/`, `scripts/m19/`, and peers: family-local runners
- root `scripts/`: family buckets plus a very small number of intentional top-level surfaces only

## Current Navigation Rule

When you are looking for an active family:

1. check `scripts/<family>/` first
2. check `scripts/control_plane/` for history, ledger, and report refresh work
3. check `scripts/data/` when the task is dataset or export preparation
4. check `scripts/training/` when the task is cross-family training rather than a family-local runner
5. check `scripts/legacy/` only when you are explicitly dealing with older experimental surfaces

## High-Signal Entrypoints

- `scripts/control_plane/run_ablation_history_backfill.py`: canonical historical backfill and normalization
- `scripts/control_plane/build_ablation_program_map.py`: family grouping and alias concentration
- `scripts/control_plane/build_ablation_program_spine.py`: ordered research spine across legacy and normalized families
- `scripts/control_plane/render_ablation_history_catalog.py`: human-readable ledger rendering
- `scripts/data/build_m3_19_resumption_pack.py`: continuation-resumption supervision pack builder
- `scripts/training/train_lora.py`: shared LoRA trainer used by several historical and restoration flows
- `scripts/m19/run_m19_mainline_suite.py`: M19 mainline wrapper
- `scripts/m19/run_m19_isolation_grid.py`: M19 isolation sweep wrapper
- `scripts/m14/run_m14_symbiote_scratchpad.py`: M14 scratchpad family runner

## Cleanup Status

This surface is now intentionally bucketed. New runnable entrypoints should land in a family directory, `control_plane/`, `data/`, `training/`, or `legacy/` rather than re-flattening the root.
