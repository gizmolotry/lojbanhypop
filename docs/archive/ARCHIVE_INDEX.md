# Archive Index

The archive preserves historical experiment material without leaving the repo root unreadable.

## Layout

- `archive/results/<family>/<date_or_window>/<legacy_name>/`
- `archive/reports/<family_or_topic>/`
- `archive/datasets/<family_or_topic>/`
- `archive/exports/<target>/`
- `archive/snapshots/`
- `archive/lineage/`

## Family Buckets

- `archive/results/m3/`: early bridge, rotary, and return-stream series
- `archive/results/m5/`: autoregressive, masked-pair, and padded-n-ary chain families
- `archive/results/m6/` through `archive/results/m10/`: later staged engine, council, manifold, and translation outputs
- `archive/results/legacy_misc/`: historical roots that do not map cleanly to a numbered family

## Legacy Name Preservation

The final path keeps the original legacy directory name as the leaf directory so old names remain browsable and searchable.

Examples:

- root legacy output `RESULTS_M9_PHASE3` -> `archive/results/m9/legacy_root/RESULTS_M9_PHASE3`
- root export mirror `scripts_txt` -> `archive/exports/notebooklm/scripts_txt`
- `RESULTS_M3_15D_ANSWER_PATH_20260311` -> `archive/results/m3/20260311/RESULTS_M3_15D_ANSWER_PATH_20260311`
- `ROOT_ARITY_DATA_20260312` -> `archive/datasets/root/20260312/ROOT_ARITY_DATA_20260312`

## Current Root Decompression

Recent cleanup passes moved two long-lived root clutter surfaces into the archive:

- `RESULTS_M9_PHASE3` was archived under `archive/results/m9/legacy_root/`
- the flat text export bundle previously living at `scripts_txt/` was archived under `archive/exports/notebooklm/`

This keeps the root focused on active source, docs, orchestration, and canonical artifacts.

## Source Of Truth

Historical meaning should be read from the canonical ablation history manifest and the experiment taxonomy docs, not inferred from raw folder names alone.
