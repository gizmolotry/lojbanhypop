# M-Series Train vs Eval Lineage

Last updated: 2026-03-07

## Purpose

This document explicitly separates **training ablations** from **evaluation-only probes** and records what checkpoint each probe used.

## Canonical Rule

- Training runs create new checkpoints.
- Evaluation runs must declare a frozen checkpoint input and must not update weights.
- Any report must include:
  - `checkpoint` path
  - dataset profile/tier used
  - mode: `train` or `eval_only`

## Run Lineage (Current)

1. `M3.8` (train)
- Artifact: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_8_diversification/m3_8_diversification_20260307_013004/m3_8_diversification_report.json`
- Mode: **training ablation family** (`M3.8.A/B/C`) across buckets.
- Buckets trained:
  - `legacy` with profile `legacy`
  - `easy` with profile `diverse_v2`
  - `medium` with profile `diverse_v2`
  - `hard` with profile `diverse_v2`
- Important: this means easy/medium/hard were used in training in M3.8 split.

2. `M3.9` (eval_only)
- Artifact: `m3_9_primitive_probe*`
- Mode: **evaluation-only** primitive probe.
- Frozen checkpoint used in latest split runs:
  - `runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_checkpoint.pt`

3. `M3.10` (eval_only)
- Artifact: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy/m3_10_20260307_r2/m3_10_ood_accuracy_report.json`
- Mode: **evaluation-only** OOD closure probe.
- Frozen checkpoint: same `M3.8.C hard` checkpoint above.
- Buckets evaluated: `legacy,easy,medium,hard`.

4. `M3.11` (eval_only)
- Artifact: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/m3_11_20260307/m3_11_winograd_failure_anatomy_report.json`
- Mode: **evaluation-only** Winograd failure anatomy.
- Frozen checkpoint: same `M3.8.C hard` checkpoint above.
- Dataset profiles used:
  - `legacy` bucket from `legacy` profile (winograd slice)
  - `easy/medium/hard` from `winograd_bench_v1` profile

## Direct Answers to Current Questions

1. Have we always carried training down the waterfall cleanly?
- Not cleanly enough in documentation. The lineage existed in artifacts, but it was not explicitly tracked in one place.

2. Since easy/medium/hard questions were introduced, has the model been trained on them?
- **Yes** for `diverse_v2` easy/medium/hard in `M3.8` training split on 2026-03-07.
- **No** for newly added `winograd_bench_v1` / `diverse_v3` expansions so far; those were used in `M3.11` evaluation-only.

3. Have we trained only on legacy problems recently?
- **No**. The checkpoint used by M3.10/M3.11 (`.../hard/m3_8_c/...`) was trained on the `hard` bucket from `diverse_v2`, not legacy-only.

## Immediate Process Upgrade

For every new M-run, write one line in this ledger before execution:

- `run_id`, `mode(train|eval_only)`, `checkpoint_in`, `checkpoint_out(if train)`, `dataset_profile`, `difficulty_tier`.

## 2026-03-10 Refactor Note

- Historical `M3.8` runs before the 2026-03-10 substrate refactor are provisional for scientific comparison because the trainer optimized on the dataset `test` split.
- Historical `M3.10` reports before the refactor are mixed-distribution probes unless they explicitly separate trained buckets from OOD buckets.
- Historical `M3.15` reports before the refactor are provisional for bridge generalization because train/val/eval were drawn from the same template family bank.
- Post-refactor policy:
  - training must optimize on `train` only
  - runtime policy selection must use validation only
  - final metrics must use held-out evaluation only
  - every report must emit explicit lineage metadata and declare whether each bucket is `in_distribution` or `ood`
