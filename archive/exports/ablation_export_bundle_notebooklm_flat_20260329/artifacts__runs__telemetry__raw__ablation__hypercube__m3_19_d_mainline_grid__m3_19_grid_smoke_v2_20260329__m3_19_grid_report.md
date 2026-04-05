# M3.19 D-Mainline Ablation Grid

| Cell | Supervision | Guardrail | Acc | FTok | Fluency | Loop | Gold On-Off | Residual Norm | Overflow | Objective |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D0 | legacy_single_token | 0.01 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0000 | 0.0016 | 0.000 | margin |
| D1 | rich_4bucket_1to5 | 0.01 | 0.000 | 0.000 | 1.000 | 0.000 | -0.0000 | 0.0046 | 0.000 | continuation_ce |
| D2 | rich_4bucket_1to5 | 0.05 | 0.000 | 0.000 | 1.000 | 0.000 | -0.0000 | 0.0046 | 0.000 | continuation_ce |
| D3 | rich_4bucket_1to5 | 0.10 | 0.000 | 0.000 | 1.000 | 0.000 | -0.0000 | 0.0046 | 0.000 | continuation_ce |

## Grid
- D0: legacy single-token anchor.
- D1: rich 4-bucket continuation CE under strict threshold 0.01.
- D2: rich 4-bucket continuation CE with threshold 0.05.
- D3: rich 4-bucket continuation CE with threshold 0.10.

Pack: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329/m3_19_resumption_pack.jsonl`