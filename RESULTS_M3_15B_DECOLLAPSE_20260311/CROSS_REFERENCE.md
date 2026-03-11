# M3.15b De-collapse Bundle

## Included Runs

### 1. M3.15b Probe v2
- Source: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/refactor_clean_small/m3_15b_20260311_probe_v2`
- Key result: direct collapse confirmed.
- Metrics:
  - overall_accuracy: `0.375`
  - mean_active_op_count: `1.0`
  - mean_operator_entropy: `0.0`
  - mean_top1_op_share: `1.0`

### 2. M3.8 De-collapse Branch
- Source: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_8_diversification/decollapse_small/m3_8_diversification_20260311_103934`
- Purpose: upstream operator diversification pressure.
- Best cell for de-collapse: `M3.8.B`
- Metrics:
  - operator_entropy: `1.6093924045562744`
  - operator_top1_share: `0.2027810513973236`
  - final_constraint_scope: `0.0`
  - final_constraint_identity: `0.0`
  - checkpoint: `runs/l_series/m3_8_diversification/decollapse_small/20260311_103934/hard/m3_8_b/20260311_104736/l_series_checkpoint.pt`

### 3. M3.15b After De-collapse
- Source: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/decollapse_followup/m3_15b_20260311_after_decollapse`
- Purpose: re-evaluate relation-local rotary bridge using the de-collapsed checkpoint.
- Metrics:
  - Cell A overall_accuracy: `0.5`
  - Cell A mean_active_op_count: `3.0`
  - Cell A mean_operator_entropy: `0.93130153729747`
  - Cell A mean_top1_op_share: `0.4554036458333333`
  - Cell B overall_accuracy: `0.5`
  - Cell B mean_active_op_count: `1.0`
  - Cell B mean_operator_entropy: `0.0`
  - Cell C overall_accuracy: `0.5`
  - Cell C mean_active_op_count: `1.5`
  - Cell C mean_operator_entropy: `0.3461394368281594`

## Interpretation
- The original M3.15b checkpoint was fully collapsed at the operator level.
- Upstream diversification pressure can de-collapse the underlying predicate inventory without regressing arity/scope/identity.
- That de-collapsed inventory carries into the follow-up M3.15b control evaluation: Cell A now uses multiple operators and reaches `0.5` accuracy.
- The bridge-aligned cells still tend to re-collapse operator usage, so the remaining failure is no longer “no operator diversity exists at all.” It is now “the bridge/training path still collapses diversity when optimized.”
