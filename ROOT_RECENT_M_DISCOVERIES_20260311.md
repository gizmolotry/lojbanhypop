# Recent M-Series Discoveries

Date: 2026-03-11

This report consolidates the recent late-M-series bridge and operator-diversification work that was run without a single root summary. It is limited to the following runs and bundles:

- `M3.8` de-collapse branch
- `M3.15b` relation-local rotary bridge
- `M3.15b` de-collapse follow-up
- `M3.15b` guarded bridge-base rerun
- `M3.15c` family-conditioned bridge

## Executive Summary

The main recent discovery is not that the bridge became useful. It did not. The main discovery is that the original `M3.15b` failure was confounded by hard operator collapse, and that this collapse can be repaired upstream without regressing structural constraints.

The sequence of results is:

1. `M3.15b` probe confirmed hard operator collapse.
2. `M3.8` upstream diversification produced a clean de-collapsed checkpoint.
3. Reusing that checkpoint improved the control representation immediately.
4. Guarding the bridge stopped bridge training from re-collapsing the operator inventory.
5. Even with collapse fixed, bridge variants still did not improve final answer accuracy.
6. Adding family-conditioned supervision in `M3.15c` also failed to improve usefulness and partially reintroduced collapse in the more aggressive cell.

The architecture state is now clearer:

- predicate inventory preservation is solvable
- bridge-induced re-collapse is solvable
- bridge usefulness is still unsolved

## Covered Artifacts

Primary bundles in root:

- [RESULTS_M3_15B_DECOLLAPSE_20260311](D:\lojbanhypop\RESULTS_M3_15B_DECOLLAPSE_20260311)
- [RESULTS_M3_15B_BRIDGE_BASE_20260311](D:\lojbanhypop\RESULTS_M3_15B_BRIDGE_BASE_20260311)
- [RESULTS_M3_15C_FAMILY_BRIDGE_20260311](D:\lojbanhypop\RESULTS_M3_15C_FAMILY_BRIDGE_20260311)

Primary reports:

- [m3_15b_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15b_relation_local_rotary\refactor_clean_small\m3_15b_20260311_probe_v2\m3_15b_report.json)
- [m3_8_diversification_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_8_diversification\decollapse_small\m3_8_diversification_20260311_103934\m3_8_diversification_report.json)
- [m3_15b_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15b_relation_local_rotary\bridge_base_v2\m3_15b_20260311_bridge_base_v2\m3_15b_report.json)
- [m3_15c_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15c_family_conditioned_bridge\bridge_base_v1\m3_15c_20260311_bridge_base_v1\m3_15c_report.json)

Baseline manifest introduced for the guarded bridge stack:

- [m_series_bridge_baseline_manifest.json](D:\lojbanhypop\docs\baselines\m_series_bridge_baseline_manifest.json)

## Run-by-Run Summary

### 1. M3.15b Probe: Collapse Confirmation

Report:

- [m3_15b_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15b_relation_local_rotary\refactor_clean_small\m3_15b_20260311_probe_v2\m3_15b_report.json)

Key metrics:

- `overall_accuracy = 0.375`
- `adjective_accuracy = 0.25`
- `causal_accuracy = 0.5`
- `mean_active_op_count = 1.0`
- `mean_operator_entropy = 0.0`
- `mean_top1_op_share = 1.0`

Conclusion:

This directly confirmed hard operator collapse. The poor bridge result was not just weak alignment. The operator inventory had effectively collapsed to one active operator.

### 2. M3.8 De-Collapse Branch: Upstream Repair

Report:

- [m3_8_diversification_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_8_diversification\decollapse_small\m3_8_diversification_20260311_103934\m3_8_diversification_report.json)

Best cell:

- `M3.8.B`

Checkpoint:

- [l_series_checkpoint.pt](D:\lojbanhypop\runs\l_series\m3_8_diversification\decollapse_small\20260311_103934\hard\m3_8_b\20260311_104736\l_series_checkpoint.pt)

Key metrics:

- `operator_entropy = 1.6094`
- `operator_top1_share = 0.2028`
- `final_constraint_scope = 0.0`
- `final_constraint_identity = 0.0`

Conclusion:

Operator diversification pressure fixed the collapse upstream without regressing the key Tier A structural constraints. This was the first clean proof that the predicate inventory can be spread out without breaking scope or identity.

### 3. M3.15b Follow-Up on De-Collapsed Checkpoint

Bundle:

- [m3_15b_after_decollapse](D:\lojbanhypop\RESULTS_M3_15B_DECOLLAPSE_20260311\m3_15b_after_decollapse)

Key metrics:

- `A`: `accuracy = 0.50`, `mean_active_op_count = 3.0`, `mean_operator_entropy = 0.9313`
- `B`: `accuracy = 0.50`, `mean_active_op_count = 1.0`, `mean_operator_entropy = 0.0`
- `C`: `accuracy = 0.50`, `mean_active_op_count = 1.5`, `mean_operator_entropy = 0.3461`

Conclusion:

The control improved as soon as the run inherited the de-collapsed checkpoint. But bridge-trained cells were still collapsing back toward a low-diversity regime. That isolated a second issue: the bridge training path itself was overwriting the useful predicate inventory.

### 4. M3.15b Guarded Bridge Base: Preservation Fixed

Report:

- [m3_15b_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15b_relation_local_rotary\bridge_base_v2\m3_15b_20260311_bridge_base_v2\m3_15b_report.json)

Baseline:

- [m_series_bridge_baseline_manifest.json](D:\lojbanhypop\docs\baselines\m_series_bridge_baseline_manifest.json)

Key metrics:

- `A`: `overall_accuracy = 0.50`, `mean_active_op_count = 3.0`, `mean_operator_entropy = 0.9313`, `mean_top1_op_share = 0.4554`
- `B`: same accuracy and diversity profile as `A`
- `C`: same accuracy and diversity profile as `A`
- `B anti_collapse_loss ~= 6.3e-07`
- `C anti_collapse_loss ~= 1.95e-05`

Conclusion:

The guard worked. Bridge training no longer destroyed the de-collapsed predicate inventory. This was a meaningful architectural repair.

But the answer boundary still did not move:

- `accuracy_up = false`
- `positive_intervention_delta = false`
- `promote_to_next = false`

So the bridge is now mechanically safe but still not useful.

### 5. M3.15c Family-Conditioned Bridge: Usefulness Still Not Recovered

Report:

- [m3_15c_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m3_15c_family_conditioned_bridge\bridge_base_v1\m3_15c_20260311_bridge_base_v1\m3_15c_report.json)

Key metrics:

- `A/B/C overall_accuracy = 0.50`
- `A/B/C family_classification_accuracy = 0.50` on eval
- `B mean_operator_entropy = 0.9313`
- `C mean_operator_entropy = 0.5180`
- `C mean_top1_op_share = 0.7887`

Conclusion:

Family-conditioned supervision did not become a useful downstream signal. It also partially reintroduced collapse in the more aggressive residual-bias cell. This suggests the remaining bridge problem is not solved by adding another auxiliary objective that stays off the main answer path.

## Cross-Run Interpretation

### What we now know

1. `M3.15b` originally failed under hard operator collapse.
2. Upstream diversification can repair the operator inventory cleanly.
3. The bridge path can be prevented from undoing that repair.
4. Once preservation is solved, the bridge still does not improve answer quality.
5. Family-label shaping is not enough to make the bridge useful.

### What this rules out

- The problem is not only “the bridge was too global.”
- The problem is not only “the bridge destroyed the operator inventory.”
- The problem is not only “we needed a family-aware auxiliary loss.”

### Current leading diagnosis

The remaining bottleneck is bridge usefulness, not bridge stability.

More concretely:

- the predicate inventory can now survive
- the bridge can now preserve it
- but the preserved signal is still not informative enough, or not injected in the right place, to change the final decision

## Architectural Read

The recent work moved the project from a blurry failure mode to a much cleaner one.

Before these runs, “bridge doesn’t help” could have meant any of:

- bad geometry
- wrong locality
- operator collapse
- loss interaction

Now that list is shorter.

The strongest recent result is upstream, not downstream:

- `M3.8.B` proved the model can maintain a non-collapsed operator inventory under the current stack.

The strongest negative result is downstream:

- even after preserving that inventory, `M3.15b` and `M3.15c` still do not move final answer accuracy.

That implies the next useful experiments should target answer-linked usefulness rather than more bridge-side elegance.

## Recommended Next Direction

The next ablation should target direct answer-linked supervision rather than new side objectives.

Priority order:

1. Train/evaluate bridge variants against gold-vs-foil answer separation directly.
2. Probe which prompt locations and relation pairs actually drive answer flips in the Winograd slices.
3. Avoid piling on more bridge regularizers until the signal is tied to the answer path.

## Git / Workspace State At Report Time

Current git status shows no staged or unstaged file entries, so the worktree is clean from git's perspective.

There are still git warnings when status runs:

- `warning: could not open directory 'src/.pytest_tmp/': Permission denied`
- `warning: could not open directory 'src/.tmp_pytest/': Permission denied`

Those warnings mean there are inaccessible temp directories under `src/`, but they are not currently showing up as normal tracked or untracked file changes in `git status --short`.

## Recent Commits

- `da84995` `Snapshot full workspace and continue M-series refactor`
- `c05bc6d` `Add M3.15b probe and de-collapse result bundle`
- `d71ebd7` `Promote de-collapsed bridge base and guard M3.15b`
- `b32956b` `Add M3.15c family-conditioned bridge ablation`

