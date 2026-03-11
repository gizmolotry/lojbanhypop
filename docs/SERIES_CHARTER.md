# Series Charter

This file defines non-overlapping responsibilities for each experiment series.

## Purpose Map

| Series | Purpose | Canonical Scripts | Canonical Local Output Roots | Core Success Metrics |
|---|---|---|---|---|
| `A-G` | Benchmark ablation comparisons | `scripts/run_coconut_ablation_matrix.py` | `runs/ablation/a_to_g` | final accuracy lift, symbolic lift, run-level deltas |
| `J` | Data invariance + adversarial synthesis diagnostics | `scripts/eval_j_1.py`..`scripts/eval_j_5.py` | `runs/j_series` | invariance rate, generator accept rate, accepted foil pair accuracy, accept rate by depth |
| `L` | Constraint-optimized training (Lexicographic Augmented Lagrangian) | `scripts/train_l_series_mvs.py` | `runs/l_series` | Tier-A violations, lambda stability, training convergence |
| `M` | Merged modern stack runs (`J -> recursive scope -> foils/truth -> L tiers -> gates`) | `scripts/run_true_coconut_h_series.py` (J/M1), `scripts/run_l6_ablation_branch.py` (M2), upcoming Phase-3 gate run | `runs/j_series`, `runs/l_series/l6_ablation` | gate pass rate, scope-by-depth, identity guardrail, foil minimal-edit integrity |

## Enforcement

Boundary enforcement is implemented in `src/lojban_evolution/series_contract.py`:

1. Output path allowlists by series.
2. Series metadata tags embedded in artifacts (`series_id`, `track`, `script`).
3. Runtime checks raise fatal errors when a script writes outside its series roots.

## Airflow Role

Airflow orchestrates execution order, scheduling, and partition validation.
Airflow does not define scientific meaning. Series meaning is defined by this charter plus script-level series checks.

## Baseline Manifest Policy

For M-series family runs, a baseline manifest is mandatory and must declare inherited upstream baselines:

- `upstream_best.j_series` (best locked J baseline)
- `upstream_best.l_series` (best locked L baseline)
- `m_base` fields: `dataset`, `constraints`, `identity_reg`, `curriculum`, `optimizer`

Canonical default path: `docs/baselines/m_series_baseline_manifest.json`.

## M-Series Mapping

`M` is a taxonomy layer over historical run IDs, not a destructive rename.

- `M0.1` = `H5-PROV`
- `M0.2` = `H5-OOD`
- `M0.3` = `H5-DPTR`
- `M1.1` = `J-1`
- `M1.2` = `J-2`
- `M1.3` = `J-4`
- `M1.4` = `J-5`
- `M2.A` = `L6-A`
- `M2.B` = `L6-B`
- `M2.C` = `L6-C`

## Metric Naming Rules

- `foil_auc` is deprecated. Use `accepted_foil_pair_accuracy` unless a true ROC-AUC is explicitly computed.
- `scope_by_depth` is deprecated when it reflects generator acceptance. Use `accept_rate_by_depth` for acceptance and reserve scope metrics for reparsed or gold scopal structure.
