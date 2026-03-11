# Relevant Metrics

## Included Runs

- `M3.8` de-collapse branch
- `M3.15b` guarded bridge base
- `M3.15c` family-conditioned bridge
- `M3.15d` answer-path forcing

## Headline Table

| Run | Key cell/result | Accuracy | Operator entropy | Top1 share | Scope | Main takeaway |
|---|---|---:|---:|---:|---:|---|
| `M3.8` | `B` de-collapse checkpoint | n/a | `1.6094` | `0.2028` | `0.0000` | clean upstream operator de-collapse |
| `M3.15b` | guarded bridge base `A/B/C` | `0.500` | `0.9313` | `0.4554` | `0.4010` | bridge no longer re-collapses inventory, but still no accuracy gain |
| `M3.15c` | family-conditioned `B` | `0.500` | `0.9313` | `0.4554` | `0.4010` | family supervision preserved structure but did not help |
| `M3.15c` | family-conditioned `C` | `0.500` | `0.5180` | `0.7887` | `0.4010` | residual bias partially reintroduced collapse |
| `M3.15d` | direct-drive `B` | `0.500` | `0.9313` | `0.4554` | `0.4010` | real answer-path gradient, still no accuracy lift |
| `M3.15d` | blindfold `C` | `0.500` | `0.9313` | `0.4554` | `0.4010` | answer margin went negative; blindfold harms |
| `M3.15d` | pointer head `D` | `0.500` | `0.9313` | `0.4554` | `0.4010` | structurally stable, still no answer lift |

## Key Numbers

### M3.8 de-collapse

- best cell: `M3.8.B`
- `operator_entropy = 1.6093924045562744`
- `operator_top1_share = 0.2027810513973236`
- `final_constraint_scope = 0.0`
- `final_constraint_identity = 0.0`

### M3.15b guarded bridge base

- `A/B/C overall_accuracy = 0.5`
- `A/B/C mean_active_op_count = 3.0`
- `A/B/C mean_operator_entropy = 0.93130153729747`
- `A/B/C mean_top1_op_share = 0.4554036458333333`
- `A/B/C mean_scope = 0.40100250626566414`
- `A/B/C mean_intervention_delta_gold = 0.0`

### M3.15c family-conditioned bridge

- `A/B/C overall_accuracy = 0.5`
- `B family_classification_accuracy = 0.5`
- `C family_classification_accuracy = 0.5`
- `C mean_operator_entropy = 0.5179627930614239`
- `C mean_top1_op_share = 0.7887369791666666`

### M3.15d answer-path forcing

- `A overall_accuracy = 0.5`
- `B overall_accuracy = 0.5`, `mean_answer_delta = 0.5091928094625473`
- `C overall_accuracy = 0.5`, `mean_answer_delta = -0.02656722068786621`
- `D overall_accuracy = 0.5`, `mean_answer_delta = 7.696915417909622e-06`
- all `B/C/D` kept:
  - `mean_active_op_count = 3.0`
  - `mean_operator_entropy = 0.93130153729747`
  - `mean_scope = 0.40100250626566414`
- all `B/C/D` still had:
  - `mean_intervention_delta_gold = 0.0`

## Interpretation

1. The upstream de-collapse fix is real.
2. The guarded bridge preserves that fix.
3. Family-conditioned supervision does not improve usefulness.
4. Real answer-path forcing also does not improve usefulness.
5. Blindfolding proves the bridge is not carrying enough answer-critical information on its own.

## Files

- `m3_8_decollapse/m3_8_diversification_report.json`
- `m3_15b_bridge_base/m3_15b_report.json`
- `m3_15b_bridge_base/m3_15b_report.md`
- `m3_15c_family_bridge/m3_15c_report.json`
- `m3_15c_family_bridge/m3_15c_report.md`
- `m3_15d_answer_path/m3_15d_report.json`
- `m3_15d_answer_path/m3_15d_report.md`
