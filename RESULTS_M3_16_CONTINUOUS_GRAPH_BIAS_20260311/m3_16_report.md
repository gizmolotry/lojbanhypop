# M3.16 Continuous Graph Bias

| Cell | Regime | Acc | Adj Acc | Causal Acc | Answer Delta | Gold On-Off | Active Ops | Entropy | Scope | Cand Mass | Cue Mass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | control frozen bridge base | 0.500 | 0.500 | 0.500 | 0.4787 | 0.0000 | 3.00 | 0.9314 | 0.4010 | 0.0000 | 0.0000 |
| B | candidate-only soft graph bias | 0.500 | 0.500 | 0.500 | 0.4801 | 0.0007 | 3.00 | 0.9314 | 0.4010 | 1.0000 | 0.0000 |
| C | candidate+cue soft graph bias | 0.500 | 0.500 | 0.500 | 0.4797 | 0.0001 | 3.00 | 0.9314 | 0.4010 | 0.7044 | 0.2956 |
| D | global prompt soft graph bias | 0.500 | 0.500 | 0.500 | 0.4787 | 0.0001 | 3.00 | 0.9314 | 0.4010 | 0.0615 | 0.0298 |

## Regimes
- B: graph-derived bias only on candidate spans.
- C: graph-derived bias on candidate spans plus cue span.
- D: graph-derived bias over the whole prompt token field.

## Promotion Gates
- B:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - seed_stability: `True`
  - promote_to_next: `False`
- C:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - seed_stability: `True`
  - promote_to_next: `False`
- D:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - seed_stability: `True`
  - promote_to_next: `False`