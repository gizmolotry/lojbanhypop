# M3.17 Advisor Re-entry Bridge

| Cell | Regime | Acc | Adj Acc | Causal Acc | Answer Delta | Gold On-Off | Return Norm | Gate | Attn Entropy | Scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | control no re-entry | 0.500 | 0.500 | 0.500 | 0.4792 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4010 |
| B | single return-state bottleneck | 0.500 | 0.500 | 0.500 | -0.1056 | -12.5828 | 0.0015 | 0.1192 | 4.9698 | 0.4010 |
| C | three return-state bottleneck | 0.500 | 0.500 | 0.500 | 0.0332 | -14.4678 | 0.0011 | 0.1192 | 4.9698 | 0.4010 |
| D | direct residual re-encoder | 0.500 | 0.500 | 0.500 | 0.4792 | -0.0000 | 0.0066 | 0.1192 | 4.9697 | 0.4010 |

## Regimes
- B: compress advisor state into one return state appended once before answer continuation.
- C: compress advisor state into a short return-state bundle appended before answer continuation.
- D: translate advisor state into one decoder-compatible residual continuation vector.

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