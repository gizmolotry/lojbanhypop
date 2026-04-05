# M3.18 Decoder Re-entry Resume

| Cell | Regime | Acc | FTok | Fluency | Contam | Loop | Mention | Answer Delta | Gold On-Off | Return Norm | Residual Norm | Scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | control no advisor | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | -6.3834 | 0.0000 | 0.0000 | 0.0000 | 0.4001 |
| B | frozen single return token | 0.500 | 0.000 | 0.850 | 0.000 | 1.000 | 0.000 | 0.2291 | -8.3788 | 0.0071 | 0.0000 | 0.4001 |
| C | frozen multi-return token bundle | 0.000 | 0.000 | 0.850 | 0.000 | 1.000 | 0.000 | -6.1453 | -8.6917 | 0.0072 | 0.0000 | 0.4001 |
| D | learned residual continuation vector | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | -6.3834 | 0.0000 | 0.0016 | 0.0016 | 0.4001 |
| E | hybrid token plus residual translator | 0.000 | 0.000 | 0.850 | 0.000 | 1.000 | 0.000 | -5.8989 | -10.7930 | 0.0049 | 0.0041 | 0.4001 |

## Regimes
- B: one-shot single return token before answer continuation.
- C: one-shot short return-token bundle before answer continuation.
- D: one-shot residual continuation vector.
- E: hybrid token bundle plus residual continuation vector.

## Continuation Metrics
- `FTok`: first-token correctness under one-shot decoder resumption.
- `Fluency`: rough English-likeness after re-entry.
- `Contam`: Lojbanic or sidecar-bleed rate in the resumed text.
- `Loop`: short repetition/degeneracy rate in resumed text.
- `Mention`: whether the resumed continuation explicitly mentions the gold answer.

## Promotion Gates
- B:
  - accuracy_up: `True`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - fluency_preserved: `False`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `False`
  - first_token_preserved: `True`
  - seed_stability: `True`
  - promote_to_next: `False`
- C:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - fluency_preserved: `False`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `False`
  - first_token_preserved: `True`
  - seed_stability: `False`
  - promote_to_next: `False`
- D:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - fluency_preserved: `True`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `True`
  - first_token_preserved: `True`
  - seed_stability: `False`
  - promote_to_next: `False`
- E:
  - accuracy_up: `False`
  - no_scope_regression: `True`
  - positive_intervention_delta: `False`
  - fluency_preserved: `False`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `False`
  - first_token_preserved: `True`
  - seed_stability: `False`
  - promote_to_next: `False`