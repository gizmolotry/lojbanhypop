# M3.15d Answer-Path Forcing

| Cell | Regime | Acc | Adj Acc | Causal Acc | Ans Delta | Delta(gold on-off) | Active Ops | Entropy | Scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | control from de-collapsed bridge base | 0.500 | 0.500 | 0.500 | 0.5092 | 0.0000 | 3.00 | 0.9313 | 0.4010 |
| B | direct-drive gradient scalpel | 0.500 | 0.500 | 0.500 | 0.5092 | 0.0000 | 3.00 | 0.9313 | 0.4010 |
| C | late-stage causal blindfold | 0.500 | 0.500 | 0.500 | -0.0266 | 0.0000 | 3.00 | 0.9313 | 0.4010 |
| D | topological candidate pointer head | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 | 3.00 | 0.9313 | 0.4010 |

## Regimes
- B: differentiable gold-vs-foil answer-path loss through the LM answer logits.
- C: same answer-path loss, but with question tokens attention-masked out during final answer scoring.
- D: no LM answer logits; a candidate-pointer head chooses between the active candidates directly from relation-local states and advisor relation nodes.

## Anti-Collapse Controls
- anti_collapse_weight: `1.000`
- collapse_entropy_floor_ratio: `0.850`
- collapse_top1_margin: `0.100`
- collapse_top1_weight: `1.000`
- collapse_kl_weight: `0.500`

## Promotion Gates
- B:
  - accuracy_up: `False`
  - no_entropy_collapse: `True`
  - no_scope_regression: `True`
  - positive_answer_delta: `True`
  - positive_intervention_delta: `False`
  - promote_to_next: `False`
- C:
  - accuracy_up: `False`
  - no_entropy_collapse: `True`
  - no_scope_regression: `True`
  - positive_answer_delta: `False`
  - positive_intervention_delta: `False`
  - promote_to_next: `False`
- D:
  - accuracy_up: `False`
  - no_entropy_collapse: `True`
  - no_scope_regression: `True`
  - positive_answer_delta: `True`
  - positive_intervention_delta: `False`
  - promote_to_next: `False`