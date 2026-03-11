# M3.15b Relation-Local Rotary Bridge

| Cell | Acc | Adj Acc | Causal Acc | Align Loss | Align Sim | Delta(gold on-off) | Active | Entropy | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.500 | 0.500 | 0.500 | 3.4253 | 0.0181 | 0.0000 | 14.50 | 0.9313 | 0.4010 |
| B | 0.500 | 0.500 | 0.500 | 0.7375 | 0.0473 | 0.0037 | 12.50 | 0.0000 | 0.4010 |
| C | 0.500 | 0.500 | 0.500 | 0.7116 | 0.0098 | -0.0003 | 13.00 | 0.3461 | 0.4010 |

## Runtime Cue Policy
- B vs A accuracy threshold: `0.020`
- policy_selection_split: `validation`
- C runtime enabled: `False`

## Promotion Gates
- accuracy_up: `False`
- no_entropy_collapse: `False`
- no_scope_regression: `True`
- positive_intervention_delta: `True`
- seed_stability: `False`
- promote_to_next: `False`

## Local Bridge Notes
- Candidate-local mention spans are used as the English-side relation anchors.
- A cue span is inserted when a stable connector/property phrase is found in the prompt.