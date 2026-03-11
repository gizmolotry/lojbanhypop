# M3.15b Relation-Local Rotary Bridge

| Cell | Acc | Adj Acc | Causal Acc | Align Loss | Align Sim | Delta(gold on-off) | Active | Entropy | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.375 | 0.250 | 0.500 | 2.0683 | 0.0000 | 0.0000 | 11.00 | 0.0000 | 0.4087 |
| B | 0.375 | 0.250 | 0.500 | 0.1839 | 0.0000 | 0.0000 | 11.00 | 0.0000 | 0.4087 |
| C | 0.375 | 0.250 | 0.500 | 0.1894 | 0.0000 | 0.0000 | 11.00 | 0.0000 | 0.4087 |

## Runtime Cue Policy
- B vs A accuracy threshold: `0.020`
- policy_selection_split: `validation`
- C runtime enabled: `False`

## Promotion Gates
- accuracy_up: `False`
- no_entropy_collapse: `True`
- no_scope_regression: `True`
- positive_intervention_delta: `False`
- seed_stability: `False`
- promote_to_next: `False`

## Local Bridge Notes
- Candidate-local mention spans are used as the English-side relation anchors.
- A cue span is inserted when a stable connector/property phrase is found in the prompt.