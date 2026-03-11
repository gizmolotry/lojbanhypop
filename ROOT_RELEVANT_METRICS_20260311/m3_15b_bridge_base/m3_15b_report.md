# M3.15b Relation-Local Rotary Bridge

| Cell | Acc | Adj Acc | Causal Acc | Align Loss | Align Sim | Delta(gold on-off) | Active | Entropy | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.500 | 0.500 | 0.500 | 3.4253 | 0.0181 | 0.0000 | 14.50 | 0.9313 | 0.4010 |
| B | 0.500 | 0.500 | 0.500 | 0.7342 | 0.0465 | 0.0000 | 14.50 | 0.9313 | 0.4010 |
| C | 0.500 | 0.500 | 0.500 | 0.7107 | 0.0097 | 0.0000 | 14.50 | 0.9313 | 0.4010 |

## Runtime Cue Policy
- B vs A accuracy threshold: `0.020`
- policy_selection_split: `validation`
- C runtime enabled: `False`

## Anti-Collapse Controls
- anti_collapse_weight: `1.000`
- collapse_entropy_floor_ratio: `0.850`
- collapse_top1_margin: `0.100`
- collapse_top1_weight: `1.000`
- collapse_kl_weight: `0.500`
- bridge_train_gate_cap: `0.080`
- relation_bias_scale: `0.150`

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