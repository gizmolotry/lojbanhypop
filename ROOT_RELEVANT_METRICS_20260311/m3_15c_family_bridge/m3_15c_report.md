# M3.15c Family-Conditioned Bridge

| Cell | Acc | Adj Acc | Causal Acc | Align Loss | Align Sim | Delta(gold on-off) | Active | Entropy | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.500 | 0.500 | 0.500 | 3.3788 | -0.0125 | 0.0000 | 14.50 | 0.9313 | 0.4010 |
| B | 0.500 | 0.500 | 0.500 | 0.6426 | -0.0034 | 0.0000 | 14.50 | 0.9313 | 0.4010 |
| C | 0.500 | 0.500 | 0.500 | 0.6149 | -0.0251 | 0.0000 | 14.50 | 0.5180 | 0.4010 |

## Runtime Cue Policy
- B vs A accuracy threshold: `0.020`
- policy_selection_split: `validation`
- C runtime enabled: `False`
- family_loss_weight: `0.500`

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
- Bridge conditioning includes a family embedding and a family classification auxiliary loss.