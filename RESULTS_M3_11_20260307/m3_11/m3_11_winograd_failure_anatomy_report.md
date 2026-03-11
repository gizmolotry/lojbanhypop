# M3.11 Winograd Failure Anatomy

## Bucket Summary

| bucket | n | accuracy | active_tokens | op_entropy | scope | articulated_rate |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 24 | 0.2917 | 25.67 | 0.9131 | 0.3305 | 1.0000 |
| `easy` | 24 | 0.4167 | 24.67 | 0.8622 | 0.3357 | 1.0000 |
| `medium` | 24 | 0.2500 | 20.58 | 0.6326 | 0.3550 | 0.6667 |
| `hard` | 24 | 0.6250 | 25.67 | 0.6532 | 0.3387 | 0.6250 |

## Failure Taxonomy (Overall)

- causal_direction_failure: 30
- adjective_property_inversion: 22
- advisor_under_articulation: 6