# M3.11 Winograd Failure Anatomy

## Bucket Summary

| bucket | n | accuracy | active_tokens | op_entropy | scope | articulated_rate |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 16 | 0.3125 | 13.00 | 0.0000 | 0.3934 | 0.0000 |
| `easy` | 16 | 0.5000 | 13.00 | 0.0000 | 0.3934 | 0.0000 |
| `medium` | 16 | 0.3125 | 10.50 | 0.0000 | 0.4120 | 0.0000 |
| `hard` | 16 | 0.6875 | 11.75 | 0.0000 | 0.4037 | 0.0000 |

## Comparison Contract

- cross_bucket_comparable: `False`
- reason: `legacy uses legacy profile; easy/medium/hard use winograd_bench_v1.`

## Failure Taxonomy (Overall)

- causal_direction_failure: 21
- adjective_property_inversion: 11
- advisor_under_articulation: 3