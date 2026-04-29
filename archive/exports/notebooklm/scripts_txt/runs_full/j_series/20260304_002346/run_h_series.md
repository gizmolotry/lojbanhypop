# Run H Series

| id | name | status | return_code | key metric |
|---|---|---|---:|---|
| `J-1` | `Graph Target (Factor Schema)` | `ok` | `0` | schema_valid_rate=1.000, graphs=4 |
| `J-2` | `Paraphrase Explosion (Invariance)` | `ok` | `0` | invariance_rate=1.000, variants=4000 |
| `J-3` | `Stop-Grad Isolation Gate` | `ok` | `0` | stopgrad_pass=0 |
| `J-4` | `Operator Curriculum Build` | `ok` | `0` | samples=1280, operators=5 |

- `Shock Tracking`: `mean_step_cos` is averaged from per-row step-wise cosine traces.
