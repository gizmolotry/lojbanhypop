# Run H Series

| id | name | status | return_code | key metric |
|---|---|---|---:|---|
| `H5-PROV` | `Provenance Trace` | `ok` | `0` | mean_l2_delta=225.294, exact_match=0.002 |
| `H5-OOD` | `OOD Stress Test` | `ok` | `0` | ood_acc=0.750, spatial=0.500, temporal=1.000 |
| `H5-DPTR` | `Dynamic Pointer Refactor Eval` | `ok` | `0` | dyn_acc=0.000, base_acc=0.000, delta=+0.000 |

- `Shock Tracking`: `mean_step_cos` is averaged from per-row step-wise cosine traces.
