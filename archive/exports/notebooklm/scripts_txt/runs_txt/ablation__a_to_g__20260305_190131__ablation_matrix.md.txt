# Coconut Fusion Ablation Matrix

| id | name | status | return_code | key metric |
|---|---|---|---:|---|
| `A` | `Control (English CoT -> English)` | `ok` | `0` | base_final=0.167, adapter_final=0.438, handoff_final=0.000 |
| `B.1` | `Legacy Text-to-Text (No Handoff)` | `ok` | `0` | base_final=0.167, adapter_final=0.438, handoff_final=0.000 |
| `C` | `Coconut Fusion (Latent KV Handoff)` | `ok` | `0` | base_final=0.167, adapter_final=0.438, handoff_final=0.000 |
| `B.2` | `Enhanced Constraint Text-to-Text (No Handoff)` | `ok` | `0` | base_final=0.167, adapter_final=0.000, handoff_final=0.083 |
| `D` | `NoPE Fusion (DroPE + latent handoff)` | `ok` | `0` | base=0.000, handoff=0.000, lift=+0.000 |
| `E` | `Babel Bridge (Projected latent handoff)` | `skipped` | `None` |  |

## Run H Series
| id | name | status | key metric |
|---|---|---|---|
| `H1` | `Multi-Vector Injection (Bandwidth)` | `pending` |  |
| `H2` | `Mid-Layer Injection (Depth)` | `pending` |  |
| `H3` | `SwiGLU Mid-Layer Bridge (Non-Linear Alignment)` | `pending` |  |
| `H4` | `Persistent SwiGLU Injection (Continuous Anchor)` | `pending` |  |

- `Shock Tracking`: log per-step cosine for injected states (`step_cosine`) to observe persistence vs evaporation.
