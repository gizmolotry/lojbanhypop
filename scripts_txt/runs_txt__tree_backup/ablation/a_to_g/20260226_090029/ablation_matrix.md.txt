# Coconut Fusion Ablation Matrix

| id | name | status | return_code | key metric |
|---|---|---|---:|---|
| `A` | `Control (English CoT -> English)` | `ok` | `0` | base_final=0.167, adapter_final=0.396, handoff_final=0.104 |
| `B` | `Rigid Lojban (Text-to-Text)` | `ok` | `0` | base_final=0.167, adapter_final=0.396, handoff_final=0.104 |
| `C` | `Coconut Fusion (Latent KV Handoff)` | `ok` | `0` | base_final=0.167, adapter_final=0.396, handoff_final=0.104 |
| `D` | `NoPE Fusion (DroPE + latent handoff)` | `ok` | `0` | base=0.000, handoff=0.000, lift=+0.000 |
| `E` | `Babel Bridge (Projected latent handoff)` | `ok` | `0` | handoff_final_lift=+0.000, handoff_symbolic_lift=+0.188, gate=PASS |

## Trinity Expansion
- `Drift Value` mean_cosine=0.3572 (partial overlap).
- `Run F (Self-Correct)` mean_acc=0.312, lift=+0.146.
- `Run E (Babel)` bridge trained on 15 examples; projection: `runs\projections\babel_bridge_trained.pt`.
- `Run H3 (SwiGLU)` bridge trained for 500 steps on 19 Run-B success pairs; projection: `runs\projections\swiglu_midlayer_bridge_h3.pt`.

## Run H Series
| id | name | status | key metric |
|---|---|---|---|
| `H1` | `Multi-Vector Injection (Bandwidth)` | `ok` | base=0.167, handoff=0.000, lift=-0.167, mean_step_cos=0.456 |
| `H2` | `Mid-Layer Injection (Depth)` | `ok` | base=0.167, handoff=0.042, lift=-0.125, mean_step_cos=0.923 |
| `H3` | `SwiGLU Mid-Layer Bridge (Non-Linear Alignment)` | `ok` | base=0.167, handoff=0.000, lift=-0.167, mean_step_cos=0.934 |
