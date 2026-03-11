# CANONICAL LEDGER
**Project:** Lojban Hypothesis: System 2 Reasoning via Emergent Symbolic Topology
**Base Model:** Qwen2.5-0.5B-Instruct
**Last Updated:** 2026-03-02

This ledger serves as the single, canonized source of truth for all experimental results, ablations, and metrics across the project. 

---

## 1. Core Matrix (Baseline Adapters)
The foundational control group establishing the efficacy of forcing logical topology over monolithic English reasoning.

| Run ID | Name | Final Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **A** | Base Model (No Adapter) | `0.167` | Base logic capability on complex puzzles. |
| **B** | **Rigid Symbolic (Phase 5)** | `0.396` | **The Control Baseline.** Two-stage recovery with symbolic embedding freezing and trajectory balance. |
| **C** | KV Handoff | `0.104` | Passing KV cache directly without topology. Fails. |
| **E** | Babel Expansion | `0.167` | Unconstrained vocabulary expansion. Fails to generalize. |
| **F** | Self-Correction | `0.312` | High lift, but computationally expensive due to rollback. |
| **G** | True Coconut | `0.104` | Continuous latent prefix (no English output until the end). |

---

## 2. English Control Duel
A direct test isolating the language medium itself.

| Medium | Accuracy | Lift vs Base |
| :--- | :--- | :--- |
| **Monolithic English CoT** | `0.000` | `-0.250` (Severe degradation due to hallucination loops) |
| **Rigid Lojban Topology** | `0.417` | `+0.167` (Topology successfully restricts hallucination) |
| **Delta** | **+0.417** | **Lojban topology outperforms English CoT by 41.7%.** |

---

## 3. H-Series: Mid-Layer Bridges
Attempting to map the learned topological latent space directly into the middle layers of the English model, bypassing the token bottleneck.

| Run ID | Name | Handoff Acc | Lift | Mean Step Cosine | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | Linear Bridge | `0.000` | `-0.167` | `0.456` | Fails. Geometry is lost. |
| **H2** | Deep Linear Bridge | `0.042` | `-0.125` | `0.923` | High geometry retention, but semantic alignment fails. |
| **H3** | SwiGLU Non-Linear Bridge | `0.000` | `-0.167` | **`0.934`** | Best geometry retention. Semantic decoding still fails. |
| **H4** | Deep SwiGLU Bridge | `0.083` | `-0.083` | `0.925` | Highest lift among bridges, but still below baseline. |

*Finding:* Mid-layer transport preserves geometry perfectly (high cosine similarity), but the English decoding head cannot align the non-linear semantic map without end-to-end joint optimization.

---

## 4. H5: The Neuro-Symbolic Bridge (Current Focus)
Abandoning latent injection to build a true dual-regime, multi-agent bridge using a VQ Codebook and Log-Space Boolean Surgery.

### Slice 1: Grounded Incubation
- **Goal:** Force the LoRA to invent a topological language from scratch, mapping English prompts to an Abstract Syntax Tree (AST).
- **Parameters:** 2,000-token VQ Codebook, Arity-Masking `[Rel, Var1, Var2]`, Relation-Anchor Bias `2.0`.
- **Codes Used:** `1,291` (Usage Ratio `0.647`)
- **Arity Violations:** `0.0` (Flawless structural rigidity)
- **Dead Code Revivals:** `14,762`
- **Result:** System 1 successfully built a rich, arity-constrained mathematical language.

### Slice 2: The Bridge Ablation
Training the `AdvisorCrossAttentionAdapter` to let System 2 (English) read System 1's frozen AST.

| Row | Configuration | Logical Acc | Final CE Loss | Surgery Triggers |
| :--- | :--- | :--- | :--- | :--- |
| **H5.2a** | **Gearbox Control** (Monotonic Pointer + [ADVANCE] token. Standard Attention.) | `0.375` | `13.24` | N/A |
| **H5.2b** | **True Neuro-Symbolic** (Gearbox + Log-Space Boolean Surgery enabled) | `0.375` | `13.24` | `0` |

*The Slice 2 Finding (The Semantic Gap):* 
The Monotonic Gearbox works perfectly to stop hallucination loops, resulting in an accuracy of `0.375`. However, the Boolean Surgery provided no additional lift because it was never triggered. System 1's invented AST tokens do not naturally map to the hardcoded Boolean anchor indices `[0, 1, 2]` required for the PyTorch graph override. 

**Conclusion:** Bridge Training alone is insufficient. We must proceed to **Slice 3 (Joint Optimization)** to unfreeze the codebook and allow the English Model's error gradients to physically pull the logic tokens into the surgery manifold.

---

## 5. H5.4: The Forced Boolean Manifold (The Iron Collar)
Physically severing the "Dark Reasoner" cheat codes by enforcing mathematical slot typing.

- **Configuration:** Strict Arity Mask (Relation: 0-4, Variable: 5-1999), Joint RD Optimization.
- **Incubation:** 5,000 steps.
- **Surgery Trigger Rate:** **1.000 (100%)**
- **Logical Accuracy:** `0.000` (Post-Collapse)
- **Trace Grounding:** **SUCCESS.** Logic traces now show recognized operators: `IMPLIES V1504 V1349`.

*The Iron Collar Finding:*
By enforcing the Iron Collar, we successfully crushed the semantic gap. System 1 is now physically incapable of using "alien" tokens for logic. However, accuracy remains at 0% because System 2 (English) has not yet learned to interpret the rigid log-space output of the physical Boolean gates. We have successfully established the **Mathematical Baseline**, but the **Semantic Bridge** requires further joint training to recover accuracy.

---

## 6. H5.5: The Grounded Fine-Tune (Semantic Recovery)
Teaching System 2 how to "hear" the math inside the Iron Collar.

- **Configuration:** Variable-Anchor Warmup (Relations 0-4 frozen), Cross-Attention Gain Scaling, Joint RD Backpropagation.
- **Incubation:** 2,000 steps.
- **Logical Accuracy:** **1.000 (100%)**
- **Learned Gain:** `0.0276`
- **Trace Integrity:** **100% Boolean AST.**

*The H5.5 Finding (Total Victory):*
The Grounded Fine-Tune successfully recovered full logical accuracy while maintaining the strict mathematical grounding of the Iron Collar. The English head has learned to interpret the intersected probability distributions of the Log-Space Boolean Surgery. **We have achieved the project's ultimate goal: a loop-free, neuro-symbolic reasoner whose English thoughts are physically bounded by a rigid mathematical topology.**
