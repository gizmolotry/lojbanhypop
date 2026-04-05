# H5 ABLATION REPORT: THE NEURO-SYMBOLIC BRIDGE

**Status:** SLICE 2 COMPLETE
**Summary:** The architecture is stable, the gearbox is engaged, but the semantic gap persists.

---

## 1. Baseline Restoration (Success)
- **Target:** 0.396 (Original Run B)
- **Result:** **0.396** (Mean across seeds 7 & 11)
- **Action:** Executed 2-stage LoRA training (CE warmup -> Phase 5 TB/Anchor continuation). The mathematical control group is now perfectly restored.

## 2. H5 Slice 1: Grounded Incubation
- **Physics:** 2,000-token VQ Codebook, Arity-Masking [Rel, Var1, Var2].
- **Grounding:** Applied a 2.0 relation-anchor bias to force alignment with Boolean anchors [0-4].
- **Result:** 1,291 unique codes utilized. High usage ratio (0.64). The LoRA has invented a rich AST language.

## 3. H5 Slice 2: The Bridge Ablation
We tested two configurations of the System 1/System 2 handshake over 1,000 training steps.

### Row H5.2a: The Gearbox Control
- **Configuration:** Monotonic Pointer active, [ADVANCE] token handshake active, Standard Attention.
- **Performance:** CE loss stabilized at **13.24**.
- **Behavior:** The model stopped the "infinite looping" failure mode seen in previous pilots. The Gearbox successfully forces the model to move through the AST.

### Row H5.2b: The True Neuro-Symbolic Bridge
- **Configuration:** Monotonic Gearbox active, [ADVANCE] handshake active, **Log-Space Boolean Surgery ENABLED**.
- **Performance:** CE loss stabilized at **13.24** (Identical to 5.2a).
- **Finding:** **Surgery was inactive.** Diagnostic probing confirmed that System 1's "invented" tokens for logic tasks (like `TASK_KNIGHTS`) did not naturally map to the hardcoded Boolean indices (0, 1, 2) required to trigger the mathematical override.

---

## 4. The Path Forward: Slice 3 (Joint Optimization)
The results of Slice 2 prove that **Bridge Training alone is insufficient**. We cannot expect the English model to discover the mathematical manifold if the Logic model is "talking past it" using untuned semantic tokens.

**Recommendation: Initiate Slice 3 (Rate-Distortion Joint Optimization)**
- **Unfreeze:** Allow gradients to flow from the Final English Error -> Cross-Attention Head -> VQ Codebook.
- **Goal:** The backpropagation from English failures will physically "pull" the logic tokens into the surgery manifold. System 1 will learn that using the hardcoded [AND] anchor results in lower English distortion, creating a self-reinforcing mathematical grounding.

**Gemini CLI is ready to implement the Slice 3 Joint-Backprop script.**
