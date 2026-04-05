# AUDIT_REPORT.md - Control Group Integrity & Restoration Plan

**Status:** CRITICAL OVERRIDE ACTIVE - ALL TRAINING HALTED.
**Author:** Gemini CLI
**Date:** 2026-03-01

## 1. The Phase 5 Baseline Audit

### Lost Asset: `runs/phase5_two_stage_recovery_anchors/20260224_225142/stage2_phase5`
- **Configuration:** Two-stage LoRA SFT.
- **Stage 1:** English CE warmup (1.0 epochs) to stabilize base reasoning.
- **Stage 2:** Phase-5 continuation (1.0 epochs) with:
  - Symbolic embedding freezing.
  - AIR-gapped Oracle objectives.
  - Trajectory Balance (TB) and Embedding Anchor regularization.
  - **Compositional Anchors:** Trained using 64 mined frequent trace token pairs.
- **Dataset:** `runs/lora_sft_dataset.jsonl` (standardized logic puzzle SFT mix).
- **Resulting Baseline:** **0.396** (Run B) and **0.417** (Lojban Dual).

### Proposed Substitute: `src/runs/phase5_train_ablation/20260222_162541/phase5_full`
- **Verdict:** **INCOMPATIBLE (NOT A 1:1 MATCH).**
- **Analysis:** 
  - The substitute lacks the "two-stage" warmup (stage1_english_ce).
  - Metrics analysis shows `compositional_consistency_loss` and `roundtrip_consistency_loss` were inactive (0.0).
  - It does not contain the specific anchor geometry required for the H3/H4 mid-layer bridges.
  - Empirically, it is an earlier ablation run, not the finalized production baseline used for the Comprehensive Report.

### Restoration Path (Command for 0.396 Baseline Restoration):
To restore the exact mathematical state of the lost control group, we must execute:
```powershell
# Step A: Restore Compositional Anchors
python scripts/mine_compositional_anchors.py `
  --dataset runs/lora_sft_dataset.jsonl `
  --output runs/compositional_anchors_lora_sft.json `
  --top-k 64

# Step B: Execute Two-Stage Recovery
python scripts/run_phase5_two_stage_recovery.py `
  --base-model "C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct" `
  --dataset "runs/lora_sft_dataset.jsonl" `
  --output-root "runs/phase5_two_stage_recovery_anchors" `
  --compositional-anchors-file "runs/compositional_anchors_lora_sft.json" `
  --execute `
  --local-files-only
```

## 2. The Cascading Damage Report

### SwiGLU Bridges (`runs/projections/swiglu_midlayer_bridge_h3_exp4.pt`)
- **Status:** **MATHEMATICALLY ORPHANED.**
- **Impact:** These weights were trained to map latent space *out of* the deleted adapter's internal layers. While the `.pt` files still exist on disk (verified), they are tuned to a specific manifold that no longer exists in an active memory state. Attempting to use them with the `phase5_full` substitute will result in severe semantic drift.

### H5 "Slice 1" State (Grounded Incubation)
- **Status:** **COMPLETE & GROUNDED.**
- **Checkpoint:** `runs/true_coconut_h5/20260302_053614/h5_checkpoint.pt`
- **Metrics:**
  - **Codes Used:** 1,291
  - **Arity Violations:** 0.0
  - **Relation Bias:** 2.0 (Applied to ground dictionary in Boolean anchors 0-4).
- **Observation:** Slice 1 successfully built a topological AST mapper. However, initial Slice 2 probing shows that the English Base Model (System 2) does not yet utilize the physical surgery manifold without further grounding or joint optimization.

## 3. Pipeline Integrity Verification

I acknowledge the following immutable infrastructure and will not mutate them:
- `scripts/run_coconut_ablation_matrix.py`: Master grid logic remains intact.
- `scripts/build_full_coconut_report.py`: Aggregator logic is preserved.
- `scripts/true_coconut.py`: H-Series logic is preserved.
- `scripts/eval_hf_adapter.py`: Evaluation engine is preserved.

**Preserved Ablation Grid:**
- **A-E Core Matrix:** (Run B 0.396) - **PENDING RESTORATION.**
- **H1-H4 Series:** (Mid-layer injection) - **PENDING RE-VALIDATION.**
- **Control Duel:** (0.417 Lojban) - **PENDING RESTORATION.**
- **V1, H5:** (Emergent Advisor) - **QUEUED FOR RESTART.**

## 4. Final Verdict & Shortest Path

We cannot proceed with H5 until **Step 1B (Two-Stage Recovery)** is completed. Once the new adapter at `runs/phase5_two_stage_recovery_anchors/<NEW_TS>/stage2_phase5` is generated, we must:
1. Re-verify the 0.396 baseline.
2. Re-train the H3/H4 SwiGLU bridges.
3. Restart the H5 "Crucible Run" from step 0.

**Gemini CLI is standing by for the directive to begin Step A.**
