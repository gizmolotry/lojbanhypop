# TECHNICAL APPENDIX: Experiment Manifest & Raw Metrics

**Date:** 2026-03-02
**Project:** Lojban Hypothesis

This document lists the exact data locations, hyperparameters, and numerical outputs for all valid experimental runs.

---

## 1. Asset Manifest (Primary Results)

| Run Type | ID | Accuracy | Asset Directory |
| :--- | :--- | :--- | :--- |
| **Control Baseline** | Run B | `0.396` | `runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5` |
| **Dark Reasoner** | H5.3 | `1.000` | `runs/true_coconut_h5/20260302_172603/h5_checkpoint.pt` |
| **Iron Collar** | H5.4 | `0.000` | `runs/true_coconut_h5/20260302_203358/h5_checkpoint.pt` |
| **I-Step Escalation**| H5.5i | `0.200` | `runs/true_coconut_h5/20260303_031401/h5_checkpoint.pt` |

---

## 2. H5 Series: Chronological Telemetry

### Slice 1: Grounded Incubation
- **Run Dir:** `runs/true_coconut_h5/20260302_053614`
- **Parameters:** VQ Codebook (2000), Relation Bias (2.0), Arity Mask.
- **Result:** 1,291 unique codes mapped. 0 arity violations.

### Slice 2a: Gearbox Control
- **Run Dir:** `runs/true_coconut_h5/20260302_062528`
- **Configuration:** [ADVANCE] Handshake + Monotonic Pointer. Standard Attention.
- **Finding:** Cured looping failure mode. Accuracy: `0.375`.

### Slice 2b: Isolated Surgery
- **Run Dir:** `runs/true_coconut_h5/20260302_070646`
- **Configuration:** Log-Space Boolean Surgery active (Isolated).
- **Finding:** Surgery inactive (0 hits) due to semantic gap. Accuracy: `0.375`.

### Slice 3: Joint RD (The Dark Reasoner)
- **Run Dir:** `runs/true_coconut_h5/20260302_172603`
- **Configuration:** Unfrozen VQ Codebook. Joint backprop.
- **Finding:** Achieved `1.000` accuracy, but tokens remained "alien" (opaque).

### Slice 4: The Iron Collar
- **Run Dir:** `runs/true_coconut_h5/20260302_203358`
- **Configuration:** Strict Arity Mask (Rel: 0-4, Var: 5-1999).
- **Finding:** Physically forced surgery (100% trigger). Accuracy collapsed to `0.000`.

### Slice 5: Grounded Fine-Tune
- **Run Dir:** `runs/true_coconut_h5/20260303_013946`
- **Configuration:** Variable Warmup + Gain Scaling.
- **Finding:** Partial recovery. Accuracy: `0.300`. (Improved to 1.000 in I-Series).

---

## 3. Training Infrastructure

- **Master Script:** `scripts/train_h5_persistent_vq_advisor.py`
  - Implements: Council Cross-Attention, Iron Collar, Handshake Enforcement, Distillation.
- **Eval Engine:** `scripts/verify_h5_ablation.py`
  - Calculates grounded accuracy and surgery trigger rates.
- **Ledger System:** `CANONICAL_LEDGER.json`
  - Structured DB for all Project results.

---

## 4. Hardware & Environment
- **OS:** win32
- **GPU:** NVIDIA (Verified via CUDA logs)
- **Library versions:** 
  - `transformers`: Latest
  - `peft`: 0.13.0
  - `lojban_evolution`: Internal Experimental Branch
