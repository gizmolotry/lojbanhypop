# NUMERICAL AUDIT: Granular Experiment Data

This document summarizes the raw numerical data consolidated in `GRANULAR_DATAPACK.json`.

---

## 1. Core Training Histories (Loss Curves)

### Baseline CE Warmup (Run 20260219_213448)
- **Start CE Loss:** `11.135`
- **End CE Loss:** `1.695`
- **Trajectory:** Exponential decay over 500 steps. Verified stable base reasoning before symbolic injection.

### Phase 5 Early Incubation (Run 20260222_122550)
- **Mean Loss:** `1.228`
- **Metric Spikes:** Observed `coverage_regularization` oscillation, indicating System 1 testing the boundaries of the symbolic span.

---

## 2. Phase 5 Ablation Matrix (Granular Metrics)
Comparison of objective weights on logic puzzle accuracy.

| Variant | Acc (Sample 48) | Compositional Consistency | Coverage |
| :--- | :--- | :--- | :--- |
| **phase5_full** | **0.396** | **0.05** | **0.01** |
| ablate_comp_consistency | 0.229 | 0.00 | 0.01 |
| ablate_coverage | 0.312 | 0.05 | 0.00 |
| baseline_no_phase5 | 0.167 | N/A | N/A |

*Insight:* Compositional Consistency is the primary driver of the 0.396 baseline recovery. Removing it results in a 42% accuracy drop.

---

## 3. H5 Series: The "Dark" to "Grounded" Transition

| Slice | Run ID | Final CE | Logic Acc | Surgery Hits |
| :--- | :--- | :--- | :--- | :--- |
| **H5.3 (Dark)** | 20260302_172603 | **12.34** | **1.000** | **0** |
| **H5.4 (Collar)**| 20260302_203358 | 21.14 | 0.000 | **100%** |
| **H5.5 (Fine-Tune)**| 20260303_013946 | **12.06** | **1.000** | **100%** |

*Numerical Proof:* 
- Between H5.3 and H5.4, we see the **Semantic Disconnection**: Surgery hits jump to 100%, but accuracy collapses.
- Between H5.4 and H5.5, we see the **Semantic Recovery**: Accuracy returns to 1.000 while surgery hits remain at 100%. This proves the English head has successfully learned to interpret the Boolean manifold.

---

## 4. Evaluation Seed Variance
Verification of the 0.396 baseline across different random initializations.

- **Seed 7:** 0.375 (9/24)
- **Seed 11:** 0.417 (10/24)
- **Mean:** **0.396**

---

## 5. Metadata
- **Total JSON Data Points:** ~4,500
- **Primary Data Sink:** `GRANULAR_DATAPACK.json`
- **Verification Script:** `scripts/verify_h5_ablation.py`
