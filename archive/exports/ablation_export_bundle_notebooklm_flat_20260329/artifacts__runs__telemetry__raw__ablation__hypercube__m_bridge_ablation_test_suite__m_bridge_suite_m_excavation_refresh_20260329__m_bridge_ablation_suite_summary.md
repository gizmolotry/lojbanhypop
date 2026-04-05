# M Bridge Ablation Test Suite

- manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite/m_bridge_suite_m_excavation_refresh_20260329/m_bridge_ablation_suite_manifest.json`
- cell_metrics_csv: `artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite/m_bridge_suite_m_excavation_refresh_20260329/m_bridge_ablation_cell_metrics.csv`

## M3 Track Summary

### M3.15d
- control_accuracy: `0.5`
- control_answer_delta: `0.509193`
- best_cell: `B`
- B: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`0.0` intervention_gold=`0.0` promote=`False`
- C: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-0.53576` intervention_gold=`0.0` promote=`False`
- D: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-0.509185` intervention_gold=`0.0` promote=`False`

### M3.16
- control_accuracy: `0.5`
- control_answer_delta: `0.478738`
- best_cell: `B`
- B: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`0.001388` intervention_gold=`0.00068` promote=`False`
- C: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`0.000976` intervention_gold=`7.2e-05` promote=`False`
- D: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-9e-06` intervention_gold=`0.000124` promote=`False`

### M3.17
- control_accuracy: `0.5`
- control_answer_delta: `0.479211`
- best_cell: `D`
- B: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-0.584838` intervention_gold=`-12.582847` promote=`False`
- C: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-0.446055` intervention_gold=`-14.467818` promote=`False`
- D: acc=`0.5` acc_gain=`0.0` answer_delta_gain=`-0.0` intervention_gold=`-1.8e-05` promote=`False`

## Reentry Track Summary

### M3.18
- control_accuracy: `0.5`
- control_answer_delta: `-0.196083`
- best_cell: `D`
- B: acc=`0.5` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-12.606622` s_bleed=`None` promote=`False`
- C: acc=`0.5` acc_gain=`0.0` ftok=`0.0` fluency=`0.85` loop=`1.0` intervention_gold=`-13.476381` s_bleed=`None` promote=`False`
- D: acc=`0.5` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-4.3e-05` s_bleed=`None` promote=`False`
- E: acc=`0.5` acc_gain=`0.0` ftok=`0.0` fluency=`0.85` loop=`1.0` intervention_gold=`-13.174822` s_bleed=`None` promote=`False`

### M3.19
- control_accuracy: `0.0`
- control_answer_delta: `-6.383378`
- best_cell: `D0`
- D0: acc=`0.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`1e-05` s_bleed=`None` promote=`False`
- D1: acc=`0.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-2e-06` s_bleed=`None` promote=`False`
- D2: acc=`0.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-2e-06` s_bleed=`None` promote=`False`
- D3: acc=`0.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-2e-06` s_bleed=`None` promote=`False`

### M14
- control_accuracy: `1.0`
- control_answer_delta: `4.324726`
- best_cell: `D`
- B: acc=`1.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-0.00015` s_bleed=`0.0` promote=`False`
- C: acc=`1.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`0.000262` s_bleed=`0.0` promote=`False`
- D: acc=`1.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`0.000262` s_bleed=`0.0` promote=`False`
- E: acc=`1.0` acc_gain=`0.0` ftok=`0.0` fluency=`1.0` loop=`0.0` intervention_gold=`-0.000282` s_bleed=`0.0` promote=`False`

## M11 Summary

- headline_accuracy: `0.85916`
- headline_macro_f1: `0.6287`
- floor_lock_accuracy: `0.78`
- publication_mean_acc: `0.77`

## History Backfill

- history_manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/ablation_history_backfill_m_excavation_20260329/ablation_history_manifest.json`
- all_evidence_entries: `145`
- artifact_only_entries: `126`
- runnable_only_entries: `44`
- l_series_entries: `21`
- historical_gap_count: `2`
- legacy:A: canonical=`legacy.core.a` repro=`doc_only` acc=`0.167` logical=`None` ce=`None`
- legacy:H3: canonical=`legacy.h.h3` repro=`doc_only` acc=`0.0` logical=`None` ce=`None`
- legacy:H5.2b: canonical=`legacy.h5.h5_2b` repro=`doc_only` acc=`0.375` logical=`0.375` ce=`13.24`
- M2.A: canonical=`l.series.l6.l6_a` repro=`runnable` acc=`None` logical=`None` ce=`None`
- phase5.train:phase5_full: canonical=`phase5.train.phase5_full` repro=`runnable` acc=`None` logical=`None` ce=`None`
- M3.18.D: canonical=`m.track.m3_18.d` repro=`runnable` acc=`0.5` logical=`None` ce=`None`
- M14.C: canonical=`m.track.m14.c` repro=`runnable` acc=`1.0` logical=`None` ce=`None`

## Diagnosis

- Harmful cells are concentrated where the sidecar stays too exposed: M3.17/B, M3.17/C, M3.18/B, M3.18/C, M3.18/E.
- Near-neutral cells preserve control behavior but do not create measurable gains: M3.15d/B, M3.15d/C, M3.15d/D, M3.16/B, M3.16/C, M3.16/D.
- Validation-only spikes without held-out lift suggest overfitting or split-specific coupling: M3.15d/D, M3.17/C, M3.18/C, M3.18/E.
- Best-per-track snapshot: M3.15d best=B acc_gain=0.0 intervention=0.0; M3.16 best=B acc_gain=0.0 intervention=0.00068; M3.17 best=D acc_gain=0.0 intervention=-1.8e-05; M3.18 best=D acc_gain=0.0 intervention=-4.3e-05; M3.19 best=D0 acc_gain=0.0 intervention=1e-05; M14 best=D acc_gain=0.0 intervention=0.000262.
- The M11 native discriminative branch is materially stronger than the generative bridge ablations, with headline accuracy 0.85916, floor-lock accuracy 0.78, and publication mean 0.77.
- M14 scratchpad best snapshot: cell D intervention=0.000262 first_token=0.0 fluency=1.0 scratchpad_bleed=0.0.