# Ablation Program Map

- Generated UTC: `2026-03-30T15:48:52.773002+00:00`
- Source history manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/ablation_history_backfill_letter_series_refined_20260330/ablation_history_manifest.json`
- Concentrated family count: `33`

## Program Layers

- `legacy_orchestration`: letter-era experiments and their early DAG architecture
- `bridge_and_serialization`: early-to-mid M-series bridge, grounding, and serialization families
- `manifold_and_return_path`: later manifold/native/discriminative/re-entry families
- `control_plane`: the backfill, catalog, and aggregate-suite layer

## Concentrated Families

### A-G

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `a_to_g.a, a_to_g.b_1, a_to_g.b_2, a_to_g.c, a_to_g.d, a_to_g.e`
- Legacy aliases: `A, A-G/A, A-G/B.1, A-G/B.2, A-G/C, A-G/D, A-G/E, B.1, B.2, C, D, E`
- Entry count: `6`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Doc-only rows: `1`
- Brief: Babel Bridge (Projected latent handoff) + 5 more
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

### H

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `h.series.h3, legacy.h.h1, legacy.h.h2, legacy.h.h3, legacy.h.h4`
- Legacy aliases: `H1, H2, H3, H4`
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Doc-only rows: `4`
- Brief: Deep Linear Bridge + 4 more
- Scripts: `D:/lojbanhypop/scripts/true_coconut.py`

### H5

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `h.series.h5_dptr, h.series.h5_ood, h.series.h5_prov`
- Legacy aliases: `H5-DPTR, H5-OOD, H5-PROV`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: Dynamic Pointer Refactor Eval + 2 more
- Scripts: `D:/lojbanhypop/scripts/eval_h5_dynamic_pointer_refactor.py, D:/lojbanhypop/scripts/eval_h5_ood_stress.py, D:/lojbanhypop/scripts/trace_h5_provenance.py`

### J

- Layer: `legacy_orchestration`
- Status: `runnable`
- Normalized IDs: `M1.1, M1.2, M1.3, M1.4, M1.5`
- Legacy aliases: `J-1, J-2, J-3, J-4, J-5`
- Entry count: `5`
- Runnable rows: `5`
- Brief: Adversarial Synthesis (Scope/Foil) + 4 more
- Scripts: `D:/lojbanhypop/scripts/eval_j_1.py, D:/lojbanhypop/scripts/eval_j_2.py, D:/lojbanhypop/scripts/eval_j_3.py, D:/lojbanhypop/scripts/eval_j_4.py, D:/lojbanhypop/scripts/eval_j_5.py, D:/lojbanhypop/scripts/train_h5_persistent_vq_advisor.py`
- DAGs: `airflow/dags/lojban_j_series_dag.py`

### L

- Layer: `legacy_orchestration`
- Status: `partially_runnable`
- Normalized IDs: `M2.1, M2.2, M2.3, M3.0, M3.1, M3.2, M3.3, M3.4, l.branch.m3_5.m3_5_a, l.branch.m3_5.m3_5_b, l.branch.m3_5.m3_5_c, l.branch.m3_6.m3_6_a, l.branch.m3_6.m3_6_b, l.branch.m3_6.m3_6_c, l.branch.m3_7.m3_7_a, l.branch.m3_7.m3_7_b, l.branch.m3_7.m3_7_c, l.branch.m3_8.m3_8_a, l.branch.m3_8.m3_8_b, l.branch.m3_8.m3_8_c, l.series.charter`
- Legacy aliases: `L-Series, L6-A, L6-B, L6-C, Lagrangian Series, M2.A, M2.B, M2.C, M3.5.A, M3.5.B, M3.5.C, M3.6.A, M3.6.B, M3.6.C, M3.6.M3.6.A, M3.6.M3.6.B, M3.6.M3.6.C, M3.7.A, M3.7.B, M3.7.C, M3.8.A, M3.8.B, M3.8.C`
- Entry count: `21`
- Runnable rows: `3`
- Artifact-only rows: `17`
- Doc-only rows: `1`
- Brief: Ablated anchor (swap-test disabled) + 20 more
- Scripts: `scripts/run_l6_ablation_branch.py, scripts/train_l_series_mvs.py`
- DAGs: `airflow/dags/lojban_l_series_dag.py`

### Phase Eval

- Layer: `legacy_orchestration`
- Status: `runnable`
- Normalized IDs: `phase5.objective.ablate_compositional_consistency_loss, phase5.objective.ablate_compression_regularization_loss, phase5.objective.ablate_coverage_regularization_loss, phase5.objective.ablate_roundtrip_consistency_loss, phase5.objective.ablate_semantic_unambiguity_loss, phase5.objective.baseline_no_phase5, phase5.objective.phase5_full, phase5.train.ablate_compositional_consistency_weight, phase5.train.ablate_compression_regularization_weight, phase5.train.ablate_coverage_regularization_weight, phase5.train.ablate_roundtrip_consistency_weight, phase5.train.ablate_semantic_unambiguity_weight, phase5.train.baseline_no_phase5, phase5.train.phase5_full`
- Legacy aliases: `ablate_compositional_consistency_loss, ablate_compositional_consistency_weight, ablate_compression_regularization_loss, ablate_compression_regularization_weight, ablate_coverage_regularization_loss, ablate_coverage_regularization_weight, ablate_roundtrip_consistency_loss, ablate_roundtrip_consistency_weight, ablate_semantic_unambiguity_loss, ablate_semantic_unambiguity_weight, baseline_no_phase5, phase5_full`
- Entry count: `14`
- Runnable rows: `14`
- Brief: ablate_compositional_consistency_loss + 11 more
- Scripts: `scripts/run_phase5_objective_ablation.py, scripts/run_phase5_train_ablation.py`
- DAGs: `airflow/dags/lojban_phase_ablation_dag.py`

### M3.9

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.9`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.9 telemetry report
- Scripts: `scripts/run_m3_9_primitive_probe.py`
- DAGs: `airflow/dags/lojban_m3_9_primitive_probe_dag.py`

### M3.10

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.10`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.10 telemetry report
- Scripts: `scripts/run_m3_10_ood_accuracy_probe.py`
- DAGs: `airflow/dags/lojban_m3_10_ood_accuracy_probe_dag.py`

### M3.11

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.11`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.11 telemetry report
- Scripts: `scripts/run_m3_11_winograd_failure_anatomy.py`
- DAGs: `airflow/dags/lojban_m3_11_winograd_failure_anatomy_dag.py`

### M3.12

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.12.A, M3.12.B, M3.12.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.12 A + 2 more
- DAGs: `airflow/dags/lojban_m3_12_geometric_return_stream_dag.py`

### M3.13

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.13.A, M3.13.B, M3.13.C, M3.13.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.13 A + 3 more

### M3.14

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.14.A, M3.14.B, M3.14.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.14 A + 2 more
- DAGs: `airflow/dags/lojban_m3_14_structural_alignment_bridge_dag.py`

### M3.15

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15.A, M3.15.B, M3.15.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15 A + 2 more
- DAGs: `airflow/dags/lojban_m3_15_rotary_coconut_dag.py, airflow/dags/lojban_m3_15_rotary_coconut_seven_dag.py`

### M3.15b

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15b.A, M3.15b.B, M3.15b.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15b A + 2 more
- DAGs: `airflow/dags/lojban_m3_15b_relation_local_rotary_dag.py`

### M3.15c

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15c.A, M3.15c.B, M3.15c.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15c A + 2 more
- DAGs: `airflow/dags/lojban_m3_15c_family_conditioned_bridge_dag.py`

### M3.15d

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15d.A, M3.15d.B, M3.15d.C, M3.15d.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.15d A + 3 more
- DAGs: `airflow/dags/lojban_m3_15d_answer_path_forcing_dag.py`

### M3.16

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.16.A, M3.16.B, M3.16.C, M3.16.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.16 A + 3 more
- DAGs: `airflow/dags/lojban_m3_16_continuous_graph_bias_dag.py`

### M3.17

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.17.A, M3.17.B, M3.17.C, M3.17.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: control no re-entry + 3 more
- DAGs: `airflow/dags/lojban_m3_17_advisor_reentry_bridge_dag.py`

### M3.18

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.18.A, M3.18.B, M3.18.C, M3.18.D, M3.18.E`
- Legacy aliases: `A, B, C, D, E`
- Entry count: `5`
- Runnable rows: `5`
- Brief: control no advisor + 4 more
- DAGs: `airflow/dags/lojban_m3_18_decoder_reentry_resume_dag.py`

### M3.19

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.19.D0, M3.19.D1, M3.19.D2, M3.19.D3`
- Legacy aliases: `D0, D1, D2, D3`
- Entry count: `4`
- Runnable rows: `4`
- Brief: M3.19 D0 + 3 more
- DAGs: `airflow/dags/lojban_m3_19_d_mainline_grid_dag.py`

### M4.0

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M4.0`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M4.0 telemetry report
- Scripts: `scripts/run_m4_0_semantic_probe.py`
- DAGs: `airflow/dags/lojban_m4_0_semantic_probe_dag.py`

### M4.2

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M4.2`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M4.2 telemetry report
- Scripts: `scripts/run_m4_2_predicate_grounding.py`
- DAGs: `airflow/dags/lojban_m4_2_predicate_grounding_dag.py`

### M5

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M5.0.A, M5.0.B, M5.0.C`
- Legacy aliases: `M5.A, M5.B, M5.C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: Reuse-oriented control + 2 more
- DAGs: `airflow/dags/lojban_m5_autoformalization_dag.py`

### M5.1

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M5.1.N0, M5.1.N1, M5.1.N2, M5.1.N3`
- Legacy aliases: `M5.N0, M5.N1, M5.N2, M5.N3`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: Add counterfactual invariance + 3 more
- DAGs: `airflow/dags/lojban_m5_padded_nary_dag.py`

### M5.2

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M5.2`
- Legacy aliases: `M5.2.autoregressive_chain.run`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M5.2.autoregressive_chain.run telemetry report
- Scripts: `scripts/run_m5_2_autoregressive_chain.py`
- DAGs: `airflow/dags/lojban_m5_2_autoregressive_chain_dag.py`

### M5.3

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M5.3`
- Legacy aliases: `M5.3.masked_pair_chain.run`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M5.3.masked_pair_chain.run telemetry report
- Scripts: `scripts/run_m5_3_masked_pair_chain.py`
- DAGs: `airflow/dags/lojban_m5_3_masked_pair_chain_dag.py`

### M6

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M6.0, M6.1, M6.2, M6.3, M6.6`
- Legacy aliases: `RESULTS_M6_1_ALIGNMENT_70ACC, RESULTS_M6_2_ALIGNED_30ACC, RESULTS_M6_3_SCRATCHPAD_35ACC, RESULTS_M6_6_DIRECTED_AST_FINAL, RESULTS_M6_SEVERED_BRIDGE_20260314`
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Brief: M6 severed bridge + 4 more
- Scripts: `scripts/eval_m6_logic_engine.py, scripts/train_m6_logic_engine.py`

### M7

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M7.0`
- Legacy aliases: `M7, RESULTS_M7_INTERLEAVED_COPROCESSOR`
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Brief: M7 interleaved coprocessor
- Scripts: `scripts/eval_m7_interleaved.py, scripts/train_m7_interleaved.py`

### M8

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M8.0`
- Legacy aliases: `M8, RESULTS_M8_COUNCIL_OF_ORACLES`
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Brief: M8 council of oracles
- Scripts: `scripts/eval_m8_council.py, scripts/train_m8_council.py`

### M9

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M9.0, M9.1`
- Legacy aliases: `M9.audit, M9.hypercube, RESULTS_M9_AUDIT, RESULTS_M9_HYPERCUBE`
- Entry count: `2`
- Runnable rows: `0`
- Artifact-only rows: `2`
- Brief: M9 duel hypercube + 1 more
- Scripts: `scripts/m9/eval_m9.py`

### M10

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M10.0, M10.1, M10.2, M10.3`
- Legacy aliases: `M10.audit, M10.final_bridge, M10.floor_lock, M10.publication, RESULTS_M10_AUDIT, final_bridge_audit, final_floor_lock, final_publication_metrics`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M10 audit + 3 more
- Scripts: `scripts/m10/final_audit.py`

### M14

- Layer: `manifold_and_return_path`
- Status: `runnable`
- Normalized IDs: `M14.A, M14.B, M14.C, M14.D, M14.E`
- Legacy aliases: `A, B, C, D, E`
- Entry count: `5`
- Runnable rows: `5`
- Brief: relaxed residual scratchpad + 4 more
- DAGs: `airflow/dags/lojban_m14_symbiote_scratchpad_dag.py`

### History

- Layer: `control_plane`
- Status: `runnable`
- Entry count: `0`
- Runnable rows: `1`
- Brief: Backfill and aggregate suite control plane
- Scripts: `scripts/run_ablation_history_backfill.py, scripts/render_ablation_history_catalog.py, scripts/run_m_bridge_ablation_test_suite.py, scripts/build_ablation_program_map.py`
- DAGs: `airflow/dags/lojban_ablation_history_backfill_dag.py, airflow/dags/lojban_m_bridge_ablation_test_suite_dag.py`

## Transition Spine

- `M1_to_M2`: `M1 -> M2` via `M1.5`
- `M2_to_M3`: `M2 -> M3` via `M2.1`
- `M3_to_M4`: `M3 -> M4` via `M3.8.C`
- `M4_to_M5`: `M4 -> M5` via `M4.2`
- `M5_to_M6`: `M5 -> M6` via `M5.3`
- `M6_to_M7`: `M6 -> M7` via `M6.3`
- `M7_to_M8`: `M7 -> M8` via `M7.0`
- `M8_to_M9`: `M8 -> M9` via `M8.0`
- `M9_to_M10`: `M9 -> M10` via `M9.0`
- `M10_to_M11`: `M10 -> M11` via `M10`
- `M11_to_M14`: `M11 -> M14` via `M11`
