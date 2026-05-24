# Whole Ablation Grid

- run_id: `m23_whole_grid_six_seed_20260524`
- generated: `2026-05-24T18:28:16.430549+00:00`
- history manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/m22_mixed_ood_history_20260523/ablation_history_manifest.json`
- program spine manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_spine/m23_spine_refresh_20260524/ablation_program_spine_manifest.json`

## Coverage

- stages: `26`
- fresh legacy surfaces: `6`
- artifact anchors: `18`
- history-only stages: `1`

## Legacy Grid Status

- legacy run_id: `legacy_grid_retrospective_20260413`
- `a_to_g`: `ok`
- `english_cot_duel`: `ok`
- `hj`: `ok`
- `l6`: `ok`
- `phase5_objective`: `ok`
- `phase5_train`: `ok`

## Stage Table

| stage | surface | counts | anchor | headline |
|---|---|---|---|---|
| `A-G` | `fresh_legacy_lane` | `e=14 r=0 a=5 d=9` | `artifacts/runs/telemetry/raw/ablation/a_to_g/legacy_grid/legacy_grid_retrospective_20260413/20260414_002554/ablation_matrix.json` | executed_runs=4.0000, control_base_final_acc=0.3333, coconut_handoff_final_acc=0.0000, nope_handoff_lift=0.0000 |
| `H` | `fresh_legacy_lane` | `e=5 r=0 a=1 d=4` | `artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid/legacy_grid_retrospective_20260413/20260414_003217/run_h_series.json` | executed_runs=12.0000, h1_handoff_lift=-0.3333, h5_ood_accuracy=0.4000, j1_schema_valid_rate=1.0000 |
| `H5` | `fresh_legacy_lane` | `e=7 r=0 a=3 d=4` | `artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid/legacy_grid_retrospective_20260413/20260414_003217/run_h_series.json` | executed_runs=12.0000, h1_handoff_lift=-0.3333, h5_ood_accuracy=0.4000, j1_schema_valid_rate=1.0000 |
| `J` | `fresh_legacy_lane` | `e=5 r=5 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid/legacy_grid_retrospective_20260413/20260414_003217/run_h_series.json` | executed_runs=12.0000, h1_handoff_lift=-0.3333, h5_ood_accuracy=0.4000, j1_schema_valid_rate=1.0000 |
| `L` | `fresh_legacy_lane` | `e=21 r=3 a=17 d=1` | `runs/l_series/l6_ablation/legacy_grid/legacy_grid_retrospective_20260413/20260414_004443/l6_ablation_manifest.json` | executed_rows=3.0000, mean_scope_constraint=0.3432, best_scope_constraint=0.3810 |
| `J/L Hypercube` | `history_only` | `e=0 r=0 a=0 d=0` | `` |  |
| `Phase Eval` | `fresh_legacy_lane` | `e=14 r=14 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid/legacy_grid_retrospective_20260413/phase5_objective_ablation.json` | full_total_regularizer=2.9140, dead_term_count=0.0000, dominant_term=0.0000, dominant_term_value=56.2299 |
| `M1` | `artifact_anchor` | `e=5 r=5 a=0 d=0` | `runs/j_series/test_run_true_coconut_d72f24907c594ce8bed4e1ff8345f686/20260523_120720/run_h_series.json` |  |
| `M2` | `artifact_anchor` | `e=3 r=3 a=0 d=0` | `runs/l_series/l6_ablation/legacy_grid/legacy_grid_retrospective_20260413/20260414_004443/l6_ablation_manifest.json` |  |
| `M3` | `artifact_anchor` | `e=48 r=12 a=36 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite/full_program_probe_20260413/m_bridge_ablation_suite_manifest.json` | bridge_track_count=3.0000, harmful_track_count=1.0000 |
| `M4` | `artifact_anchor` | `e=2 r=2 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding/bridge_base_v1/m4_2_20260312_bridge_base_v1/m4_2_predicate_grounding_report.json` |  |
| `M5` | `artifact_anchor` | `e=9 r=2 a=7 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_052704/m5_autoformalization_report.json` |  |
| `M6` | `artifact_anchor` | `e=5 r=0 a=5 d=0` | `archive/results/m6/20260314/RESULTS_M6_SEVERED_BRIDGE_20260314/m6_eval_report.json` |  |
| `M7` | `artifact_anchor` | `e=1 r=0 a=1 d=0` | `archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR/m7_eval_report.json` |  |
| `M8` | `artifact_anchor` | `e=1 r=0 a=1 d=0` | `archive/results/m8/active/RESULTS_M8_COUNCIL_OF_ORACLES/m8_eval_report.json` |  |
| `M9` | `artifact_anchor` | `e=2 r=0 a=2 d=0` | `archive/results/m9/active/RESULTS_M9_AUDIT/m9_audit_report.json` |  |
| `M10` | `artifact_anchor` | `e=4 r=0 a=4 d=0` | `archive/results/m10/active/RESULTS_M10_AUDIT/m10_audit_report.json` |  |
| `M11` | `artifact_anchor` | `e=0 r=0 a=0 d=0` | `archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json` | headline_accuracy=0.8592, headline_macro_f1=0.6287, bridge_audit_accuracy=0.8333, floor_lock_accuracy=0.7800 |
| `M14` | `artifact_anchor` | `e=0 r=0 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/m14_5_decompressor/m14_5_report.json` | cell_count=4.0000, best_cell_accuracy=0.0000, all_cells_zero=1.0000 |
| `M18` | `artifact_anchor` | `e=14 r=14 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409/m18_family_report.json` | sapir_english_accuracy=0.1000, sapir_chinese_accuracy=0.5000, harmonized_en_concise_accuracy=0.6000, harmonized_l_typed_accuracy=0.6000 |
| `M19` | `artifact_anchor` | `e=14 r=14 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m19_31_e2_promoted_package_20260506/direct_unified_eval_manifest.json` | mainline_overall_accuracy=0.6000, mainline_avg_tokens=30.8400, mainline_lift_vs_random=0.5700, mainline_audit_qformer_accuracy=0.6000 |
| `M20` | `artifact_anchor` | `e=7 r=7 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/direct_m20_whole_ledger_20260514/direct_unified_eval_manifest.json` | strict_accuracy=0.9998, synthetic_world_accuracy=0.9998, dictionary_coverage=0.9998, factorized_exact_accuracy=0.9998 |
| `M21` | `artifact_anchor` | `e=21 r=21 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m21_role_curriculum_mno_direct_48e_20260521/direct_unified_eval_manifest.json` | strict_accuracy=0.8494, bridi_trace_exact_accuracy=0.9996, gismu_accuracy=0.9999, cmavo_accuracy=0.9996 |
| `M22` | `artifact_anchor` | `e=1 r=1 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m22_blended_s_96e_confirm2_mixed_ood_direct_20260523/direct_unified_eval_manifest.json` | m22_promotion_candidate=0.0000, m22_promotion_gate_pass_rate=0.9444, strict_accuracy=0.8467, m22_candidate_cell_count=1.0000 |
| `M23` | `artifact_anchor` | `e=0 r=0 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m23_relevance_six_seed_direct_20260524/direct_unified_eval_manifest.json` | strict_accuracy=0.9387, decoy_relation_ood_accuracy=0.9663, worst_surface_accuracy=0.8296, bridi_trace_exact_accuracy=0.0361 |
| `Control Plane` | `control_plane_manifest` | `e=0 r=1 a=0 d=0` | `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/m22_mixed_ood_history_20260523/ablation_history_manifest.json` |  |

## Comparison Policy

### M1

- automatic compare-against: `M1, J, M1.1, M1.5`
- historical families carried forward: `J`
- required test contracts: `j.accept_rate_by_depth, j.accepted_foil_pair_accuracy, j.invariance_rate, j.schema_validity`

### M2

- automatic compare-against: `M2, M1, J, L, M2.1, M2.3`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M3

- automatic compare-against: `M3, M2, M1, J, L, M3.15d, M3.17, M3.18, M3.19`
- historical families carried forward: `J, L`
- required test contracts: `m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M4

- automatic compare-against: `M4, M3, M2, M1, J, L, M4.0, M4.2`
- historical families carried forward: `J, L`
- required test contracts: `m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M5

- automatic compare-against: `M5, M4, M3, M2, M1, J, L, M5.1, M5.2, M5.3`
- historical families carried forward: `J, L`
- required test contracts: `m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M6

- automatic compare-against: `M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M7

- automatic compare-against: `M7, M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M8

- automatic compare-against: `M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M9

- automatic compare-against: `M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M10

- automatic compare-against: `M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M11

- automatic compare-against: `M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- historical families carried forward: `J, L`
- required test contracts: `m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M14

- automatic compare-against: `M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M3.18.D, M3.19.D0, M14.C`
- historical families carried forward: `J, L`
- required test contracts: `m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M18

- automatic compare-against: `M18, M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M14.D`
- historical families carried forward: `J, L`
- required test contracts: `m18.kill_random_gap, m18.language_tax_compactness, m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M19

- automatic compare-against: `M19, M18, M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M19.3.A, M19.3.B, M19.31, M19.32`
- historical families carried forward: `J, L`
- required test contracts: `m19.runway_efficiency, m19.zh_branch_control, m19.dynamic_pacing_guardrails, m19.integrity_controls, m19.replication_stability, m19.kill_test_suite, m19.typed_faithfulness, m19.hyperbolic_geometry, m18.kill_random_gap, m18.language_tax_compactness, m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`

### M20

- automatic compare-against: `M20, M19, M19.31, M20.1.A, M20.1.B, M20.1.C, M20.1.D, M20.1.E, M20.1.F`
- historical families carried forward: `M19`
- required test contracts: `m20.dictionary_precedence, m20.factorized_predicate_dictionary, m20.counterfactual_quotient, m20.brivi_lock, m20.synthetic_world_pretraining, m20.soft_dictionary_annealing`

### M21

- automatic compare-against: `M21, M20, M19, M19.31, M20.1.F, M21.1.A, M21.1.B, M21.1.C, M21.1.D, M21.1.E, M21.1.F, M21.1.G, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L`
- historical families carried forward: `M19, M20`
- required test contracts: `m21.dynamic_frame_count, m21.bridi_reconstruction, m21.cmavo_causality, m21.judri_binding, m21.judri_gated_bridge, m21.pointer_necessity, m21.m19_gauntlet_port, m21.frame_necessity, m21.actual_bridge_transfer, m21.adversarial_heldout, m21.adversarial_augmentation, m21.semantic_coverage`

### M22

- automatic compare-against: `M22, M21, M20, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L, M21.1.M, M21.1.N, M21.1.O, M21.1.P, M21.1.Q, M21.1.R, M21.1.S, M21.1.T`
- historical families carried forward: `M21`
- required test contracts: `m22.semantic_coverage_generalization`

### M23

- automatic compare-against: `M23, M22, M21, M20, M21.1.S, M21.1.T, M23.A, M23.B`
- historical families carried forward: `M22`
- required test contracts: `m23.causal_relevance_router`


## Read

- The fresh part of the whole grid is now the recovered legacy runnable surface: A-G, H/H5/J, L6, and the phase-eval lanes under one manifest.
- The modern M rows are represented through artifact-backed anchors and the control-plane lineage manifests, so the whole program is visible without pretending every stage was freshly retrained.
- M3 remains the generative bridge archaeology block, M11 the discriminative oracle, M18 the controller-era comparison family, M19 the bounded runway mainline, M20 the dictionary-first substrate branch, M21 the dynamic bridi substrate branch, M22 the semantic-coverage generalization gate, and M23 the causal relevance-router fork.
