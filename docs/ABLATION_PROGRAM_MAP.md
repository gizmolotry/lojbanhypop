# Ablation Program Map

- Generated UTC: `2026-05-25T17:19:11.878099+00:00`
- Source history manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/m22_mixed_ood_history_20260523/ablation_history_manifest.json`
- Concentrated family count: `38`

## Program Layers

- `legacy_orchestration`: letter-era experiments and their early DAG architecture
- `bridge_and_serialization`: early-to-mid M-series bridge, grounding, and serialization families
- `manifold_and_return_path`: later manifold/native/discriminative/re-entry families
- `dictionary_first_substrate`: M20 dictionary-first substrate branch
- `dynamic_bridi_substrate`: M21 dynamic bridi substrate branch
- `semantic_generalization_substrate`: M22 semantic coverage generalization gate
- `causal_relevance_substrate`: M23 causal relevance-router fork
- `substrate_compression`: M24 substrate compression branch
- `control_plane`: the backfill, catalog, and aggregate-suite layer

## Concentrated Families

### A-G

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `a_to_g.a, a_to_g.b_1, a_to_g.b_2, a_to_g.c, a_to_g.d, a_to_g.e, legacy.core.a, legacy.core.b, legacy.core.c, legacy.core.e, legacy.core.f, legacy.core.g, legacy.duel.english_cot, legacy.duel.lojban_topology`
- Legacy aliases: `A, A-G/A, A-G/B.1, A-G/B.2, A-G/C, A-G/D, A-G/E, B, B.1, B.2, C, Control Duel/English, Control Duel/Lojban, Core/A, Core/B, Core/C, Core/E, Core/F, Core/G, D, E, English CoT, F, G`
- Entry count: `14`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Doc-only rows: `9`
- Brief: early benchmark matrix covering base control, projected handoff, coconut variants, and English-vs-Lojban control comparisons.
- Family groups: `a_to_g_matrix, control_duel, core_matrix`
- Docs: `docs/history/reports/AUDIT_REPORT.md, docs/ledger/CANONICAL_LEDGER.md`
- Scripts: `scripts/legacy/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/control_plane/lojban_ablation_matrix_dag.py`
- Artifact roots: `docs/ledger/CANONICAL_LEDGER.md, runs/ablation/a_to_g/20260305_033123, runs/ablation/a_to_g/20260305_033123/ablation_matrix.json, runs/ablation/a_to_g/20260305_190131, runs/ablation/a_to_g/20260305_190131/ablation_matrix.json`

### H

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `h.series.h3, legacy.h.h1, legacy.h.h2, legacy.h.h3, legacy.h.h4`
- Legacy aliases: `H1, H2, H3, H4`
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Doc-only rows: `4`
- Brief: mid-layer bridge experiments testing linear and SwiGLU geometric handoff into the host decoder.
- Family groups: `h_series`
- Docs: `docs/history/reports/NUMERICAL_AUDIT.md, docs/ledger/CANONICAL_LEDGER.md`
- Scripts: `scripts/legacy/true_coconut.py`
- Artifact roots: `docs/ledger/CANONICAL_LEDGER.md, runs/h_series/20260228_190640, runs/h_series/20260228_190640/run_h_series.json`

### H5

- Layer: `legacy_orchestration`
- Status: `artifact_only`
- Normalized IDs: `h.series.h5_dptr, h.series.h5_ood, h.series.h5_prov, legacy.h5.h5_2a, legacy.h5.h5_2b, legacy.h5.h5_4, legacy.h5.h5_5`
- Legacy aliases: `Gearbox Control, H5-DPTR, H5-OOD, H5-PROV, H5.2a, H5.2b, H5.4, H5.5, Iron Collar, True Neuro-Symbolic`
- Entry count: `7`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Doc-only rows: `4`
- Brief: boolean-surgery, persistent-advisor, and bridge-extension experiments that fed into the later J and L stacks.
- Family groups: `h5_bridge`
- Docs: `docs/history/reports/H5_ABLATION_REPORT.md, docs/history/reports/H5_SUMMARY.md`
- Scripts: `archive/legacy_scaffolding/trace_h5_provenance.py, scripts/legacy/eval_h5_dynamic_pointer_refactor.py, scripts/legacy/eval_h5_ood_stress.py, scripts/legacy/train_h5_persistent_vq_advisor.py`
- Artifact roots: `docs/ledger/CANONICAL_LEDGER.md, runs/h_series/20260303_175314, runs/h_series/20260303_175314/run_h_series.json, runs/h_series/20260303_175413, runs/h_series/20260303_175413/run_h_series.json, runs/h_series/20260303_175526, runs/h_series/20260303_175526/run_h_series.json, runs/h_series/20260303_180224`

### J

- Layer: `legacy_orchestration`
- Status: `runnable`
- Normalized IDs: `M1.1, M1.2, M1.3, M1.4, M1.5`
- Legacy aliases: `J-1, J-2, J-3, J-4, J-5, M1.1, M1.2, M1.3, M1.4, M1.5`
- Entry count: `5`
- Runnable rows: `5`
- Brief: advisor-side data generation and adversarial synthesis family that seeded the later numeric M-line.
- Family groups: `j_series`
- Docs: `docs/SERIES_CHARTER.md, docs/ledger/CANONICAL_LEDGER.md`
- Scripts: `archive/legacy_scaffolding/eval_j_1.py, archive/legacy_scaffolding/eval_j_2.py, archive/legacy_scaffolding/eval_j_3.py, archive/legacy_scaffolding/eval_j_4.py, archive/legacy_scaffolding/eval_j_5.py, scripts/legacy/train_h5_persistent_vq_advisor.py`
- DAGs: `airflow/dags/legacy/lojban_j_series_dag.py`
- Artifact roots: `runs/j_series/20260304_001943, runs/j_series/20260304_001943/run_h_series.json, runs/j_series/20260304_002346, runs/j_series/20260304_002346/run_h_series.json, runs/j_series/20260304_002412, runs/j_series/20260304_002412/run_h_series.json, runs/j_series/20260304_050706, runs/j_series/20260304_050706/j-4.json`

### L

- Layer: `legacy_orchestration`
- Status: `partially_runnable`
- Normalized IDs: `M2.1, M2.2, M2.3, M3.0, M3.1, M3.2, M3.3, M3.4, l.branch.m3_5.m3_5_a, l.branch.m3_5.m3_5_b, l.branch.m3_5.m3_5_c, l.branch.m3_6.m3_6_a, l.branch.m3_6.m3_6_b, l.branch.m3_6.m3_6_c, l.branch.m3_7.m3_7_a, l.branch.m3_7.m3_7_b, l.branch.m3_7.m3_7_c, l.branch.m3_8.m3_8_a, l.branch.m3_8.m3_8_b, l.branch.m3_8.m3_8_c, l.series.charter`
- Legacy aliases: `L-Series, L6-A, L6-B, L6-C, Lagrangian Series, M2.1, M2.2, M2.3, M2.A, M2.B, M2.C, M3.0, M3.1, M3.2, M3.3, M3.4, M3.5.A, M3.5.B, M3.5.C, M3.6.A, M3.6.B, M3.6.C, M3.6.M3.6.A, M3.6.M3.6.B`
- Entry count: `21`
- Runnable rows: `3`
- Artifact-only rows: `17`
- Doc-only rows: `1`
- Brief: lagrangian constrained-manifold family and its branch lineages before the later M unification.
- Family groups: `l_series`
- Docs: `archive/reports/relevant/REPORTS_RELEVANT/l6_ablation_manifest.md, docs/SERIES_CHARTER.md`
- Scripts: `scripts/legacy/run_l6_ablation_branch.py, scripts/legacy/train_l_series_mvs.py`
- DAGs: `airflow/dags/legacy/lojban_l_series_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry/manual_20260305/m3_5_symmetry_20260305_175506, artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry/manual_20260305/m3_5_symmetry_20260305_175506/m3_5_symmetry_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry/manual_20260305/m3_5_symmetry_20260305_181349, artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry/manual_20260305/m3_5_symmetry_20260305_181349/m3_5_symmetry_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle/m3_6_1_20260306, artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle/m3_6_1_20260306/m3_6_symmetry_oracle_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle/m3_6_1_20260306_rerun, artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle/m3_6_1_20260306_rerun/m3_6_symmetry_oracle_report.json`

### J/L Hypercube

- Layer: `legacy_orchestration`
- Status: `mixed_historical`
- Entry count: `0`
- Runnable rows: `0`
- Brief: cross-family orchestration and hypercube reporting layer that consolidated J/L-era runs before the modern M suite.
- Docs: `archive/reports/relevant/REPORTS_RELEVANT/ablation_hypercube_report.md, archive/results/legacy_misc/20260305/RESULTS_FULL_GRID_20260305/ablation_hypercube_report.md`
- DAGs: `airflow/dags/control_plane/lojban_ablation_hypercube_report_dag.py`

### Phase Eval

- Layer: `legacy_orchestration`
- Status: `runnable`
- Normalized IDs: `phase5.objective.ablate_compositional_consistency_loss, phase5.objective.ablate_compression_regularization_loss, phase5.objective.ablate_coverage_regularization_loss, phase5.objective.ablate_roundtrip_consistency_loss, phase5.objective.ablate_semantic_unambiguity_loss, phase5.objective.baseline_no_phase5, phase5.objective.phase5_full, phase5.train.ablate_compositional_consistency_weight, phase5.train.ablate_compression_regularization_weight, phase5.train.ablate_coverage_regularization_weight, phase5.train.ablate_roundtrip_consistency_weight, phase5.train.ablate_semantic_unambiguity_weight, phase5.train.baseline_no_phase5, phase5.train.phase5_full`
- Legacy aliases: `ablate_compositional_consistency_loss, ablate_compositional_consistency_weight, ablate_compression_regularization_loss, ablate_compression_regularization_weight, ablate_coverage_regularization_loss, ablate_coverage_regularization_weight, ablate_roundtrip_consistency_loss, ablate_roundtrip_consistency_weight, ablate_semantic_unambiguity_loss, ablate_semantic_unambiguity_weight, baseline_no_phase5, phase5_full`
- Entry count: `14`
- Runnable rows: `14`
- Brief: phase-5 train/objective ablations used to stress semantic and compression loss surfaces before later M-series serialization work.
- Family groups: `phase5_objective_ablation, phase5_train_ablation`
- Docs: `docs/SERIES_CHARTER.md`
- Scripts: `scripts/legacy/run_phase5_objective_ablation.py, scripts/legacy/run_phase5_train_ablation.py`
- DAGs: `airflow/dags/control_plane/lojban_phase_ablation_dag.py`
- Artifact roots: `src/runs, src/runs/phase5_objective_ablation.json, src/runs/phase5_train_ablation/20260222_162211, src/runs/phase5_train_ablation/20260222_162211/ablation_manifest.json, src/runs/phase5_train_ablation/20260222_162502, src/runs/phase5_train_ablation/20260222_162502/ablation_manifest.json, src/runs/phase5_train_ablation/20260222_162541, src/runs/phase5_train_ablation/20260222_162541/ablation_manifest.json`

### M3.9

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.9`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.9 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_9_primitive_probe.py`
- DAGs: `airflow/dags/m3/lojban_m3_9_primitive_probe_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/m3_9_20260306, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/m3_9_20260306/m3_9_primitive_probe_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/m3_9_20260306_r2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/m3_9_20260306_r2/m3_9_primitive_probe_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/refactor_clean/m3_9_20260310_refactor, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/refactor_clean/m3_9_20260310_refactor/m3_9_primitive_probe_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/refactor_clean_v2/m3_9_20260310_refactor_v2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe/refactor_clean_v2/m3_9_20260310_refactor_v2/m3_9_primitive_probe_report.json`

### M3.10

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.10`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.10 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_10_ood_accuracy_probe.py`
- DAGs: `airflow/dags/m3/lojban_m3_10_ood_accuracy_probe_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy/m3_10_20260307_r2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy/m3_10_20260307_r2/m3_10_ood_accuracy_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy/refactor_clean_small/m3_10_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy/refactor_clean_small/m3_10_20260310_refactor_small/m3_10_ood_accuracy_report.json`

### M3.11

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.11`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M3.11 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_11_winograd_failure_anatomy.py`
- DAGs: `airflow/dags/m3/lojban_m3_11_winograd_failure_anatomy_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/m3_11_20260307, artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/m3_11_20260307/m3_11_winograd_failure_anatomy_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/refactor_clean_small/m3_11_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/refactor_clean_small/m3_11_20260310_refactor_small/m3_11_winograd_failure_anatomy_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/refactor_clean_v2/m3_11_20260310_refactor_v2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy/refactor_clean_v2/m3_11_20260310_refactor_v2/m3_11_winograd_failure_anatomy_report.json`

### M3.12

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.12.A, M3.12.B, M3.12.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.12 A + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_12_geometric_return_stream.py`
- DAGs: `airflow/dags/m3/lojban_m3_12_geometric_return_stream_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_12_geometric_return_stream/refactor_clean_small/m3_12_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_12_geometric_return_stream/refactor_clean_small/m3_12_20260310_refactor_small/m3_12_return_stream_report.json`

### M3.13

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.13.A, M3.13.B, M3.13.C, M3.13.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.13 A + 3 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_13_geometric_ablation_grid.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_13_relational_grid/refactor_clean_small/m3_13_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_13_relational_grid/refactor_clean_small/m3_13_20260310_refactor_small/m3_13_report.json`

### M3.14

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.14.A, M3.14.B, M3.14.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.14 A + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_14_structural_alignment_bridge.py`
- DAGs: `airflow/dags/m3/lojban_m3_14_structural_alignment_bridge_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/m3_14_prelim_20260309, artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/m3_14_prelim_20260309/m3_14_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/m3_14_prelim_guarded_20260309, artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/m3_14_prelim_guarded_20260309/m3_14_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/refactor_clean_small/m3_14_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_14_structural_alignment_bridge/refactor_clean_small/m3_14_20260310_refactor_small/m3_14_report.json`

### M3.15

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15.A, M3.15.B, M3.15.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15 A + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_15_rotary_coconut.py`
- DAGs: `airflow/dags/m3/lojban_m3_15_rotary_coconut_dag.py, airflow/dags/m3/lojban_m3_15_rotary_coconut_seven_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut/refactor_clean_small/m3_15_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut/refactor_clean_small/m3_15_20260310_refactor_small/m3_15_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R1, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R1/m3_15_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R2/m3_15_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R3, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R3/m3_15_report.json`

### M3.15b

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15b.A, M3.15b.B, M3.15b.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15b A + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_15b_relation_local_rotary.py`
- DAGs: `airflow/dags/m3/lojban_m3_15b_relation_local_rotary_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/bridge_base_v2/m3_15b_20260311_bridge_base_v2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/bridge_base_v2/m3_15b_20260311_bridge_base_v2/m3_15b_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/decollapse_followup/m3_15b_20260311_after_decollapse, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/decollapse_followup/m3_15b_20260311_after_decollapse/m3_15b_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/refactor_clean_small/m3_15b_20260310_refactor_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/refactor_clean_small/m3_15b_20260310_refactor_small/m3_15b_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/refactor_clean_small/m3_15b_20260311_probe_v2, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15b_relation_local_rotary/refactor_clean_small/m3_15b_20260311_probe_v2/m3_15b_report.json`

### M3.15c

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15c.A, M3.15c.B, M3.15c.C`
- Legacy aliases: `A, B, C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: M3.15c A + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_15c_family_conditioned_bridge.py`
- DAGs: `airflow/dags/m3/lojban_m3_15c_family_conditioned_bridge_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15c_family_conditioned_bridge/bridge_base_v1/m3_15c_20260311_bridge_base_v1, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15c_family_conditioned_bridge/bridge_base_v1/m3_15c_20260311_bridge_base_v1/m3_15c_report.json`

### M3.15d

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.15d.A, M3.15d.B, M3.15d.C, M3.15d.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.15d A + 3 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_15d_answer_path_forcing.py`
- DAGs: `airflow/dags/m3/lojban_m3_15d_answer_path_forcing_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing/bridge_base_v1/m3_15d_20260311_bridge_base_v1, artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing/bridge_base_v1/m3_15d_20260311_bridge_base_v1/m3_15d_report.json`

### M3.16

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.16.A, M3.16.B, M3.16.C, M3.16.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M3.16 A + 3 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_16_continuous_graph_bias.py`
- DAGs: `airflow/dags/m3/lojban_m3_16_continuous_graph_bias_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias/bridge_base_v1_small/m3_16_20260311_bridge_base_v1_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias/bridge_base_v1_small/m3_16_20260311_bridge_base_v1_small/m3_16_report.json`

### M3.17

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M3.17.A, M3.17.B, M3.17.C, M3.17.D`
- Legacy aliases: `A, B, C, D`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: control no re-entry + 3 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_17_advisor_reentry_bridge.py`
- DAGs: `airflow/dags/m3/lojban_m3_17_advisor_reentry_bridge_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge/bridge_base_v1_small/m3_17_20260328_bridge_base_v1_small, artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge/bridge_base_v1_small/m3_17_20260328_bridge_base_v1_small/m3_17_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge/m3_17_contract_smoke_20260328, artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge/m3_17_contract_smoke_20260328/m3_17_report.json`

### M3.18

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.18.A, M3.18.B, M3.18.C, M3.18.D, M3.18.E`
- Legacy aliases: `A, B, C, D, E`
- Entry count: `5`
- Runnable rows: `5`
- Brief: control no advisor + 4 more
- Family groups: `m_track`
- Scripts: `scripts/m3/run_m3_18_decoder_reentry_resume.py`
- DAGs: `airflow/dags/m3/lojban_m3_18_decoder_reentry_resume_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_d_sft_smoke_20260328, artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_d_sft_smoke_20260328/m3_18_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_first_train_test_20260328, artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_first_train_test_20260328/m3_18_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_resume_smoke_small_20260328, artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_resume_smoke_small_20260328/m3_18_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329/D0/m3_19_grid_smoke_v2_20260329_d0, artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329/D0/m3_19_grid_smoke_v2_20260329_d0/m3_18_report.json`

### M3.19

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M3.19.D0, M3.19.D1, M3.19.D2, M3.19.D3`
- Legacy aliases: `D0, D1, D2, D3`
- Entry count: `4`
- Runnable rows: `4`
- Brief: M3.19 D0 + 3 more
- Family groups: `m_track`
- DAGs: `airflow/dags/m3/lojban_m3_19_d_mainline_grid_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329, artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329/m3_19_grid_report.json`

### M4.0

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M4.0`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M4.0 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m4/run_m4_0_semantic_probe.py`
- DAGs: `airflow/dags/m4/lojban_m4_0_semantic_probe_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m4_0_semantic_probe/bridge_base_v1/m4_0_20260312_bridge_base_v1, artifacts/runs/telemetry/raw/ablation/hypercube/m4_0_semantic_probe/bridge_base_v1/m4_0_20260312_bridge_base_v1/m4_0_semantic_probe_report.json`

### M4.2

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M4.2`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M4.2 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m4/run_m4_2_predicate_grounding.py`
- DAGs: `airflow/dags/m4/lojban_m4_2_predicate_grounding_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding/bridge_base_v1/m4_2_20260312_bridge_base_v1, artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding/bridge_base_v1/m4_2_20260312_bridge_base_v1/m4_2_predicate_grounding_report.json`

### M5

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M5.0.A, M5.0.B, M5.0.C`
- Legacy aliases: `M5.A, M5.B, M5.C`
- Entry count: `3`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Brief: Reuse-oriented control + 2 more
- Family groups: `m_track`
- Scripts: `scripts/m5/run_m5_autoformalization.py`
- DAGs: `airflow/dags/m5/lojban_m5_autoformalization_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260313_000102, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260313_000102/m5_autoformalization_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_034252, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_034252/m5_autoformalization_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_040549, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_040549/m5_autoformalization_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_052704, artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization/m5_20260330_052704/m5_autoformalization_report.json`

### M5.1

- Layer: `bridge_and_serialization`
- Status: `artifact_only`
- Normalized IDs: `M5.1.N0, M5.1.N1, M5.1.N2, M5.1.N3`
- Legacy aliases: `M5.N0, M5.N1, M5.N2, M5.N3`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: Add counterfactual invariance + 3 more
- Family groups: `m_track`
- DAGs: `airflow/dags/m5/lojban_m5_padded_nary_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013336, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013336/m5_padded_nary_family_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013447, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013447/m5_padded_nary_family_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013929, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_013929/m5_padded_nary_family_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_014528, artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary/m5_padded_nary_20260313_014528/m5_padded_nary_family_report.json`

### M5.2

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M5.2`
- Legacy aliases: `M5.2.autoregressive_chain.run`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M5.2.autoregressive_chain.run telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m5/run_m5_2_autoregressive_chain.py`
- DAGs: `airflow/dags/m5/lojban_m5_2_autoregressive_chain_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain/m5_2_20260313_133612, artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain/m5_2_20260313_133612/m5_2_autoregressive_chain_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain/m5_2_20260330_034252, artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain/m5_2_20260330_034252/m5_2_autoregressive_chain_report.json`

### M5.3

- Layer: `bridge_and_serialization`
- Status: `runnable`
- Normalized IDs: `M5.3`
- Legacy aliases: `M5.3.masked_pair_chain.run`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M5.3.masked_pair_chain.run telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m5/run_m5_3_masked_pair_chain.py`
- DAGs: `airflow/dags/m5/lojban_m5_3_masked_pair_chain_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_151121, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_151121/m5_3_masked_pair_chain_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_155758, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_155758/m5_3_masked_pair_chain_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_155835, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260313_155835/m5_3_masked_pair_chain_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260330_034252, artifacts/runs/telemetry/raw/ablation/hypercube/m5_3_masked_pair_chain/m5_3_20260330_034252/m5_3_masked_pair_chain_report.json`

### M6

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M6.0, M6.1, M6.2, M6.3, M6.6`
- Legacy aliases: `RESULTS_M6_1_ALIGNMENT_70ACC, RESULTS_M6_2_ALIGNED_30ACC, RESULTS_M6_3_SCRATCHPAD_35ACC, RESULTS_M6_6_DIRECTED_AST_FINAL, RESULTS_M6_SEVERED_BRIDGE_20260314`
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Brief: M6 severed bridge + 4 more
- Family groups: `m_track`
- Scripts: `scripts/m6/eval_m6_logic_engine.py, scripts/m6/train_m6_logic_engine.py`
- Artifact roots: `archive/results/m6/20260314/RESULTS_M6_SEVERED_BRIDGE_20260314, archive/results/m6/20260314/RESULTS_M6_SEVERED_BRIDGE_20260314/m6_eval_report.json, archive/results/m6_1/active/RESULTS_M6_1_ALIGNMENT_70ACC, archive/results/m6_1/active/RESULTS_M6_1_ALIGNMENT_70ACC/m6_eval_report.json, archive/results/m6_2/active/RESULTS_M6_2_ALIGNED_30ACC, archive/results/m6_2/active/RESULTS_M6_2_ALIGNED_30ACC/m6_eval_report.json, archive/results/m6_3/active/RESULTS_M6_3_SCRATCHPAD_35ACC, archive/results/m6_3/active/RESULTS_M6_3_SCRATCHPAD_35ACC/m6_directed_eval_report.json`

### M7

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M7.0`
- Legacy aliases: `M7, RESULTS_M7_INTERLEAVED_COPROCESSOR`
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Brief: M7 interleaved coprocessor
- Family groups: `m_track`
- Scripts: `scripts/m7/eval_m7_interleaved.py, scripts/m7/train_m7_interleaved.py`
- Artifact roots: `archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR, archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR/m7_eval_report.json`

### M8

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M8.0`
- Legacy aliases: `M8, RESULTS_M8_COUNCIL_OF_ORACLES`
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Brief: M8 council of oracles
- Family groups: `m_track`
- Scripts: `scripts/m8/eval_m8_council.py, scripts/m8/train_m8_council.py`
- Artifact roots: `archive/results/m8/active/RESULTS_M8_COUNCIL_OF_ORACLES, archive/results/m8/active/RESULTS_M8_COUNCIL_OF_ORACLES/m8_eval_report.json`

### M9

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M9.0, M9.1`
- Legacy aliases: `M9.audit, M9.hypercube, RESULTS_M9_AUDIT, RESULTS_M9_HYPERCUBE`
- Entry count: `2`
- Runnable rows: `0`
- Artifact-only rows: `2`
- Brief: M9 duel hypercube + 1 more
- Family groups: `m_track`
- Scripts: `scripts/m9/eval_m9.py`
- Artifact roots: `archive/results/m9/active/RESULTS_M9_AUDIT, archive/results/m9/active/RESULTS_M9_AUDIT/m9_audit_report.json, archive/results/m9/active/RESULTS_M9_HYPERCUBE, archive/results/m9/active/RESULTS_M9_HYPERCUBE/duel_report.json`

### M10

- Layer: `manifold_and_return_path`
- Status: `artifact_only`
- Normalized IDs: `M10.0, M10.1, M10.2, M10.3`
- Legacy aliases: `M10.audit, M10.final_bridge, M10.floor_lock, M10.publication, RESULTS_M10_AUDIT, final_bridge_audit, final_floor_lock, final_publication_metrics`
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Brief: M10 audit + 3 more
- Family groups: `m_track`
- Scripts: `scripts/m10/final_audit.py`
- Artifact roots: `archive/results/m10/active/RESULTS_M10_AUDIT, archive/results/m10/active/RESULTS_M10_AUDIT/m10_audit_report.json, archive/results/m10/active/RESULTS_M10_FINAL_AUDIT, archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_bridge_audit.json, archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_floor_lock.json, archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_publication_metrics.json`

### M18

- Layer: `manifold_and_return_path`
- Status: `runnable`
- Normalized IDs: `M18, M18.1.A, M18.1.B, M18.1.C, M18.1.D, M18.1.E, M18.1.F, M18.1.G, M18.2.A, M18.2.B, M18.2.C, M18.2.D, M18.2.E, M18.2.F`
- Legacy aliases: `EN-CONCISE, EN-COT, EN-COT+KILL-RANDOM, EN-COT+L-TYPED, EN-COT+U-TYPED, KILL-LABEL, KILL-RANDOM, L-TYPED, M18.harmonized_audit.EN-CONCISE, M18.harmonized_audit.EN-COT, M18.harmonized_audit.KILL-LABEL, M18.harmonized_audit.KILL-RANDOM, M18.harmonized_audit.L-TYPED, M18.harmonized_audit.U-TYPED, M18.harmonized_audit.ZH-COT, M18.hybrid_cot_audit.EN-CONCISE, M18.hybrid_cot_audit.EN-COT, M18.hybrid_cot_audit.EN-COT+KILL-RANDOM, M18.hybrid_cot_audit.EN-COT+L-TYPED, M18.hybrid_cot_audit.EN-COT+U-TYPED, M18.hybrid_cot_audit.ZH-COT, U-TYPED, ZH-COT`
- Entry count: `14`
- Runnable rows: `14`
- Brief: M18 EN-CONCISE + 10 more
- Family groups: `m_track`
- Scripts: `scripts/m18/run_harmonized_audit.py, scripts/m18/run_hybrid_cot_audit.py, scripts/m18/run_m18_controller_family.py, scripts/m18/run_sapir_whorf_audit.py`
- DAGs: `airflow/dags/m18/lojban_m18_controller_family_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409, artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409/harmonized_audit_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409/hybrid_cot_audit_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409/m18_family_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m18_controller_family/m18_frontier_audits_20260409/sapir_whorf_audit_report.json`

### M19

- Layer: `manifold_and_return_path`
- Status: `runnable`
- Normalized IDs: `M19, M19.3.A, M19.3.B, M19.3.C, M19.3.D, M19.3.E, M19.3.F, M19.3.R11, M19.3.R12, M19.3.R21, M19.3.R22, M19.4, M19.31, M19.32`
- Legacy aliases: `A, B, C, D, E, F, R11, R12, R21, R22`
- Entry count: `14`
- Runnable rows: `14`
- Brief: 16Q / 64D / 12S + 13 more
- Family groups: `m_track`
- Scripts: `scripts/m19/run_isolation_grid.py, scripts/m19/run_m19_arity_causal_suite.py, scripts/m19/run_m19_bridge_channel_suite.py, scripts/m19/run_m19_godtier_benchmark.py, scripts/m19/run_m19_integrity_suite.py, scripts/m19/run_m19_kill_test_suite.py, scripts/m19/run_m19_mainline_suite.py, scripts/m19/run_m19_order_sensitivity_suite.py, scripts/m19/run_m19_pointer_counterfactual_suite.py, scripts/m19/run_m19_replication_suite.py, scripts/m19/run_m19_stability_microgrid.py`
- DAGs: `airflow/dags/m19/lojban_m19_family_dag.py, airflow/dags/m19/lojban_m19_isolation_grid_dag.py, airflow/dags/m19/lojban_m19_kill_test_suite_dag.py, airflow/dags/m19/lojban_m19_mainline_suite_dag.py, airflow/dags/m19/lojban_m19_paper_package_dag.py, airflow/dags/m19/lojban_m19_replication_suite_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/force_1, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/force_1/benchmark_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/force_8, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/force_8/benchmark_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/m19_arity_causal_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/no_mask, artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark/arity_causal/m19_31_arity_causal_promoted_e100_20260510/no_mask/benchmark_report.json`

### M20

- Layer: `dictionary_first_substrate`
- Status: `runnable`
- Normalized IDs: `M20.1, M20.1.A, M20.1.B, M20.1.C, M20.1.D, M20.1.E, M20.1.F`
- Legacy aliases: `A, B, C, D, E, F`
- Entry count: `7`
- Runnable rows: `7`
- Brief: Brivi-locked predicate formation + 6 more
- Family groups: `m_track`
- Scripts: `scripts/m20/run_m20_dictionary_first_suite.py, scripts/m20/run_m20_lock_suite.py, scripts/m20/run_m20_predicate_induction.py`
- DAGs: `airflow/dags/m20/lojban_m20_dictionary_first_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_branch_accuracy_20260514, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_branch_accuracy_20260514/m20_dictionary_first_suite_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_full_retrain_20260513, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_full_retrain_20260513/m20_dictionary_first_suite_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_retrain_20260513, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_retrain_20260513/m20_dictionary_first_suite_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_retrain_commitment_20260513, artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite/m20_1_six_lock_retrain_commitment_20260513/m20_dictionary_first_suite_report.json`

### M21

- Layer: `dynamic_bridi_substrate`
- Status: `runnable`
- Normalized IDs: `M21.1, M21.1.M, M21.1.A, M21.1.B, M21.1.C, M21.1.D, M21.1.E, M21.1.F, M21.1.G, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L, M21.1.N, M21.1.O, M21.1.P, M21.1.Q, M21.1.R, M21.1.S, M21.1.T`
- Legacy aliases: `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T`
- Entry count: `21`
- Runnable rows: `21`
- Brief: Actual bridge adapter + 20 more
- Family groups: `m_track`
- Scripts: `scripts/m21/run_m21_actual_bridge_suite.py, scripts/m21/run_m21_adversarial_audit.py, scripts/m21/run_m21_dynamic_bridi_suite.py, scripts/m21/run_m21_lock_suite.py, scripts/m21/run_m21_synthetic_assay_suite.py`
- DAGs: `airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_actual_bridge_full_20260515, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_actual_bridge_full_20260515/m21_actual_bridge_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_actual_bridge_vram_48e_20260515, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_actual_bridge_vram_48e_20260515/m21_actual_bridge_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_advtrain_H_actual_48e_20260518, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_advtrain_H_actual_48e_20260518/m21_actual_bridge_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_hyperbolic_tangent_actual_48e_20260517, artifacts/runs/telemetry/raw/ablation/hypercube/m21_actual_bridge_suite/m21_hyperbolic_tangent_actual_48e_20260517/m21_actual_bridge_report.json`

### M22

- Layer: `semantic_generalization_substrate`
- Status: `runnable`
- Normalized IDs: `M22`
- Entry count: `1`
- Runnable rows: `1`
- Brief: M22 telemetry report
- Family groups: `m_track`
- Scripts: `scripts/m22/run_m22_seed_stability_aggregate.py, scripts/m22/run_m22_semantic_generalization.py`
- DAGs: `airflow/dags/m22/lojban_m22_semantic_generalization_dag.py`
- Artifact roots: `artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_mixed_ood_stability_20260523, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_mixed_ood_stability_20260523/m22_seed_stability_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_stability_20260523, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_stability_20260523/m22_seed_stability_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_stability_hardened_20260523, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_s_96e_six_seed_stability_hardened_20260523/m22_seed_stability_report.json, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_t025_96e_six_seed_mixed_ood_stability_20260523, artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability/m22_blended_t025_96e_six_seed_mixed_ood_stability_20260523/m22_seed_stability_report.json`

### History

- Layer: `control_plane`
- Status: `runnable`
- Entry count: `0`
- Runnable rows: `1`
- Brief: Backfill and aggregate suite control plane
- Family groups: `control_plane`
- Docs: `docs/ABLATION_HISTORY_FULL.md, docs/ABLATION_PROGRAM_MAP.md, docs/ABLATION_PROGRAM_SPINE.md`
- Scripts: `scripts/control_plane/build_ablation_program_map.py, scripts/control_plane/build_ablation_program_spine.py, scripts/control_plane/render_ablation_history_catalog.py, scripts/control_plane/run_ablation_history_backfill.py, scripts/m_bridge/run_m_bridge_ablation_test_suite.py`
- DAGs: `airflow/dags/control_plane/lojban_ablation_history_backfill_dag.py, airflow/dags/control_plane/lojban_ablation_master_spine_dag.py, airflow/dags/control_plane/lojban_ablation_program_spine_dag.py, airflow/dags/m_bridge/lojban_m_bridge_ablation_test_suite_dag.py`

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
- `M14_to_M18`: `M14 -> M18` via `M14.D`
- `M18_to_M19`: `M18 -> M19` via `M18`
- `M20_to_M21`: `M20 -> M21` via `M20.1.F`
- `M21_to_M22`: `M21 -> M22` via `M21.1.O`
