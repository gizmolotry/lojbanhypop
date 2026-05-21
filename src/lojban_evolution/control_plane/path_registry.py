from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from .artifact_io import latest_named_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_history_backfill"

# Legacy runnable surfaces moved into family/control-plane folders over time.
# Keep aliases centralized so DAGs, ledgers, and report renderers agree.
PATH_CANONICALIZATION: dict[str, str] = {
    'airflow/dags/lojban_ablation_history_backfill_dag.py': 'airflow/dags/control_plane/lojban_ablation_history_backfill_dag.py',
    'airflow/dags/lojban_ablation_hypercube_report_dag.py': 'airflow/dags/control_plane/lojban_ablation_hypercube_report_dag.py',
    'airflow/dags/lojban_ablation_master_spine_dag.py': 'airflow/dags/control_plane/lojban_ablation_master_spine_dag.py',
    'airflow/dags/lojban_ablation_matrix_dag.py': 'airflow/dags/control_plane/lojban_ablation_matrix_dag.py',
    'airflow/dags/lojban_ablation_program_spine_dag.py': 'airflow/dags/control_plane/lojban_ablation_program_spine_dag.py',
    'airflow/dags/lojban_experiment_dag.py': 'airflow/dags/control_plane/lojban_experiment_dag.py',
    'airflow/dags/lojban_j_series_dag.py': 'airflow/dags/legacy/lojban_j_series_dag.py',
    'airflow/dags/lojban_l_series_dag.py': 'airflow/dags/legacy/lojban_l_series_dag.py',
    'airflow/dags/lojban_m11_discriminative_suite_dag.py': 'airflow/dags/m11/lojban_m11_discriminative_suite_dag.py',
    'airflow/dags/lojban_m14_5_decompressor_dag.py': 'airflow/dags/m14/lojban_m14_5_decompressor_dag.py',
    'airflow/dags/lojban_m14_symbiote_scratchpad_dag.py': 'airflow/dags/m14/lojban_m14_symbiote_scratchpad_dag.py',
    'airflow/dags/lojban_m18_controller_family_dag.py': 'airflow/dags/m18/lojban_m18_controller_family_dag.py',
    'airflow/dags/lojban_m19_family_dag.py': 'airflow/dags/m19/lojban_m19_family_dag.py',
    'airflow/dags/lojban_m19_isolation_grid_dag.py': 'airflow/dags/m19/lojban_m19_isolation_grid_dag.py',
    'airflow/dags/lojban_m19_mainline_suite_dag.py': 'airflow/dags/m19/lojban_m19_mainline_suite_dag.py',
    'airflow/dags/lojban_m20_dictionary_first_dag.py': 'airflow/dags/m20/lojban_m20_dictionary_first_dag.py',
    'airflow/dags/lojban_m21_dynamic_bridi_dag.py': 'airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py',
    'airflow/dags/lojban_m3_10_ood_accuracy_probe_dag.py': 'airflow/dags/m3/lojban_m3_10_ood_accuracy_probe_dag.py',
    'airflow/dags/lojban_m3_11_winograd_failure_anatomy_dag.py': 'airflow/dags/m3/lojban_m3_11_winograd_failure_anatomy_dag.py',
    'airflow/dags/lojban_m3_12_geometric_return_stream_dag.py': 'airflow/dags/m3/lojban_m3_12_geometric_return_stream_dag.py',
    'airflow/dags/lojban_m3_14_structural_alignment_bridge_dag.py': 'airflow/dags/m3/lojban_m3_14_structural_alignment_bridge_dag.py',
    'airflow/dags/lojban_m3_15_rotary_coconut_dag.py': 'airflow/dags/m3/lojban_m3_15_rotary_coconut_dag.py',
    'airflow/dags/lojban_m3_15_rotary_coconut_seven_dag.py': 'airflow/dags/m3/lojban_m3_15_rotary_coconut_seven_dag.py',
    'airflow/dags/lojban_m3_15b_relation_local_rotary_dag.py': 'airflow/dags/m3/lojban_m3_15b_relation_local_rotary_dag.py',
    'airflow/dags/lojban_m3_15c_family_conditioned_bridge_dag.py': 'airflow/dags/m3/lojban_m3_15c_family_conditioned_bridge_dag.py',
    'airflow/dags/lojban_m3_15d_answer_path_forcing_dag.py': 'airflow/dags/m3/lojban_m3_15d_answer_path_forcing_dag.py',
    'airflow/dags/lojban_m3_16_continuous_graph_bias_dag.py': 'airflow/dags/m3/lojban_m3_16_continuous_graph_bias_dag.py',
    'airflow/dags/lojban_m3_17_advisor_reentry_bridge_dag.py': 'airflow/dags/m3/lojban_m3_17_advisor_reentry_bridge_dag.py',
    'airflow/dags/lojban_m3_18_decoder_reentry_resume_dag.py': 'airflow/dags/m3/lojban_m3_18_decoder_reentry_resume_dag.py',
    'airflow/dags/lojban_m3_19_d_mainline_grid_dag.py': 'airflow/dags/m3/lojban_m3_19_d_mainline_grid_dag.py',
    'airflow/dags/lojban_m3_5_symmetry_dag.py': 'airflow/dags/m3/lojban_m3_5_symmetry_dag.py',
    'airflow/dags/lojban_m3_6_symmetry_oracle_dag.py': 'airflow/dags/m3/lojban_m3_6_symmetry_oracle_dag.py',
    'airflow/dags/lojban_m3_7_shadow_alignment_dag.py': 'airflow/dags/m3/lojban_m3_7_shadow_alignment_dag.py',
    'airflow/dags/lojban_m3_8_operator_diversification_dag.py': 'airflow/dags/m3/lojban_m3_8_operator_diversification_dag.py',
    'airflow/dags/lojban_m3_9_primitive_probe_dag.py': 'airflow/dags/m3/lojban_m3_9_primitive_probe_dag.py',
    'airflow/dags/lojban_m3_plus_dag.py': 'airflow/dags/m3/lojban_m3_plus_dag.py',
    'airflow/dags/lojban_m4_0_semantic_probe_dag.py': 'airflow/dags/m4/lojban_m4_0_semantic_probe_dag.py',
    'airflow/dags/lojban_m4_2_predicate_grounding_dag.py': 'airflow/dags/m4/lojban_m4_2_predicate_grounding_dag.py',
    'airflow/dags/lojban_m4_series_dag.py': 'airflow/dags/m4/lojban_m4_series_dag.py',
    'airflow/dags/lojban_m5_2_autoregressive_chain_dag.py': 'airflow/dags/m5/lojban_m5_2_autoregressive_chain_dag.py',
    'airflow/dags/lojban_m5_3_masked_pair_chain_dag.py': 'airflow/dags/m5/lojban_m5_3_masked_pair_chain_dag.py',
    'airflow/dags/lojban_m5_autoformalization_dag.py': 'airflow/dags/m5/lojban_m5_autoformalization_dag.py',
    'airflow/dags/lojban_m5_padded_nary_dag.py': 'airflow/dags/m5/lojban_m5_padded_nary_dag.py',
    'airflow/dags/lojban_m_bridge_ablation_test_suite_dag.py': 'airflow/dags/m_bridge/lojban_m_bridge_ablation_test_suite_dag.py',
    'airflow/dags/lojban_phase_ablation_dag.py': 'airflow/dags/control_plane/lojban_phase_ablation_dag.py',
    'scripts/build_ablation_program_map.py': 'scripts/control_plane/build_ablation_program_map.py',
    'scripts/build_ablation_program_spine.py': 'scripts/control_plane/build_ablation_program_spine.py',
    'scripts/build_airflow_ablation_hypercube_report.py': 'scripts/control_plane/build_airflow_ablation_hypercube_report.py',
    'scripts/build_english_cot_control_dataset.py': 'scripts/data/build_english_cot_control_dataset.py',
    'scripts/build_full_coconut_report.py': 'scripts/control_plane/build_full_coconut_report.py',
    'scripts/build_gguf_pack.py': 'scripts/data/build_gguf_pack.py',
    'scripts/build_lora_dataset.py': 'scripts/data/build_lora_dataset.py',
    'scripts/build_m3_15_winograd_pack.py': 'scripts/data/build_m3_15_winograd_pack.py',
    'scripts/build_m3_19_resumption_pack.py': 'scripts/data/build_m3_19_resumption_pack.py',
    'scripts/build_mixed_curriculum_dataset.py': 'scripts/data/build_mixed_curriculum_dataset.py',
    'scripts/build_science_metrics_pack.py': 'scripts/data/build_science_metrics_pack.py',
    'scripts/build_synthetic_lora_dataset.py': 'scripts/data/build_synthetic_lora_dataset.py',
    'scripts/coconut_handoff.py': 'scripts/legacy/coconut_handoff.py',
    'scripts/eval_h5_dynamic_pointer_refactor.py': 'scripts/legacy/eval_h5_dynamic_pointer_refactor.py',
    'scripts/eval_h5_ood_stress.py': 'scripts/legacy/eval_h5_ood_stress.py',
    'scripts/eval_hf_adapter.py': 'scripts/legacy/eval_hf_adapter.py',
    'scripts/eval_hf_dual_mode_gate.py': 'scripts/legacy/eval_hf_dual_mode_gate.py',
    'scripts/eval_hf_dual_mode_handoff.py': 'scripts/legacy/eval_hf_dual_mode_handoff.py',
    'scripts/eval_j_1.py': 'scripts/legacy/eval_j_1.py',
    'scripts/eval_j_2.py': 'scripts/legacy/eval_j_2.py',
    'scripts/eval_j_3.py': 'scripts/legacy/eval_j_3.py',
    'scripts/eval_j_4.py': 'scripts/legacy/eval_j_4.py',
    'scripts/eval_j_5.py': 'scripts/legacy/eval_j_5.py',
    'scripts/eval_m6_logic_engine.py': 'scripts/m6/eval_m6_logic_engine.py',
    'scripts/eval_m7_interleaved.py': 'scripts/m7/eval_m7_interleaved.py',
    'scripts/eval_m8_council.py': 'scripts/m8/eval_m8_council.py',
    'scripts/eval_with_lms.py': 'scripts/legacy/eval_with_lms.py',
    'scripts/h_series_h5_metrics.py': 'scripts/legacy/h_series_h5_metrics.py',
    'scripts/latent_handoff_eval.py': 'scripts/legacy/latent_handoff_eval.py',
    'scripts/mine_compositional_anchors.py': 'scripts/data/mine_compositional_anchors.py',
    'scripts/multi_stage_cot.py': 'scripts/legacy/multi_stage_cot.py',
    'scripts/pipeline_eval_manifold.py': 'scripts/control_plane/pipeline_eval_manifold.py',
    'scripts/pipeline_train_grounded_reasoner.py': 'scripts/control_plane/pipeline_train_grounded_reasoner.py',
    'scripts/probe_m7_semantics.py': 'scripts/m7/probe_m7_semantics.py',
    'scripts/quick_baseline_test.py': 'scripts/legacy/quick_baseline_test.py',
    'scripts/reality_check.py': 'scripts/legacy/reality_check.py',
    'scripts/reconstruct_lora_dataset.py': 'scripts/data/reconstruct_lora_dataset.py',
    'scripts/render_ablation_history_catalog.py': 'scripts/control_plane/render_ablation_history_catalog.py',
    'scripts/run_ablation_history_backfill.py': 'scripts/control_plane/run_ablation_history_backfill.py',
    'scripts/run_causal_probe_matrix.py': 'scripts/legacy/run_causal_probe_matrix.py',
    'scripts/run_coconut_ablation_matrix.py': 'scripts/legacy/run_coconut_ablation_matrix.py',
    'scripts/run_drope_recalibration.py': 'scripts/legacy/run_drope_recalibration.py',
    'scripts/run_english_cot_control_duel.py': 'scripts/legacy/run_english_cot_control_duel.py',
    'scripts/run_experiment.py': 'scripts/legacy/run_experiment.py',
    'scripts/run_l6_ablation_branch.py': 'scripts/legacy/run_l6_ablation_branch.py',
    'scripts/run_m11_discriminative_suite.py': 'scripts/m11/run_m11_discriminative_suite.py',
    'scripts/run_m14_5_decompressor.py': 'scripts/m14/run_m14_5_decompressor.py',
    'scripts/run_m14_symbiote_scratchpad.py': 'scripts/m14/run_m14_symbiote_scratchpad.py',
    'scripts/run_m18_controller_family.py': 'scripts/m18/run_m18_controller_family.py',
    'scripts/run_m19_isolation_grid.py': 'scripts/m19/run_m19_isolation_grid.py',
    'scripts/run_m19_mainline_suite.py': 'scripts/m19/run_m19_mainline_suite.py',
    'scripts/run_m19_mainline_symbiote.py': 'scripts/m19/run_m19_mainline_symbiote.py',
    'scripts/run_m20_dictionary_first_suite.py': 'scripts/m20/run_m20_dictionary_first_suite.py',
    'scripts/run_m20_lock_suite.py': 'scripts/m20/run_m20_lock_suite.py',
    'scripts/run_m20_predicate_induction.py': 'scripts/m20/run_m20_predicate_induction.py',
    'scripts/run_m21_actual_bridge_suite.py': 'scripts/m21/run_m21_actual_bridge_suite.py',
    'scripts/run_m21_adversarial_audit.py': 'scripts/m21/run_m21_adversarial_audit.py',
    'scripts/run_m21_dynamic_bridi_suite.py': 'scripts/m21/run_m21_dynamic_bridi_suite.py',
    'scripts/run_m21_lock_suite.py': 'scripts/m21/run_m21_lock_suite.py',
    'scripts/run_m21_synthetic_assay_suite.py': 'scripts/m21/run_m21_synthetic_assay_suite.py',
    'scripts/run_m3_10_ood_accuracy_probe.py': 'scripts/m3/run_m3_10_ood_accuracy_probe.py',
    'scripts/run_m3_11_winograd_failure_anatomy.py': 'scripts/m3/run_m3_11_winograd_failure_anatomy.py',
    'scripts/run_m3_12_geometric_return_stream.py': 'scripts/m3/run_m3_12_geometric_return_stream.py',
    'scripts/run_m3_13_geometric_ablation_grid.py': 'scripts/m3/run_m3_13_geometric_ablation_grid.py',
    'scripts/run_m3_14_structural_alignment_bridge.py': 'scripts/m3/run_m3_14_structural_alignment_bridge.py',
    'scripts/run_m3_15_rotary_coconut.py': 'scripts/m3/run_m3_15_rotary_coconut.py',
    'scripts/run_m3_15_rotary_coconut_seven.py': 'scripts/m3/run_m3_15_rotary_coconut_seven.py',
    'scripts/run_m3_15b_relation_local_rotary.py': 'scripts/m3/run_m3_15b_relation_local_rotary.py',
    'scripts/run_m3_15c_family_conditioned_bridge.py': 'scripts/m3/run_m3_15c_family_conditioned_bridge.py',
    'scripts/run_m3_15d_answer_path_forcing.py': 'scripts/m3/run_m3_15d_answer_path_forcing.py',
    'scripts/run_m3_16_continuous_graph_bias.py': 'scripts/m3/run_m3_16_continuous_graph_bias.py',
    'scripts/run_m3_17_advisor_reentry_bridge.py': 'scripts/m3/run_m3_17_advisor_reentry_bridge.py',
    'scripts/run_m3_18_decoder_reentry_resume.py': 'scripts/m3/run_m3_18_decoder_reentry_resume.py',
    'scripts/run_m3_19_d_mainline_grid.py': 'scripts/m3/run_m3_19_d_mainline_grid.py',
    'scripts/run_m3_5_symmetry.py': 'scripts/m3/run_m3_5_symmetry.py',
    'scripts/run_m3_6_symmetry_oracle.py': 'scripts/m3/run_m3_6_symmetry_oracle.py',
    'scripts/run_m3_7_shadow_alignment.py': 'scripts/m3/run_m3_7_shadow_alignment.py',
    'scripts/run_m3_8_operator_diversification.py': 'scripts/m3/run_m3_8_operator_diversification.py',
    'scripts/run_m3_9_primitive_probe.py': 'scripts/m3/run_m3_9_primitive_probe.py',
    'scripts/run_m3_plus_family.py': 'scripts/m3/run_m3_plus_family.py',
    'scripts/run_m4_0_semantic_probe.py': 'scripts/m4/run_m4_0_semantic_probe.py',
    'scripts/run_m4_2_predicate_grounding.py': 'scripts/m4/run_m4_2_predicate_grounding.py',
    'scripts/run_m4_operator_family_eval.py': 'scripts/m4/run_m4_operator_family_eval.py',
    'scripts/run_m4_series.py': 'scripts/m4/run_m4_series.py',
    'scripts/run_m5_2_autoregressive_chain.py': 'scripts/m5/run_m5_2_autoregressive_chain.py',
    'scripts/run_m5_3_masked_pair_chain.py': 'scripts/m5/run_m5_3_masked_pair_chain.py',
    'scripts/run_m5_autoformalization.py': 'scripts/m5/run_m5_autoformalization.py',
    'scripts/run_m5_padded_nary_family.py': 'scripts/m5/run_m5_padded_nary_family.py',
    'scripts/run_m_bridge_ablation_test_suite.py': 'scripts/m_bridge/run_m_bridge_ablation_test_suite.py',
    'scripts/run_phase5_objective_ablation.py': 'scripts/legacy/run_phase5_objective_ablation.py',
    'scripts/run_phase5_train_ablation.py': 'scripts/legacy/run_phase5_train_ablation.py',
    'scripts/run_phase5_two_stage_recovery.py': 'scripts/legacy/run_phase5_two_stage_recovery.py',
    'scripts/run_phase_ablation.py': 'scripts/legacy/run_phase_ablation.py',
    'scripts/run_sapir_whorf_baseline.py': 'scripts/legacy/run_sapir_whorf_baseline.py',
    'scripts/run_three_engine_comparison.py': 'scripts/legacy/run_three_engine_comparison.py',
    'scripts/run_true_coconut_h_series.py': 'scripts/legacy/run_true_coconut_h_series.py',
    'scripts/trace_h5_provenance.py': 'scripts/legacy/trace_h5_provenance.py',
    'scripts/train_babel.py': 'scripts/training/train_babel.py',
    'scripts/train_cpu_hf.ps1': 'scripts/training/train_cpu_hf.ps1',
    'scripts/train_h5_persistent_vq_advisor.py': 'scripts/legacy/train_h5_persistent_vq_advisor.py',
    'scripts/train_h5_slice2_bridge.py': 'scripts/legacy/train_h5_slice2_bridge.py',
    'scripts/train_l_series_mvs.py': 'scripts/legacy/train_l_series_mvs.py',
    'scripts/train_lora.py': 'scripts/training/train_lora.py',
    'scripts/train_m20_dictionary.py': 'scripts/m20/train_m20_dictionary.py',
    'scripts/train_m21_dynamic_bridi.py': 'scripts/m21/train_m21_dynamic_bridi.py',
    'scripts/train_m5_2_autoregressive_chain.py': 'scripts/m5/train_m5_2_autoregressive_chain.py',
    'scripts/train_m5_3_masked_pair_chain.py': 'scripts/m5/train_m5_3_masked_pair_chain.py',
    'scripts/train_m5_padded_nary.py': 'scripts/m5/train_m5_padded_nary.py',
    'scripts/train_m6_logic_engine.py': 'scripts/m6/train_m6_logic_engine.py',
    'scripts/train_m7_interleaved.py': 'scripts/m7/train_m7_interleaved.py',
    'scripts/train_m8_council.py': 'scripts/m8/train_m8_council.py',
    'scripts/train_mixed_curriculum_cpu.ps1': 'scripts/training/train_mixed_curriculum_cpu.ps1',
    'scripts/train_swiglu_bridge.py': 'scripts/legacy/train_swiglu_bridge.py',
    'scripts/train_vq_reasoning_pilot.py': 'scripts/legacy/train_vq_reasoning_pilot.py',
    'scripts/true_coconut.py': 'scripts/legacy/true_coconut.py',
    'scripts/verify_h5_ablation.py': 'scripts/legacy/verify_h5_ablation.py',
    'scripts/visualize_shock.py': 'scripts/legacy/visualize_shock.py',
}


def _as_repo_relative(value: str | Path, repo_root: Path = REPO_ROOT) -> str:
    normalized = str(value).replace("\\", "/")
    root = repo_root.resolve().as_posix().rstrip("/")
    if normalized.startswith(root + "/"):
        normalized = normalized[len(root) + 1 :]
    if normalized == ".":
        return ""
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def canonical_repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> str:
    relative = _as_repo_relative(value, repo_root)
    return PATH_CANONICALIZATION.get(relative, relative)


def canonicalize_path_list(paths: Iterable[str | Path], repo_root: Path = REPO_ROOT) -> list[str]:
    return sorted({canonical_repo_path(path, repo_root) for path in paths if str(path).strip()})


def canonicalize_manifest_paths(
    payload: MutableMapping[str, Any],
    *,
    script_keys: Sequence[str] = ("script_paths", "scripts"),
    dag_keys: Sequence[str] = ("dag_paths", "dags"),
    repo_root: Path = REPO_ROOT,
) -> MutableMapping[str, Any]:
    for key in (*script_keys, *dag_keys):
        value = payload.get(key)
        if isinstance(value, list):
            payload[key] = canonicalize_path_list(value, repo_root)
    return payload


def canonicalize_manifest_tree(value: Any, repo_root: Path = REPO_ROOT) -> Any:
    if isinstance(value, MutableMapping):
        canonicalize_manifest_paths(value, repo_root=repo_root)
        for child in value.values():
            canonicalize_manifest_tree(child, repo_root)
    elif isinstance(value, list):
        for child in value:
            canonicalize_manifest_tree(child, repo_root)
    return value


def repo_relative(path: Path | None, repo_root: Path = REPO_ROOT) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def canonical_path_exists(value: str | Path, repo_root: Path = REPO_ROOT) -> bool:
    return (repo_root / canonical_repo_path(value, repo_root)).exists()


def latest_history_manifest(root: Path | None = None) -> Path:
    history_root = root or DEFAULT_HISTORY_ROOT
    path = latest_named_manifest(
        history_root,
        "ablation_history_manifest.json",
        recursive=False,
        newest_first=True,
        path_filter=None,
    )
    if path is None:
        raise FileNotFoundError(f"No ablation_history_manifest.json found under {history_root}")
    return path
