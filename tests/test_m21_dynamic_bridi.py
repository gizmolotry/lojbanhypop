from __future__ import annotations

import torch

from lojban_evolution.m21 import (
    CMAVO,
    GISMU,
    M21DynamicBridiQFormer,
    build_vocab,
    clamp_poincare_norm,
    compute_m21_loss,
    generate_dynamic_bridi_adversarial_examples,
    generate_dynamic_bridi_examples,
    judri_grounding_gate_from_logits,
    m21_collate,
    m21_default_grid,
    poincare_tangent_handoff,
    pointer_necessity_contrast_loss,
    train_m21_dynamic_bridi,
)
from lojban_evolution.m21.bridi import (
    CMAVO_TO_ID,
    GISMU_TO_ID,
    M21BridiDataset,
    RESERVED_ADVERSARIAL_AUDIT_SURFACES,
    SEMANTIC_COVERAGE_TRAINING_SURFACES,
    SEMANTIC_ROLE_CURRICULUM_TRAINING_SURFACES,
    adversarial_training_surface_names,
)
from lojban_evolution.m21.gauntlet import build_m21_gauntlet_payload, m21_to_m19_reservoir_shim


def test_generator_emits_variable_length_counterfactual_traces() -> None:
    examples = generate_dynamic_bridi_examples(192, seed=23, floating_fraction=0.08)

    frame_counts = {sum(1 for frame in row.frames if frame.active) for row in examples}
    assert min(frame_counts) == 0
    assert max(frame_counts) > 1

    groups: dict[str, set[tuple[int, tuple[int, ...]]]] = {}
    for row in examples:
        active = tuple((frame.gismu_id, tuple(frame.cmavo_ids)) for frame in row.frames if frame.active)
        groups.setdefault(row.counterfactual_group, set()).add(active)
    assert any(len(signatures) == 1 for signatures in groups.values())


def test_adversarial_generator_emits_heldout_surfaces_with_valid_traces() -> None:
    examples = generate_dynamic_bridi_adversarial_examples(144, seed=24)

    surfaces = {row.surface for row in examples}
    assert {"heldout_paraphrase", "role_distractor", "clausal_permutation", "oov_synonym"}.issubset(surfaces)
    assert all(any(frame.active for frame in row.frames) for row in examples)
    assert {row.answer_label for row in examples} == set(row.answer_label for row in examples)
    assert len({row.answer_label for row in examples}) == 18


def test_cmavo_minimal_pairs_change_semantics_without_changing_gismu() -> None:
    examples = generate_dynamic_bridi_examples(256, seed=29, floating_fraction=0.0, surfaces=("base",))
    excess = next(row for row in examples if row.answer_label == "size_excess")
    deficit = next(row for row in examples if row.answer_label == "size_deficit")
    excess_size = next(frame for frame in excess.frames if frame.active and frame.gismu_id == GISMU_TO_ID["size"])
    deficit_size = next(frame for frame in deficit.frames if frame.active and frame.gismu_id == GISMU_TO_ID["size"])

    assert excess_size.gismu_id == deficit_size.gismu_id
    assert CMAVO_TO_ID["excess"] in excess_size.cmavo_ids
    assert CMAVO_TO_ID["deficit"] in deficit_size.cmavo_ids
    assert excess.answer_id != deficit.answer_id


def test_judri_bindings_attach_to_gismu_place_structure() -> None:
    examples = generate_dynamic_bridi_examples(128, seed=31, floating_fraction=0.0)
    transfer = next(row for row in examples if row.answer_label == "transfer_success")
    frame = next(frame for frame in transfer.frames if frame.active and frame.gismu_id == GISMU_TO_ID["transfer"])

    assert frame.judri_place_bindings[:3] == (4, 5, 1)
    assert any(active.stop for active in transfer.frames if active.active)


def test_floating_gismu_without_judri_is_gated_silent() -> None:
    examples = generate_dynamic_bridi_examples(64, seed=41, floating_fraction=1.0)
    row = examples[0]

    assert row.is_floating
    assert all(not frame.active for frame in row.frames)


def test_loss_and_metrics_contract_runs_on_small_batch() -> None:
    examples = generate_dynamic_bridi_examples(48, seed=53)
    vocab = build_vocab(examples)
    dataset = M21BridiDataset(examples, vocab)
    batch = m21_collate([dataset[i] for i in range(16)])
    model = M21DynamicBridiQFormer(vocab_size=len(vocab), embedding_dim=16, hidden_dim=32)

    outputs = model(batch["input_ids"])
    loss, pieces = compute_m21_loss(outputs, batch, pointer_necessity_weight=0.5)

    assert torch.isfinite(loss)
    assert pieces["loss_trace"] > 0
    assert "loss_pointer_necessity" in pieces
    assert outputs["gismu_logits"].shape[-1] == len(GISMU)
    assert outputs["cmavo_logits"].shape[-1] == len(CMAVO)


def test_tiny_training_renders_token_efficiency_metrics() -> None:
    result = train_m21_dynamic_bridi(train_size=96, eval_size=48, epochs=1, batch_size=32, seed=23, embedding_dim=16, hidden_dim=32)
    metrics = result["metrics"]

    assert "strict_accuracy" in metrics
    assert "bridi_trace_exact_accuracy" in metrics
    assert "accuracy_per_token" in metrics
    assert "accuracy_per_trace_token" in metrics


def test_tiny_training_accepts_adversarial_augmentation() -> None:
    result = train_m21_dynamic_bridi(
        train_size=96,
        eval_size=48,
        epochs=1,
        batch_size=32,
        seed=25,
        embedding_dim=16,
        hidden_dim=32,
        adversarial_train_fraction=0.25,
        adversarial_train_surfaces="heldout_paraphrase,clausal_permutation",
    )

    assert result["config"]["adversarial_train_fraction"] == 0.25
    assert result["config"]["adversarial_train_surfaces"] == ["heldout_paraphrase", "clausal_permutation"]
    assert result["config"]["base_train_count"] == 72
    assert result["config"]["adversarial_train_count"] == 24
    assert result["config"]["effective_train_size"] == 96
    assert len(result["train_examples"]) == 96


def test_semantic_training_surfaces_are_explicit_and_accepted() -> None:
    training_surfaces = set(adversarial_training_surface_names())
    semantic_surfaces = set(SEMANTIC_COVERAGE_TRAINING_SURFACES) | set(SEMANTIC_ROLE_CURRICULUM_TRAINING_SURFACES)
    reserved_surfaces = set(RESERVED_ADVERSARIAL_AUDIT_SURFACES)

    assert semantic_surfaces.issubset(training_surfaces)
    assert semantic_surfaces.isdisjoint(reserved_surfaces)
    assert training_surfaces.isdisjoint(reserved_surfaces)

    for surface in SEMANTIC_COVERAGE_TRAINING_SURFACES:
        examples = generate_dynamic_bridi_adversarial_examples(18, seed=28, surfaces=(surface,))

        assert len(examples) == 18
        assert {row.surface for row in examples} == {surface}
        assert len({row.answer_label for row in examples}) == 18
        assert all(any(frame.active for frame in row.frames) for row in examples)
        if surface == "role_binding_train" or surface in SEMANTIC_ROLE_CURRICULUM_TRAINING_SURFACES:
            assert all(row.entities[0] != row.entities[6] for row in examples)

    result = train_m21_dynamic_bridi(
        train_size=36,
        eval_size=18,
        epochs=1,
        batch_size=18,
        seed=28,
        embedding_dim=16,
        hidden_dim=32,
        adversarial_train_fraction=0.5,
        adversarial_train_surfaces=",".join(SEMANTIC_COVERAGE_TRAINING_SURFACES),
    )

    assert result["config"]["adversarial_train_surfaces"] == list(SEMANTIC_COVERAGE_TRAINING_SURFACES)
    assert {row.surface for row in result["train_examples"]}.isdisjoint(reserved_surfaces)


def test_m21_grid_contains_semantic_coverage_ablation_cells() -> None:
    grid = m21_default_grid()
    cell_ids = [row["cell_id"] for row in grid]
    cells = {row["cell_key"]: row for row in grid}

    assert len(cell_ids) == len(set(cell_ids))
    assert cells["I"]["cell_id"] == "M21.1.I"
    assert cells["I"]["variant"]["adversarial_train_surfaces"] == (
        "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train"
    )
    assert cells["J"]["variant"]["adversarial_train_surfaces"] == "heldout_paraphrase,clausal_permutation,lexical_shift_train"
    assert cells["K"]["variant"]["adversarial_train_surfaces"] == "heldout_paraphrase,clausal_permutation,role_binding_train"
    assert cells["L"]["variant"]["adversarial_train_surfaces"] == (
        "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train"
    )
    assert cells["L"]["variant"]["adversarial_train_fraction"] == 0.5
    assert cells["M"]["variant"]["adversarial_train_surfaces"] == (
        "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train"
    )
    assert cells["N"]["variant"]["adversarial_train_surfaces"] == "heldout_paraphrase,clausal_permutation,role_binding_swap_train"
    assert cells["O"]["variant"]["adversarial_train_fraction"] == 0.35


def test_adversarial_training_rejects_reserved_audit_surfaces() -> None:
    for surface in RESERVED_ADVERSARIAL_AUDIT_SURFACES:
        try:
            train_m21_dynamic_bridi(
                train_size=32,
                eval_size=16,
                epochs=1,
                batch_size=16,
                seed=27,
                embedding_dim=16,
                hidden_dim=32,
                adversarial_train_fraction=0.25,
                adversarial_train_surfaces=surface,
            )
        except ValueError as exc:
            assert "Reserved M21 adversarial surfaces" in str(exc)
            assert surface in str(exc)
        else:
            raise AssertionError(f"Expected reserved adversarial surface {surface!r} to raise ValueError.")

    try:
        train_m21_dynamic_bridi(
            train_size=32,
            eval_size=16,
            epochs=1,
            batch_size=16,
            seed=27,
            embedding_dim=16,
            hidden_dim=32,
            adversarial_train_fraction=0.25,
            adversarial_train_surfaces="lexical_shift_train,oov_synonym",
        )
    except ValueError as exc:
        assert "Reserved M21 adversarial surfaces" in str(exc)
        assert "oov_synonym" in str(exc)
    else:
        raise AssertionError("Expected mixed semantic and reserved adversarial surfaces to raise ValueError.")


def test_adversarial_training_rejects_unknown_surfaces() -> None:
    try:
        train_m21_dynamic_bridi(
            train_size=32,
            eval_size=16,
            epochs=1,
            batch_size=16,
            seed=26,
            embedding_dim=16,
            hidden_dim=32,
            adversarial_train_fraction=0.25,
            adversarial_train_surfaces="typo_surface",
        )
    except ValueError as exc:
        assert "Unknown M21 adversarial surfaces" in str(exc)
    else:
        raise AssertionError("Expected invalid adversarial surface to raise ValueError.")


def test_pointer_necessity_hinge_matches_m19_contract() -> None:
    full_loss = torch.tensor(1.00, requires_grad=True)
    no_judri_loss = torch.tensor(0.98, requires_grad=True)

    loss = pointer_necessity_contrast_loss(full_loss, no_judri_loss, margin=0.05)
    loss.backward()

    assert torch.isclose(loss.detach(), torch.tensor(0.07), atol=1e-6)
    assert full_loss.grad is not None and full_loss.grad.item() > 0
    assert no_judri_loss.grad is not None and no_judri_loss.grad.item() < 0


def test_judri_grounding_gate_tracks_non_pad_mass() -> None:
    logits = torch.full((2, 3, 4, 5), -10.0)
    logits[0, :, :, 0] = 10.0
    logits[1, :, :, 2] = 10.0

    gate = judri_grounding_gate_from_logits(logits)

    assert gate.shape == (2, 3)
    assert float(gate[0].max().item()) < 1e-3
    assert float(gate[1].min().item()) > 0.999


def test_judri_bridge_gate_silences_ablated_predicate_paths() -> None:
    examples = generate_dynamic_bridi_examples(48, seed=57, floating_fraction=0.0)
    vocab = build_vocab(examples)
    dataset = M21BridiDataset(examples, vocab)
    batch = m21_collate([dataset[i] for i in range(16)])
    model = M21DynamicBridiQFormer(
        vocab_size=len(vocab),
        embedding_dim=16,
        hidden_dim=32,
        judri_bridge_gate=True,
    )

    outputs = model(batch["input_ids"])

    assert outputs["judri_bridge_gate_enabled"].item() == 1.0
    assert torch.isfinite(outputs["judri_bridge_gate_active_mean"])
    assert torch.isfinite(outputs["judri_bridge_gate_silenced_predicate_energy_mean"])
    assert torch.allclose(outputs["no_judri_answer_logits"], outputs["gismu_only_answer_logits"])
    assert not torch.allclose(outputs["answer_logits"], outputs["no_judri_answer_logits"])


def test_poincare_guardrail_and_forward_are_finite() -> None:
    examples = generate_dynamic_bridi_examples(32, seed=61)
    vocab = build_vocab(examples)
    batch = m21_collate([M21BridiDataset(examples, vocab)[i] for i in range(12)])
    huge = torch.randn(4, 8) * 100.0
    guarded, clipped = clamp_poincare_norm(huge, max_norm=0.99)
    tangent, _ = poincare_tangent_handoff(guarded, max_norm=0.99)

    assert bool(clipped.any().item())
    assert float(guarded.norm(dim=-1).max().item()) <= 0.9901
    assert torch.isfinite(tangent).all()

    model = M21DynamicBridiQFormer(
        vocab_size=len(vocab),
        embedding_dim=16,
        hidden_dim=32,
        geometry_mode="poincare",
        poincare_max_norm=0.99,
    )
    outputs = model(batch["input_ids"])
    loss, pieces = compute_m21_loss(outputs, batch, hyperbolic_topology_weight=0.01)

    assert torch.isfinite(loss)
    assert pieces["hyperbolic_max_norm"] <= 0.9901
    assert pieces["hyperbolic_distance_mean"] >= 0.0
    assert pieces["hyperbolic_tangent_handoff_finite_rate"] == 1.0
    assert pieces["hyperbolic_tangent_handoff_norm_mean"] >= 0.0


def test_m21_gauntlet_adapter_and_reservoir_shim() -> None:
    frames = torch.ones(2, 3, 5)
    activity = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    reservoir = m21_to_m19_reservoir_shim(frames, activity_mask=activity, max_m19_slots=6)

    assert reservoir.shape == (2, 6, 5)
    assert torch.allclose(reservoir[0, 1], torch.zeros(5))
    assert torch.allclose(reservoir[:, 3:], torch.zeros(2, 3, 5))

    suite_payload = {
        "cells": {
            "F": {
                "seed_reports": [
                    {
                        "metrics": {
                            "strict_accuracy": 0.8,
                            "bridi_trace_exact_accuracy": 0.9,
                            "no_judri_accuracy": 0.6,
                            "surface_metrics": {
                                "purged": {"strict_accuracy": 0.82},
                                "flattened": {"strict_accuracy": 0.70},
                                "renamed": {"strict_accuracy": 0.75},
                                "anonymized": {"strict_accuracy": 0.76},
                                "numeric": {"strict_accuracy": 0.78},
                            },
                        }
                    }
                ]
            }
        }
    }
    payload = build_m21_gauntlet_payload(
        suite_payload=suite_payload,
        actual_payload={"metrics": {"full_accuracy": 0.8, "no_judri_accuracy": 0.6, "judri_causal_delta": 0.2, "scratchpad_only_accuracy": 0.0}},
        run_id="test",
    )

    assert payload["metrics"]["purged_accuracy"] == 0.82
    assert payload["metrics"]["format_accuracy"] == 0.70
    assert payload["metrics"]["judri_causal_delta"] == 0.2
