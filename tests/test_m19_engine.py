from __future__ import annotations

import torch

from lojban_evolution.m19.engine import (
    M19SymbioteBridge,
    batched_pairwise_cosine_stats,
    compute_query_repulsion_loss,
    operator_distribution_stats,
    pairwise_cosine_stats,
)
from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.m19.typed_physics import (
    apply_radius_bands,
    build_typed_targets,
    expmap0,
    load_typed_physics_config,
    logmap0,
    parse_typed_slot_layout,
)


def test_pairwise_cosine_stats_detects_identical_vectors() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    stats = pairwise_cosine_stats(vectors)

    assert stats["pairwise_cosine_mean"] == 1.0
    assert stats["pairwise_cosine_max"] == 1.0
    assert stats["anisotropy"] == 1.0


def test_batched_pairwise_cosine_stats_respects_lengths() -> None:
    trace = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    lengths = torch.tensor([2, 2])
    stats = batched_pairwise_cosine_stats(trace, lengths=lengths)

    assert abs(stats["pairwise_cosine_mean"] - 0.5) < 1e-6
    assert stats["pairwise_cosine_max"] == 1.0


def test_query_repulsion_penalizes_collapsed_queries() -> None:
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    collapsed_loss = compute_query_repulsion_loss(collapsed, margin=0.15)
    separated_loss = compute_query_repulsion_loss(separated, margin=0.15)

    assert float(collapsed_loss.item()) > 0.8
    assert float(separated_loss.item()) == 0.0


def test_operator_distribution_stats_reports_entropy_and_top1_share() -> None:
    logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        ]
    )
    stats = operator_distribution_stats(logits)

    assert stats["operator_top1_share_mean"] > 0.99
    assert stats["operator_entropy_ratio_mean"] < 0.05


def test_typed_slot_layout_and_targets_resolve_lojbanic_families() -> None:
    config = load_typed_physics_config("D:/lojbanhypop/configs/m19_typed_physics_ontology.json")
    layout = parse_typed_slot_layout("gismu:2,cmavo:2,judri:4")
    targets = build_typed_targets(
        raw_text="QUESTION: demo\nTRACE: REL_LEFT_OF BIND_AGENT VERIFY_RESULT\nANSWER: yes",
        mode="crystal",
        config=config,
    )

    assert layout == ["gismu", "gismu", "cmavo", "cmavo", "judri", "judri", "judri", "judri"]
    assert targets.has_supervision is True
    assert targets.primary_arity == 2
    assert targets.pointer_budget == 2
    assert len(targets.family_ids) == 3


def test_gumbel_hard_routing_zeroes_unused_judri_slots_and_keeps_gradients() -> None:
    bridge = M19SymbioteBridge(
        hidden_size=16,
        bottleneck_dim=8,
        scratchpad_len=4,
        num_queries=8,
        typed_slot_layout=["gismu", "gismu", "cmavo", "cmavo", "judri", "judri", "judri", "judri"],
        arity_router_mode="gumbel_hard",
        gumbel_hard=True,
    )
    with torch.no_grad():
        assert bridge.arity_head is not None
        bridge.arity_head.weight.zero_()
        bridge.arity_head.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))

    h_tap = torch.randn(1, 6, 16, requires_grad=False)
    delta, _, _, telemetry = bridge(h_tap, active_steps=4, gumbel_temperature=0.05)
    query_state = telemetry["query_state"]
    judri_mask = telemetry["judri_mask"]

    assert judri_mask is not None
    assert judri_mask.shape[-1] == 4
    assert judri_mask[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.allclose(query_state[0, 5:, :], torch.zeros_like(query_state[0, 5:, :]))

    loss = delta.sum()
    loss.backward()
    assert bridge.arity_head.weight.grad is not None


def test_hyperbolic_projection_and_logexp_are_well_behaved() -> None:
    x = torch.randn(2, 8, 6) * 0.3
    y = expmap0(logmap0(x, curvature=1.0), curvature=1.0)
    layout = ["gismu", "gismu", "cmavo", "cmavo", "judri", "judri", "judri", "judri"]
    projected, clip_mask = apply_radius_bands(
        x,
        layout,
        {
            "gismu": {"min": 0.05, "max": 0.30},
            "cmavo": {"min": 0.18, "max": 0.48},
            "judri": {"min": 0.48, "max": 0.82},
            "control": {"min": 0.08, "max": 0.40},
        },
        curvature=1.0,
    )

    assert torch.isfinite(y).all()
    assert (projected.norm(dim=-1) < 1.0).all()
    assert clip_mask.shape == projected.shape[:2]


def test_m19_registry_exposes_typed_physics_tracks_and_aliases() -> None:
    assert "M19.31" in M19_REGISTRY
    assert "M19.32" in M19_REGISTRY
    assert "M19.3c" in M19_REGISTRY
    assert "M19.3d" in M19_REGISTRY
