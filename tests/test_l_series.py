from __future__ import annotations

import torch

from lojban_evolution.l_series import (
    AugmentedLagrangianController,
    build_scope_tokens_from_triples,
    compute_arity_violation,
    compute_scope_violation_rate,
    crispness_loss,
    entropy_floor_penalty,
    make_swap_variant,
    overflow_penalty,
    parse_scope_trace,
)


def test_augmented_lagrangian_controller_updates_monotonic() -> None:
    ctl = AugmentedLagrangianController(rho=0.5, init_lambda=0.0, max_lambda=10.0)
    ctl.update({"arity": 0.2, "scope": 0.0, "identity": 0.3})
    assert ctl.lambdas["arity"] == 0.1
    assert ctl.lambdas["scope"] == 0.0
    assert ctl.lambdas["identity"] == 0.15


def test_arity_violation_rate() -> None:
    triples = [(0, 5, 7), (4, 9, 10), (6, 4, 8)]
    v = compute_arity_violation(triples, relation_vocab=5, var_min_id=5)
    assert abs(v - (1.0 / 3.0)) < 1e-6


def test_scope_parser_detects_unbound_and_mismatch() -> None:
    toks = ["SCOPE_OPEN", "FORALL", "VAR_5", "VAR_7", "SCOPE_CLOSE", "SCOPE_CLOSE"]
    s = parse_scope_trace(toks)
    assert s.unbound >= 1
    assert s.mismatch >= 1


def test_scope_tokens_builder_roundtrip_has_low_violation() -> None:
    triples = [(0, 5, 6), (1, 5, 6), (2, 6, 5)]
    toks = build_scope_tokens_from_triples(triples)
    v = compute_scope_violation_rate(toks)
    assert v <= 0.25


def test_swap_variant() -> None:
    out = make_swap_variant("Alice is north of Bob.", "Alice")
    assert out is not None
    p2, a2 = out
    assert "Bob" in p2 and "Alice" in p2
    assert a2 == "Bob"


def test_crispness_and_entropy_penalties() -> None:
    p = torch.tensor([0.1, 0.5, 0.9])
    c = crispness_loss(p)
    assert float(c.item()) > 0.0

    q = torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32)
    e = entropy_floor_penalty(q, h_min=0.8)
    assert float(e.item()) >= 0.0


def test_overflow_penalty() -> None:
    assert overflow_penalty(active_refs=65, capacity=64, ovf_emitted=False) == 1.0
    assert overflow_penalty(active_refs=65, capacity=64, ovf_emitted=True) == 0.0
    assert overflow_penalty(active_refs=10, capacity=64, ovf_emitted=False) == 0.0
