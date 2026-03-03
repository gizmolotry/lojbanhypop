from __future__ import annotations

import pytest

from lojban_evolution.safety_assertions import (
    FatalSafetyAssertionError,
    assert_gearbox_halt_behavior,
    assert_manifold_relation_token_index,
)


def test_gearbox_infinite_loop_requires_halt_signal() -> None:
    with pytest.raises(FatalSafetyAssertionError, match="infinite-loop guard"):
        assert_gearbox_halt_behavior(step_count=1000, max_steps=1000, halt_emitted=False)


def test_gearbox_allows_halt_at_limit() -> None:
    assert_gearbox_halt_behavior(step_count=1000, max_steps=1000, halt_emitted=True)


def test_manifold_illegal_relation_token_index_raises_fatal() -> None:
    with pytest.raises(FatalSafetyAssertionError, match="Illegal Manifold relation token index: 2005"):
        assert_manifold_relation_token_index(2005)


def test_manifold_rejects_non_integer_relation_index() -> None:
    with pytest.raises(FatalSafetyAssertionError, match="must be an integer"):
        assert_manifold_relation_token_index(1.5)  # type: ignore[arg-type]
