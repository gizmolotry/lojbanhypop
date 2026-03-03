from __future__ import annotations

EXPECTED_VARIABLE_TOKEN_DISTRIBUTION_LENGTH = 1995


class FatalSafetyAssertionError(RuntimeError):
    """Raised when a safety-critical assertion fails."""


def assert_gearbox_halt_behavior(
    *,
    step_count: int,
    max_steps: int,
    halt_emitted: bool,
) -> None:
    """
    Validate that Gearbox emits a halt when a loop reaches the step limit.
    """
    if step_count >= max_steps and not halt_emitted:
        raise FatalSafetyAssertionError(
            "Gearbox infinite-loop guard failed to halt at step limit."
        )


def assert_manifold_relation_token_index(
    relation_token_index: int,
    *,
    vocabulary_size: int = EXPECTED_VARIABLE_TOKEN_DISTRIBUTION_LENGTH,
) -> None:
    """
    Validate a Manifold relation token index is within the legal vocabulary.
    """
    if not isinstance(relation_token_index, int) or isinstance(relation_token_index, bool):
        raise FatalSafetyAssertionError("Manifold relation token index must be an integer.")
    if relation_token_index < 0 or relation_token_index >= vocabulary_size:
        raise FatalSafetyAssertionError(
            f"Illegal Manifold relation token index: {relation_token_index}. "
            f"Expected [0, {vocabulary_size - 1}]."
        )
