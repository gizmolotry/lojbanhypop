from __future__ import annotations

from conftest import load_script_module


def test_loss_weighting_application():
    mod = load_script_module("train_lora", "scripts/train_lora.py")

    trace_multiplier = 2.0
    answer_multiplier = 1.0
    assert trace_multiplier > answer_multiplier, "System 2 rigidity is not prioritized over System 1 fluidity."

    text = "QUESTION: x\nTRACE: A B C\nANSWER: y"
    ids_len = len(text)
    trace_start = mod.find_anchor_boundary(text, "\nTRACE:", ids_len, prefix_token_len_fn=len)
    answer_start = mod.find_anchor_boundary(text, "\nANSWER:", ids_len, prefix_token_len_fn=len)
    prompt_end_pos = text.find("\nTRACE:")
    anchor_pos = answer_start

    assert trace_start >= 0
    assert answer_start > trace_start
    assert anchor_pos > prompt_end_pos, "Loss anchor is misaligned with the reasoning trace."

    weights = mod.compute_segment_weights(
        ids_len=ids_len,
        prompt_w=0.2,
        trace_w=trace_multiplier,
        answer_w=answer_multiplier,
        trace_start=trace_start,
        answer_start=answer_start,
    )
    assert weights[0] == 0.2
    assert weights[trace_start] == trace_multiplier
    assert weights[answer_start] == answer_multiplier

