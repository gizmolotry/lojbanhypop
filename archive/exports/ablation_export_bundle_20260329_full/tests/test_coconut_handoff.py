from __future__ import annotations

import types
from unittest.mock import Mock

import torch
import pytest

from conftest import load_script_module


mod = load_script_module("coconut_handoff", "scripts/coconut_handoff.py")


def _make_cache(num_layers: int = 2, heads: int = 2, seq: int = 4, dim: int = 3):
    layer = (
        torch.zeros((1, heads, seq, dim), dtype=torch.float32),
        torch.zeros((1, heads, seq, dim), dtype=torch.float32),
    )
    return tuple(layer for _ in range(num_layers))


def test_validate_cache_shape_flags_layer_mismatch():
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=3, hidden_size=6))
    ok, msg = mod._validate_cache_shape(_make_cache(num_layers=2), model)
    assert ok is False
    assert "Layer mismatch" in msg


def test_validate_cache_shape_flags_head_dim_mismatch():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(num_hidden_layers=2, num_key_value_heads=4, head_dim=8)
    )
    ok, msg = mod._validate_cache_shape(_make_cache(num_layers=2, heads=2, dim=3), model)
    assert ok is False
    assert "Head-dim mismatch" in msg


def test_coconut_handoff_raises_before_airlock_on_bad_cache():
    model = types.SimpleNamespace(
        device=torch.device("cpu"),
        config=types.SimpleNamespace(num_hidden_layers=2, hidden_size=6),
    )
    tok = Mock()
    tok.eos_token_id = 2
    tok.side_effect = None

    def _tok(_text, return_tensors="pt"):
        return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    tok.__call__ = _tok
    tok.decode = Mock(return_value="trace _E")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "_generate_with_cache", Mock(return_value=(torch.tensor([[9, 9]]), _make_cache(num_layers=1))))
        with pytest.raises(RuntimeError, match="KV-cache alignment error before airlock"):
            mod.coconut_handoff_answer(
                model=model,
                tokenizer=tok,
                question="Q?",
                max_logic_new_tokens=4,
                max_final_new_tokens=4,
                lojban_exit_token="_E",
            )


def test_patch_nope_qwen2_overwrites_rotary_function():
    fake_qwen2_mod = types.SimpleNamespace()

    def _orig_rotary(*_args, **_kwargs):
        return "orig"

    fake_qwen2_mod.apply_rotary_pos_emb = _orig_rotary
    fake_qwen2_pkg = types.SimpleNamespace(modeling_qwen2=fake_qwen2_mod)
    fake_models_pkg = types.SimpleNamespace(qwen2=fake_qwen2_pkg)
    fake_transformers_pkg = types.SimpleNamespace(models=fake_models_pkg)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(__import__("sys").modules, "transformers", fake_transformers_pkg)
        mp.setitem(__import__("sys").modules, "transformers.models", fake_models_pkg)
        mp.setitem(__import__("sys").modules, "transformers.models.qwen2", fake_qwen2_pkg)
        mp.setitem(__import__("sys").modules, "transformers.models.qwen2.modeling_qwen2", fake_qwen2_mod)

        patched = mod.patch_nope_qwen2()
        assert patched is True
        assert fake_qwen2_mod.apply_rotary_pos_emb is not _orig_rotary


def test_adapter_disabled_calls_disable_adapter_context():
    ctx = Mock()
    ctx.__enter__ = Mock(return_value=None)
    ctx.__exit__ = Mock(return_value=False)

    model = Mock()
    model.disable_adapter = Mock(return_value=ctx)

    with mod.adapter_disabled(model):
        pass

    model.disable_adapter.assert_called_once()
    ctx.__enter__.assert_called_once()
    ctx.__exit__.assert_called_once()


def test_extract_answer_prefers_marker_and_handles_default_line():
    assert mod.extract_answer("x\nANSWER: 42\njunk") == "42"
    assert mod.extract_answer("x\nFinal answer: 9") == "9"
    assert mod.extract_answer("line1\nline2") == "line2"


def test_answers_match_with_symbolic_tokens_present():
    expected = "A=knight, B=knave, C=knight"
    predicted = "TRACE => ... _E ANSWER: A is knight, B is knave, C is knight"
    assert mod.answers_match(expected, predicted) is True
