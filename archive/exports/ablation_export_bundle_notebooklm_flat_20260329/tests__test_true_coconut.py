from __future__ import annotations

import types
from unittest.mock import Mock

import torch

from conftest import load_script_module


mod = load_script_module("true_coconut", "scripts/true_coconut.py")


class _TokenizerStub:
    def __init__(self, lengths: dict[str, int], eos_token_id: int = 2):
        self.lengths = lengths
        self.eos_token_id = eos_token_id

    def __call__(self, text: str, return_tensors: str = "pt"):
        seq_len = self.lengths[text]
        return types.SimpleNamespace(input_ids=torch.ones((1, seq_len), dtype=torch.long))

    def decode(self, _ids, skip_special_tokens: bool = True) -> str:
        return "ANSWER: stub"


def _build_embedder(hidden_size: int):
    def _embed(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((ids.shape[0], ids.shape[1], hidden_size), dtype=torch.float32)

    return _embed


def test_true_coconut_generate_concatenates_prefix_virtual_suffix_to_expected_shape(monkeypatch):
    hidden_size = 8
    question = "Q?"
    handoff_suffix = " SUFFIX "
    logic_prompt = mod.build_logic_prompt(question)
    prefix_prompt = mod.build_final_prompt_prefix(question)
    tokenizer = _TokenizerStub({logic_prompt: 4, prefix_prompt: 10, handoff_suffix: 5})

    model = types.SimpleNamespace(
        device=torch.device("cpu"),
        get_input_embeddings=Mock(return_value=_build_embedder(hidden_size)),
        generate=Mock(return_value=torch.tensor([[11, 12, 13]], dtype=torch.long)),
    )

    monkeypatch.setattr(
        mod,
        "_greedy_logic_with_last_hidden",
        Mock(return_value=(torch.tensor([[7, 8]], dtype=torch.long), torch.zeros((1, hidden_size)))),
    )

    out = mod.true_coconut_generate(
        model=model,
        tokenizer=tokenizer,
        question=question,
        max_logic_new_tokens=4,
        max_final_new_tokens=6,
        handoff_suffix=handoff_suffix,
    )

    assert out["final_answer"] == "stub"
    inputs_embeds = model.generate.call_args.kwargs["inputs_embeds"]
    assert tuple(inputs_embeds.shape) == (1, 16, hidden_size)


def test_hidden_state_extraction_uses_final_layer_last_token_with_token_axis_preserved():
    hidden_size = 6
    start_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    layer0 = torch.zeros((1, 4, hidden_size), dtype=torch.float32)
    layer1 = torch.full((1, 4, hidden_size), 7.0, dtype=torch.float32)
    layer2 = torch.arange(1 * 4 * hidden_size, dtype=torch.float32).reshape(1, 4, hidden_size)
    expected = layer2[:, -1:, :]

    model_out = types.SimpleNamespace(
        past_key_values=((torch.zeros((1, 1, 4, 4)), torch.zeros((1, 1, 4, 4))),),
        logits=torch.tensor([[[0.1, 0.2], [0.2, 0.1], [0.3, 0.1], [0.9, 0.1]]], dtype=torch.float32),
        hidden_states=(layer0, layer1, layer2),
    )

    class _Model:
        def __init__(self):
            self.config = types.SimpleNamespace(hidden_size=hidden_size)

        def __call__(self, **_kwargs):
            return model_out

    _logic_ids, last_hidden = mod._greedy_logic_with_last_hidden(
        model=_Model(),
        start_ids=start_ids,
        max_new_tokens=1,
        eos_token_id=None,
    )

    assert tuple(last_hidden.shape) == (1, 1, hidden_size), (
        "Expected final-layer last-token hidden state to preserve token axis as [1,1,H]."
    )
    assert torch.equal(last_hidden, expected)


def test_true_coconut_generate_enters_adapter_disabled_context_before_final_generate(monkeypatch):
    hidden_size = 5
    question = "Q?"
    handoff_suffix = " END"
    logic_prompt = mod.build_logic_prompt(question)
    prefix_prompt = mod.build_final_prompt_prefix(question)
    tokenizer = _TokenizerStub({logic_prompt: 3, prefix_prompt: 10, handoff_suffix: 5})

    events: list[str] = []
    state = {"entered": False}

    class _DisableCtx:
        def __enter__(self):
            state["entered"] = True
            events.append("enter")
            return None

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            state["entered"] = False
            return False

    def _generate_side_effect(*_args, **_kwargs):
        events.append("generate")
        assert state["entered"] is True, "model.generate was called before adapter-disabled context entry."
        return torch.tensor([[101, 102]], dtype=torch.long)

    model = types.SimpleNamespace(
        device=torch.device("cpu"),
        get_input_embeddings=Mock(return_value=_build_embedder(hidden_size)),
        generate=Mock(side_effect=_generate_side_effect),
        disable_adapter=Mock(return_value=_DisableCtx()),
    )

    monkeypatch.setattr(
        mod,
        "_greedy_logic_with_last_hidden",
        Mock(return_value=(torch.tensor([[1]], dtype=torch.long), torch.zeros((1, hidden_size)))),
    )

    mod.true_coconut_generate(
        model=model,
        tokenizer=tokenizer,
        question=question,
        max_logic_new_tokens=2,
        max_final_new_tokens=2,
        handoff_suffix=handoff_suffix,
    )

    model.disable_adapter.assert_called_once()
    assert events == ["enter", "generate", "exit"]
