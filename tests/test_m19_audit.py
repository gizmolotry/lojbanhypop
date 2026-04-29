from __future__ import annotations

import importlib.util
from pathlib import Path
import torch


def _load_module():
    script_path = Path("scripts/m19/run_m19_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("run_m19_audit_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_infer_max_latent_steps_prefers_checkpoint_reservoir() -> None:
    module = _load_module()
    bridge_state = {
        "output_map.bias": torch.zeros(64),
        "collar.spatial_embeddings": torch.zeros(64, 128),
    }

    assert module._infer_max_latent_steps(bridge_state, configured_steps=8) == 64


def test_infer_max_latent_steps_respects_larger_configured_steps() -> None:
    module = _load_module()
    bridge_state = {
        "output_map.bias": torch.zeros(32),
    }

    assert module._infer_max_latent_steps(bridge_state, configured_steps=48) == 48
