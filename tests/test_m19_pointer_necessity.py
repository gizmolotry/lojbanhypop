from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_train_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "train_m19_mainline.py"
    spec = importlib.util.spec_from_file_location("train_m19_mainline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load_train_module()
_pointer_necessity_contrast_loss = _train._pointer_necessity_contrast_loss


def test_pointer_necessity_contrast_pushes_full_path_over_no_judri_path() -> None:
    full_loss = torch.tensor(1.00, requires_grad=True)
    no_judri_loss = torch.tensor(0.98, requires_grad=True)

    loss = _pointer_necessity_contrast_loss(full_loss, no_judri_loss, margin=0.05)
    loss.backward()

    assert abs(float(loss.item()) - 0.07) < 1e-6
    assert float(full_loss.grad.item()) > 0.0
    assert float(no_judri_loss.grad.item()) < 0.0


def test_pointer_necessity_contrast_is_zero_once_margin_is_satisfied() -> None:
    full_loss = torch.tensor(1.00, requires_grad=True)
    no_judri_loss = torch.tensor(1.20, requires_grad=True)

    loss = _pointer_necessity_contrast_loss(full_loss, no_judri_loss, margin=0.05)
    loss.backward()

    assert float(loss.item()) == 0.0
    assert float(full_loss.grad.item()) == 0.0
    assert float(no_judri_loss.grad.item()) == 0.0
