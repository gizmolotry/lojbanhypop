from __future__ import annotations
from typing import Dict, Any

M19_REGISTRY: Dict[str, Any] = {
    "M19-v1": {
        "label": "Neuro-Symbolic Mainline",
        "defaults": {
            "tap_layer": 12,
            "intervention_layers": [12, 13, 14],
            "bottleneck_dim": 64,
            "scratchpad_length": 4,
            "residual_guard_weight": 5.0,
            "min_entropy": 0.85
        },
        "cells": {
            "BASE": {"label": "Direct Baseline", "intervention": "none"},
            "MAINLINE": {"label": "M19 Mainline Bridge", "intervention": "magnetic"}
        }
    }
}
