from __future__ import annotations
from typing import TypedDict, List, Dict, Any

class M18AblationCell(TypedDict):
    label: str
    ontology: str  # 'U' (Unconstrained), 'L' (Logebonic), 'None'
    intervention: str # 'typed', 'random_sparse', 'shuffled_labels', 'shuffled_pos', 'scrambled_links', 'wrong_heads', 'uniform_dense', 'none'
    pass_count: int
    description: str

M18_REGISTRY = {
    "M18-v0": {
        "cells": {
            "BASE": {
                "label": "Base Host",
                "ontology": "None",
                "intervention": "none",
                "pass_count": 1,
                "description": "Frozen base model, no symbiote."
            },
            "U-TYPED": {
                "label": "Condition U: Typed Symbiote",
                "ontology": "U",
                "intervention": "typed",
                "pass_count": 2,
                "description": "Two-pass, unconstrained latent relations."
            },
            "L-TYPED": {
                "label": "Condition L: Logebonic Symbiote",
                "ontology": "L",
                "intervention": "typed",
                "pass_count": 2,
                "description": "Two-pass, constrained Logebonic ontology."
            },
            "KILL-RANDOM": {
                "label": "Kill: Random Sparse",
                "ontology": "None",
                "intervention": "random_sparse",
                "pass_count": 2,
                "description": "Random sparse bias, same mass as symbiote."
            },
            "KILL-LABEL": {
                "label": "Kill: Shuffled Labels",
                "ontology": "U",
                "intervention": "shuffled_labels",
                "pass_count": 2,
                "description": "Correct positions, shuffled relation types."
            },
            "KILL-POS": {
                "label": "Kill: Shuffled Positions",
                "ontology": "U",
                "intervention": "shuffled_pos",
                "pass_count": 2,
                "description": "Shuffled salient positions, same relations."
            },
            "KILL-LINK": {
                "label": "Kill: Scrambled Links",
                "ontology": "L",
                "intervention": "scrambled_links",
                "pass_count": 2,
                "description": "Correct operator classes, scrambled pointer links."
            },
            "KILL-HEAD": {
                "label": "Kill: Wrong Heads/Layers",
                "ontology": "U",
                "intervention": "wrong_heads",
                "pass_count": 2,
                "description": "Correct biases applied to random intervention heads."
            },
            "KILL-DENSE": {
                "label": "Kill: Uniform Dense",
                "ontology": "None",
                "intervention": "uniform_dense",
                "pass_count": 2,
                "description": "Uniform dense bias across the context."
            }
        },
        "defaults": {
            "tap_layer": 12,
            "intervention_layers": [12, 13, 14],
            "top_k": 6,
            "bias_magnitude_target": 0.15, # 15% of native logit range
            "hidden_size": 896
        }
    }
}
