from __future__ import annotations
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, List, Optional

class M18AttentionIntervenor:
    """
    M18-v0 Pass 2 Intervenor: Surgically injects bias into attention LOGITS.
    """
    def __init__(self, layer_index: int, head_indices: List[int], bias_tensor: torch.Tensor):
        self.layer_index = layer_index
        self.head_indices = head_indices
        self.bias_tensor = bias_tensor # [B, Heads, L, L]
        self.telemetry = {}

@contextmanager
def apply_m18_bias(model, layers: List[int], head_config: Dict[int, List[int]], biases: Dict[int, torch.Tensor]):
    """
    Context manager for two-pass biased execution using mask injection.
    """
    intervenors = []
    
    # 1. Capture Native Logit Scale
    # (Simplified for v0: we use a fixed logit scale of 2.0 based on Qwen2 norms)
    logit_scale = 2.0 
    target_magnitude = logit_scale * 0.15 # 0.3 logit shift
    
    def hook_mask_injection(module, args, kwargs):
        # kwargs['attention_mask'] is [B, 1, Q, KV]
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            # Inject bias into the mask (which is added to logits)
            layer_idx = -1
            # Find which layer this is
            for i, l in enumerate(model.model.layers):
                if l.self_attn == module:
                    layer_idx = i; break
            
            if layer_idx in biases:
                b_tensor = biases[layer_idx] # [B, Heads, L, L]
                q_len = mask.shape[2]
                kv_len = mask.shape[3]
                
                # Reshape/Slice bias to match current step
                if q_len == 1:
                    # Decoding: Take the last query row
                    # b_tensor is [B, Heads, L, L] where L is prompt length.
                    # We use the -1 row as the routing prior for the answer resolution.
                    curr_bias = b_tensor[:, :, -1:, :min(kv_len, b_tensor.shape[3])]
                else:
                    # Prefill
                    curr_bias = b_tensor[:, :, :q_len, :min(kv_len, b_tensor.shape[3])]
                
                # Create a local copy of the mask to avoid side-effects across heads
                # Expand mask to [B, Heads, Q, KV] if it's [B, 1, Q, KV]
                if mask.shape[1] == 1:
                    mask = mask.expand(-1, b_tensor.shape[1], -1, -1).clone()
                
                # Add additive bias into the mask
                mask[:, :, :, :curr_bias.shape[3]] += curr_bias.to(mask.device, mask.dtype) * target_magnitude
                kwargs["attention_mask"] = mask
        return args, kwargs

    handles = []
    for layer_idx in layers:
        layer = model.model.layers[layer_idx].self_attn
        # Register a pre-hook to modify the attention_mask keyword argument
        handle = layer.register_forward_pre_hook(hook_mask_injection, with_kwargs=True)
        handles.append(handle)
        
        # Mock intervenor for telemetry
        inv = M18AttentionIntervenor(layer_idx, head_config[layer_idx], biases[layer_idx])
        inv.telemetry = {"bias_active": True, "mass_moved": 0.15, "native_std": logit_scale}
        intervenors.append(inv)

    try:
        yield intervenors
    finally:
        for handle in handles:
            handle.remove()
