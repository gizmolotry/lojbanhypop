from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

class M6MatrixCore(nn.Module):
    """
    The Topological Bedrock for M6.
    Enforces the strictly fixed 10-slot width: [OP, x1, x2, x3... x9]
    """
    def __init__(self, hidden_size: int, codebook_size: int = 2000, num_slots: int = 10):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        
        # OP_QUOTE is hardcoded to invoke the pointer network
        self.OP_QUOTE_IDX = 0
        self.OP_STOP_IDX = 1
        self.PAD_IDX = 2
        
        # Emits logits for the 10 slots
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size) for _ in range(num_slots)
        ])

    def forward(self, s1_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Takes the autoregressive hidden state from System 1 and produces
        the 10-slot discrete logic matrix.
        """
        # s1_hidden_state: [batch, length, hidden_size]
        slot_logits = []
        for head in self.heads:
            slot_logits.append(head(s1_hidden_state))
            
        # Output: [batch, length, num_slots, codebook_size]
        return torch.stack(slot_logits, dim=2)
    
    def apply_lc6_constraints(self, slot_logits: torch.Tensor, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Dynamic Arity via Decay:
        If gate_values[slot_i] is low, slot_i is forced to PAD_IDX.
        """
        # slot_logits: [B, num_slots, codebook_size]
        # gate_values: [B, num_slots] (0.0 to 1.0)
        
        # Create a strong bias toward PAD_IDX (index 2)
        pad_mask = torch.zeros_like(slot_logits)
        pad_mask[:, :, self.PAD_IDX] = 1e9 
        
        # Interpolate between raw logits and the PAD force based on arity-decay
        # gate_values == 1.0 means active, 0.0 means decay to <PAD>
        decayed_logits = (gate_values.unsqueeze(-1) * slot_logits) + ((1.0 - gate_values.unsqueeze(-1)) * pad_mask)
        return decayed_logits
