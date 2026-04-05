from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

class M6MatrixCore(nn.Module):
    """
    The Topological Bedrock for M6.
    Enforces the strictly fixed 10-slot width: [OP, x1, x2, x3... x9]
    Slot 0: Operator (Codebook K=2000)
    Slots 1-9: Hard Pointers (Indices into S2a prompt embeddings)
    """
    def __init__(self, hidden_size: int, codebook_size: int = 2000, max_prompt_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.max_prompt_len = max_prompt_len
        self.num_x_slots = 9
        
        # Hardcoded indices from spec
        self.OP_QUOTE_IDX = 0
        self.OP_STOP_IDX = 1
        self.PAD_IDX = 2
        
        # Head for the Operator slot (K=2000)
        self.op_head = nn.Linear(hidden_size, codebook_size)
        
        # Heads for the 9 Variable slots (Pointers to prompt locations)
        self.x_heads = nn.ModuleList([
            nn.Linear(hidden_size, max_prompt_len) for _ in range(self.num_x_slots)
        ])

    def forward(self, s1_hidden_state: torch.Tensor, use_iron_collar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces the 10-slot discrete logic matrix components.
        Returns:
            op_logits: [B, K]
            x_logits: [B, 9, Max_Prompt_Len]
        """
        op_logits = self.op_head(s1_hidden_state)
        
        if use_iron_collar:
            # Physical Law #1: The Iron Collar
            # Strictly isolate Operator to indices 0-4 (AND, OR, NOT, IMPLIES, XOR)
            mask = torch.full_like(op_logits, -1e9)
            mask[:, :5] = 0
            op_logits = op_logits + mask
            
        x_logits_list = []
        for head in self.x_heads:
            x_logits_list.append(head(s1_hidden_state))
            
        x_logits = torch.stack(x_logits_list, dim=1)
        return op_logits, x_logits
    
    def apply_iron_collar(self, op_logits: torch.Tensor, x_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Strict Arity-Typing Mask (Physical Law #1).
        In M6, the slots are architecturally separated, so the collar is implicit.
        However, we can enforce the Operator slot to exclude 'Variable' semantics 
        if we choose to restrict the 2000-token codebook.
        """
        return op_logits, x_logits
