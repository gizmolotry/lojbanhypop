from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class BlankSlateCodebook(nn.Module):
    """
    The M7 Blank Slate Codebook with Vector Choke.
    Dimension is bottlenecked to d=16 to kill semantic bleed.
    """
    def __init__(self, codebook_size: int = 2000, hidden_size: int = 896, choke_dim: int = 16):
        super().__init__()
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.choke_dim = choke_dim
        
        # Project full hidden state down to syntax-only bottleneck
        self.down_proj = nn.Linear(hidden_size, choke_dim)
        
        # The true blank slate codebook (K=2000, D=16)
        self.emb = nn.Parameter(torch.empty(codebook_size, choke_dim))
        torch.nn.init.normal_(self.emb, mean=0.0, std=0.02)
        
        # Project choked syntax back up for injection
        self.up_proj = nn.Linear(choke_dim, hidden_size)
        
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes System 1 hidden state, bottlenecks it, and quantizes.
        """
        b, h = z.shape
        z_choked = self.down_proj(z) # [B, 16]
        
        dist = (
            z_choked.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_choked @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = torch.argmin(dist, dim=1) # [B]
        
        z_q = self.emb[idx] # [B, 16]
        
        # Straight-through estimator for backprop
        z_st = z_choked + (z_q - z_choked).detach()
        return z_st, idx

class System1Coprocessor(nn.Module):
    """
    System 1: The Logical Pre-Frontal Cortex.
    Takes continuous state from System 2, outputs a discrete matrix constraint.
    """
    def __init__(self, hidden_size: int, max_prompt_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook = BlankSlateCodebook(hidden_size=hidden_size)
        
        self.num_pointers = 2 # e.g., Ptr1, Ptr2
        self.pointer_heads = nn.ModuleList([
            nn.Linear(hidden_size, max_prompt_len) for _ in range(self.num_pointers)
        ])
        
    def forward(self, call_advisor_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates: [OP, Ptr1, Ptr2]
        Returns:
            choked_op_state: [B, 16] (Straight-through syntax representation)
            pointer_indices: [B, 2] (Indices into S2 prompt)
        """
        # 1. Operator Selection (Syntax Choke)
        z_st, op_idx = self.codebook.quantize(call_advisor_state)
        
        # 2. Hard Pointers
        ptr_indices_list = []
        for head in self.pointer_heads:
            logits = head(call_advisor_state)
            idx = torch.argmax(logits, dim=-1)
            ptr_indices_list.append(idx)
            
        pointer_indices = torch.stack(ptr_indices_list, dim=1) # [B, 2]
        
        return z_st, pointer_indices

class InterleavedRouter(nn.Module):
    """
    The Synchronous Loop: System 2 -> System 1 -> System 2.
    """
    def __init__(self, s1_coprocessor: System1Coprocessor, hidden_size: int):
        super().__init__()
        self.s1 = s1_coprocessor
        # Maps the discrete Lojbanic matrix back into System 2's residual stream
        self.injection_proj = nn.Linear(s1_coprocessor.codebook.choke_dim + (hidden_size * s1_coprocessor.num_pointers), hidden_size)
        
    def route_and_inject(self, s2_prompt_embeddings: torch.Tensor, call_advisor_state: torch.Tensor) -> torch.Tensor:
        """
        Step 3-5: S2 state routes to S1, S1 generates matrix, Matrix is injected.
        """
        b = call_advisor_state.shape[0]
        
        # 1. System 1 processing (Discrete Choke)
        z_st, pointer_indices = self.s1(call_advisor_state) # z_st: [B, 16], ptrs: [B, 2]
        
        # 2. Resolution: Re-hydrate Pointers using System 2's semantic cache
        borrowed_entities = []
        for i in range(self.s1.num_pointers):
            # Gather exact continuous tensor for the borrowed noun
            borrowed_v = []
            for batch_idx in range(b):
                idx = min(pointer_indices[batch_idx, i].item(), s2_prompt_embeddings.shape[1] - 1)
                borrowed_v.append(s2_prompt_embeddings[batch_idx, idx, :])
            borrowed_entities.append(torch.stack(borrowed_v))
            
        # 3. Assemble the Lojbanic Constraint Matrix
        # [Syntax Vector (16d)] + [Semantics 1 (896d)] + [Semantics 2 (896d)]
        matrix_components = [z_st] + borrowed_entities
        assembled_matrix = torch.cat(matrix_components, dim=-1)
        
        # 4. Inject back into System 2's residual stream
        injected_constraint = self.injection_proj(assembled_matrix)
        
        return injected_constraint
