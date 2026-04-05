from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from contextlib import contextmanager, nullcontext
import math

@contextmanager
def adapter_disabled(model):
    disable_ctx = None
    if hasattr(model, "disable_adapter"):
        disable_ctx = model.disable_adapter()
    elif hasattr(model, "disable_adapters"):
        disable_ctx = model.disable_adapters()
    if disable_ctx is None:
        with nullcontext():
            yield
    else:
        with disable_ctx:
            yield

class M8BlankSlateCodebook(nn.Module):
    """
    The M8 Blank Slate Codebook with Vector Choke.
    Dimension is bottlenecked to d=16.
    """
    def __init__(self, codebook_size: int = 2000, hidden_size: int = 896, choke_dim: int = 16):
        super().__init__()
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.choke_dim = choke_dim
        
        self.down_proj = nn.Linear(hidden_size, choke_dim)
        self.emb = nn.Parameter(torch.empty(codebook_size, choke_dim))
        torch.nn.init.normal_(self.emb, mean=0.0, std=0.02)
        
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h = z.shape
        z_choked = self.down_proj(z)
        
        dist = (
            z_choked.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_choked @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = torch.argmin(dist, dim=1)
        z_q = self.emb[idx]
        z_st = z_choked + (z_q - z_choked).detach()
        return z_st, idx

class OracleHead(nn.Module):
    """
    A single independent Oracle in the Council.
    """
    def __init__(self, hidden_size: int, codebook: M8BlankSlateCodebook, max_prompt_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook = codebook
        self.num_pointers = 2 # Simplification for M8 bedrock: [OP, Ptr1, Ptr2]
        
        self.pointer_heads = nn.ModuleList([
            nn.Linear(hidden_size, max_prompt_len) for _ in range(self.num_pointers)
        ])
        
    def forward(self, state: torch.Tensor, prompt_len: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generates a 16-dimensional syntax vector.
        """
        # Inject high-temperature latent noise for divergence (M8.1)
        noise = torch.randn_like(state) * temperature
        diverged_state = state + noise
        
        # 1. Operator Selection (Syntax Choke)
        z_st, _ = self.codebook.quantize(diverged_state) # [B, 16]
        
        # 2. Hard Pointers with Dynamic Masking (from M7)
        borrowed_entities = []
        for head in self.pointer_heads:
            logits = head(diverged_state)
            mask = torch.full_like(logits, -float('inf'))
            mask[:, :prompt_len] = 0.0
            masked_logits = logits + mask
            # For simplicity in M8 PoC, we use hard pointers but keep logic differentiable via straight-through if needed
            # Here we just want the index to 'borrow' the embedding
            ptr_idx = torch.argmax(masked_logits, dim=-1)
            
            # This logic should be handled by the router to reach into S2 embeddings
            # We return the index for now.
            
        return z_st # Just the operator syntax for this specific hypothesis

class CouncilOfOracles(nn.Module):
    """
    M8.1: Parallel Latent Broadcast.
    Instantiates N independent Oracle heads.
    """
    def __init__(self, hidden_size: int, num_oracles: int = 4, max_prompt_len: int = 128):
        super().__init__()
        self.num_oracles = num_oracles
        self.codebook = M8BlankSlateCodebook(hidden_size=hidden_size)
        self.oracles = nn.ModuleList([
            OracleHead(hidden_size, self.codebook, max_prompt_len) for _ in range(num_oracles)
        ])
        
    def forward(self, call_advisor_state: torch.Tensor, prompt_len: int, temperature: float = 0.5) -> torch.Tensor:
        """
        Returns Hypothesis Matrix: [B, N, 16]
        """
        hypotheses = []
        for oracle in self.oracles:
            hypotheses.append(oracle(call_advisor_state, prompt_len, temperature))
            
        return torch.stack(hypotheses, dim=1) # [B, N, 16]

class M8InterleavedRouter(nn.Module):
    """
    M8.2: Hypothesis Matrix Injection.
    """
    def __init__(self, council: CouncilOfOracles, hidden_size: int):
        super().__init__()
        self.council = council
        self.hidden_size = hidden_size
        
        # Map [N, 16] into a format System 2 can attend to
        # We can either project 16 up to H or let S2 cross-attend to 16d.
        # Following M7 logic, we project back to hidden_size.
        self.syntax_up_proj = nn.Linear(self.council.codebook.choke_dim, hidden_size)
        
    def route_hypotheses(self, call_advisor_state: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """
        Broadcast to Council, get Hypothesis Matrix [B, N, 16]
        """
        return self.council(call_advisor_state, prompt_len)

    def prepare_for_injection(self, hypothesis_matrix: torch.Tensor) -> torch.Tensor:
        """
        Projects each hypothesis back to hidden_size.
        Returns: [B, N, H]
        """
        return self.syntax_up_proj(hypothesis_matrix)
