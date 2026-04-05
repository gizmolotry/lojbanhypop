from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Tuple, List

class M11ProvenanceManifold(nn.Module):
    """
    M11 Phase 1: Separated Codebooks with Provenance Tags.
    Physically separates Predicates, Gates, and Pointers.
    """
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. The Three Codebooks
        self.gismu_emb = nn.Embedding(2000, hidden_size) # Predicates
        self.cmavo_emb = nn.Embedding(50, hidden_size)   # Logic Gates
        self.judri_emb = nn.Embedding(128, hidden_size)  # Positional Pointers
        
        # 2. Provenance Tags (Learned Semantic Flavor)
        self.type_emb = nn.Embedding(3, hidden_size) # 0: gismu, 1: cmavo, 2: judri

    def get_vector(self, idx: torch.Tensor, token_type: int) -> torch.Tensor:
        """E_final = E_dict(ID) + E_type(Type)"""
        if token_type == 0:
            dict_vec = self.gismu_emb(idx)
        elif token_type == 1:
            dict_vec = self.cmavo_emb(idx)
        else:
            dict_vec = self.judri_emb(idx)
            
        flavor = self.type_emb(torch.tensor([token_type], device=idx.device))
        return dict_vec + flavor

from .taxonomy import get_arity

class M11HyperModulator(nn.Module):
    """
    M11 Phase 2: The Role Basis Generator (R_g).
    Generates a gismu-specific Role Tensor [max_slots, H].
    """
    def __init__(self, hidden_size: int, max_slots: int = 9):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_slots = max_slots
        # Expansion to Role Basis
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * max_slots),
            nn.Sigmoid() # Bound scaling to [0, 1] for relative weighting
        )

    def forward(self, gismu_vec: torch.Tensor) -> torch.Tensor:
        """
        Returns: R_g of shape [B, max_slots, H]
        """
        b = gismu_vec.shape[0]
        role_basis = self.net(gismu_vec).view(b, self.max_slots, self.hidden_size)
        return role_basis

class M11CmavoHead(nn.Module):
    """
    Family B: Structural / Logical Operators.
    Operates over already-built graph fragments (subgraphs).
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.op_proj = nn.Linear(hidden_size, hidden_size)
        self.subgraph_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, cmavo_vec: torch.Tensor, subgraph_state: torch.Tensor) -> torch.Tensor:
        # z_logic = f(cmavo_vec, subgraph_summary)
        return torch.tanh(self.op_proj(cmavo_vec) + self.subgraph_proj(subgraph_state))

class M9System1(nn.Module):
    """
    System 1: The M11 Hyper-Modulated Symbiote with Local Role Geometry.
    """
    def __init__(self, hidden_size: int = 896, max_prompt_len: int = 128, max_slots: int = 9):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_prompt_len = max_prompt_len
        self.max_slots = max_slots
        
        self.manifold = M11ProvenanceManifold(hidden_size=hidden_size)
        self.hyper_mod = M11HyperModulator(hidden_size=hidden_size, max_slots=max_slots)
        
        # Policy Heads
        self.op_head = nn.Linear(hidden_size, 2000) # Selects from gismu_emb
        self.cmavo_head = M11CmavoHead(hidden_size=hidden_size)
        
    def build_graph(self, s2_state: torch.Tensor, prompt_embs: torch.Tensor, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamic PointerBind: Projects prompt tokens onto the generated Role Basis.
        prompt_embs: [B, Seq_L, H]
        """
        b, seq_l, h = prompt_embs.shape
        
        # 1. Operator Selection
        op_logits = self.op_head(s2_state)
        op_idx = torch.argmax(op_logits, dim=-1)
        op_vector = self.manifold.get_vector(op_idx, token_type=0)
        
        # 2. Local Role Geometry Unfolding (R_g)
        role_basis = self.hyper_mod(op_vector) # [B, max_slots, H]
        
        # 3. Dynamic PointerBind
        # We calculate similarity between each Role Vector and every Prompt Token
        # logits: [B, max_slots, Seq_L]
        role_logits = torch.matmul(role_basis, prompt_embs.transpose(-1, -2))
        
        # Masking for arity and prompt length
        arity = get_arity(op_idx[0].item()) # Slot-based mask
        arity_mask = torch.ones(b, self.max_slots, seq_l, device=s2_state.device)
        for i in range(self.max_slots):
            if i >= arity:
                arity_mask[:, i, :] = -float('inf')
        
        # apply Gumbel-Softmax over sequence dimension
        x_probs = F.gumbel_softmax(role_logits + arity_mask, tau=tau, hard=True, dim=-1)
        
        return op_vector, x_probs, op_idx

class M9HardNegativeGenerator:
    def __init__(self, manifold: M11ProvenanceManifold):
        self.manifold = manifold

    def generate(self, premise: torch.Tensor, pos_op_vec: torch.Tensor, ptr_probs: torch.Tensor, op_idx: torch.Tensor) -> torch.Tensor:
        b = pos_op_vec.shape[0]
        neg_hyps = []
        neg_hyps.append(pos_op_vec * 0.95) 
        arity = get_arity(op_idx[0].item())
        arity_scale = 1.05 if arity < 3 else 0.95
        neg_hyps.append(pos_op_vec * arity_scale)
        for _ in range(2):
            rand_idx = torch.randint(100, 2000, (b,), device=pos_op_vec.device)
            neg_hyps.append(self.manifold.get_vector(rand_idx, token_type=0))
        for _ in range(1):
            rand_idx = torch.randint(0, 50, (b,), device=pos_op_vec.device)
            neg_hyps.append(self.manifold.get_vector(rand_idx, token_type=1))
        return torch.stack(neg_hyps, dim=1)

class InfoNCEForge(nn.Module):
    def __init__(self, hidden_size: int = 896, temp: float = 0.07):
        super().__init__()
        self.temperature = temp
        self.premise_proj = nn.Linear(hidden_size, hidden_size)
        self.hypothesis_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, premise: torch.Tensor, pos_hyp: torch.Tensor, neg_hyps: torch.Tensor) -> torch.Tensor:
        b = premise.shape[0]
        p = F.normalize(self.premise_proj(premise), dim=-1)
        h_pos = F.normalize(self.hypothesis_proj(pos_hyp), dim=-1)
        h_neg = F.normalize(self.hypothesis_proj(neg_hyps), dim=-1)
        pos_score = (p * h_pos).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(h_neg, p.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([pos_score, neg_scores], dim=1) / self.temperature
        labels = torch.zeros(b, dtype=torch.long, device=premise.device)
        return F.cross_entropy(logits, labels)

class MoVGate(nn.Module):
    def __init__(self, hidden_size: int, base_vocab_size: int):
        super().__init__()
        self.router = nn.Linear(hidden_size, 1)
        self.base_vocab_size = base_vocab_size

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.router(hidden_states))
        w_eng = lm_head_weight[:self.base_vocab_size, :]
        w_loj = lm_head_weight[self.base_vocab_size:, :]
        logits_eng = F.linear(hidden_states, w_eng)
        logits_loj = F.linear(hidden_states, w_loj)
        prob_eng = F.softmax(logits_eng, dim=-1) * gate
        prob_loj = F.softmax(logits_loj, dim=-1) * (1.0 - gate)
        return torch.log(torch.cat([prob_eng, prob_loj], dim=-1) + 1e-12)
