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

class M11HyperModulator(nn.Module):
    """
    M11 Phase 2: The Terna Generator.
    Generates dynamic scaling vectors based on the active predicate.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid() # Bound scaling to [0, 1]
        )

    def forward(self, gismu_vec: torch.Tensor) -> torch.Tensor:
        return self.net(gismu_vec)

class M9System1(nn.Module):
    """
    System 1: The M11 Hyper-Modulated Symbiote.
    """
    def __init__(self, hidden_size: int = 896, max_prompt_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_prompt_len = max_prompt_len
        self.num_x_slots = 9 
        
        self.manifold = M11ProvenanceManifold(hidden_size=hidden_size)
        self.hyper_mod = M11HyperModulator(hidden_size=hidden_size)
        
        # Policy Heads
        self.op_head = nn.Linear(hidden_size, 2000) # Selects from gismu_emb
        self.x_heads = nn.ModuleList([
            nn.Linear(hidden_size, max_prompt_len) for _ in range(self.num_x_slots)
        ])
        
    def build_graph(self, s2_state: torch.Tensor, prompt_len: int, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = s2_state.shape[0]
        
        # 1. Operator Selection (gismu)
        op_logits = self.op_head(s2_state)
        op_idx = torch.argmax(op_logits, dim=-1)
        op_vector = self.manifold.get_vector(op_idx, token_type=0)
        
        # 2. Hyper-Modulation (The Place Structure Unfolder)
        # Generate dynamic scaling vector based on the verb
        delta_gismu = self.hyper_mod(op_vector)
        
        # 3. Modulated Pointer Projections
        x_probs_list = []
        # We modulate the English context by the Lojbanic place structure
        modulated_state = s2_state * delta_gismu
        
        for head in self.x_heads:
            logits = head(modulated_state)
            mask = torch.full_like(logits, -float('inf'))
            mask[:, :prompt_len] = 0.0
            soft_ptr = F.gumbel_softmax(logits + mask, tau=tau, hard=True, dim=-1)
            x_probs_list.append(soft_ptr[:, :prompt_len])
            
        x_probs = torch.stack(x_probs_list, dim=1)
        return op_vector, x_probs, op_idx

class M9HardNegativeGenerator:
    """
    Adversarial Physics: Dynamically generates hard negatives to flood the InfoNCE denominator.
    """
    def __init__(self, manifold: M11ProvenanceManifold):
        self.manifold = manifold

    def generate(self, premise: torch.Tensor, pos_op_vec: torch.Tensor, ptr_probs: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B, N_neg, H]
        """
        b = pos_op_vec.shape[0]
        neg_hyps = []

        # 1. THE CAUSAL REVERSAL: Swap the pointers
        reversed_ptrs = ptr_probs.flip(dims=[1])
        
        # 2. THE NOISE INJECTION: Sample from the gismu (predicate) playground
        for _ in range(3):
            rand_idx = torch.randint(100, 2000, (b,), device=pos_op_vec.device)
            neg_hyps.append(self.manifold.get_vector(rand_idx, token_type=0))

        # 3. THE ARITY TRAP: Sample from the judri (pointer) space
        for _ in range(2):
            rand_idx = torch.randint(0, 128, (b,), device=pos_op_vec.device)
            neg_hyps.append(self.manifold.get_vector(rand_idx, token_type=2))

        return torch.stack(neg_hyps, dim=1) # [B, 5, H]

class InfoNCEForge(nn.Module):
    """
    M9 Contrastive NLI Engine.
    """
    def __init__(self, hidden_size: int = 896, temp: float = 0.07):
        super().__init__()
        self.temperature = temp
        self.premise_proj = nn.Linear(hidden_size, hidden_size)
        self.hypothesis_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, premise: torch.Tensor, pos_hyp: torch.Tensor, neg_hyps: torch.Tensor) -> torch.Tensor:
        """
        Calculates InfoNCE loss. correct answer is index 0.
        """
        b = premise.shape[0]
        p = F.normalize(self.premise_proj(premise), dim=-1)
        h_pos = F.normalize(self.hypothesis_proj(pos_hyp), dim=-1)
        h_neg = F.normalize(self.hypothesis_proj(neg_hyps), dim=-1)
        
        pos_score = (p * h_pos).sum(dim=-1, keepdim=True) # [B, 1]
        neg_scores = torch.bmm(h_neg, p.unsqueeze(-1)).squeeze(-1) # [B, N_neg]
        
        logits = torch.cat([pos_score, neg_scores], dim=1) / self.temperature
        labels = torch.zeros(b, dtype=torch.long, device=premise.device)
        return F.cross_entropy(logits, labels)

class MoVGate(nn.Module):
    """
    M9.5 Mixture-of-Vocabularies Gate.
    Dynamically routes probability mass between English and Lojban heads.
    """
    def __init__(self, hidden_size: int, base_vocab_size: int):
        super().__init__()
        self.router = nn.Linear(hidden_size, 1)
        self.base_vocab_size = base_vocab_size

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        """
        P(x) = g*Softmax(W_eng*h) + (1-g)*Softmax(W_loj*h)
        """
        # gate: [B, L, 1]
        gate = torch.sigmoid(self.router(hidden_states))
        
        # English head: [V_eng, H]
        w_eng = lm_head_weight[:self.base_vocab_size, :]
        # Lojban head: [V_loj, H]
        w_loj = lm_head_weight[self.base_vocab_size:, :]
        
        # Logits
        logits_eng = F.linear(hidden_states, w_eng)
        logits_loj = F.linear(hidden_states, w_loj)
        
        # Softmax blending
        prob_eng = F.softmax(logits_eng, dim=-1) * gate
        prob_loj = F.softmax(logits_loj, dim=-1) * (1.0 - gate)
        
        # Combined Log-Probs for NLLLoss compatibility
        # We return the log sum of probabilities
        return torch.log(torch.cat([prob_eng, prob_loj], dim=-1) + 1e-12)
