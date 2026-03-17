from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCEForge(nn.Module):
    """
    M9 Contrastive NLI Engine.
    Physically pulls valid logical graphs closer to English semantic states 
    and repels hallucinated/invalid graphs.
    """
    def __init__(self, hidden_size: int, temp: float = 0.07):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temp
        # Projection heads to align the two modalities before contrastive loss
        self.premise_proj = nn.Linear(hidden_size, hidden_size)
        self.hypothesis_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, premise_states: torch.Tensor, pos_hypotheses: torch.Tensor, neg_hypotheses: torch.Tensor) -> torch.Tensor:
        """
        Calculates the InfoNCE loss for a batch of premises against positive and negative hypotheses.
        
        Args:
            premise_states: [B, H] (System 2's continuous English reasoning)
            pos_hypotheses: [B, H] (System 1's valid Lojban graph)
            neg_hypotheses: [B, N_neg, H] (System 1's hallucinated graphs)
            
        Returns:
            loss: Scalar InfoNCE loss
        """
        b = premise_states.shape[0]
        n_neg = neg_hypotheses.shape[1]
        
        # 1. Project into joint contrastive space
        p = F.normalize(self.premise_proj(premise_states), dim=-1) # [B, H]
        h_pos = F.normalize(self.hypothesis_proj(pos_hypotheses), dim=-1) # [B, H]
        h_neg = F.normalize(self.hypothesis_proj(neg_hypotheses), dim=-1) # [B, N_neg, H]
        
        # 2. Calculate similarities
        # Positive score: dot product of premise and positive hypothesis
        pos_score = (p * h_pos).sum(dim=-1, keepdim=True) # [B, 1]
        
        # Negative scores: dot product of premise and all negative hypotheses
        # p: [B, 1, H], h_neg: [B, N_neg, H]
        neg_scores = torch.bmm(h_neg, p.unsqueeze(-1)).squeeze(-1) # [B, N_neg]
        
        # 3. Concatenate scores: [pos, neg1, neg2...]
        logits = torch.cat([pos_score, neg_scores], dim=1) # [B, 1 + N_neg]
        logits = logits / self.temperature
        
        # 4. InfoNCE Loss: The correct answer is always index 0
        labels = torch.zeros(b, dtype=torch.long, device=premise_states.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss

class M9System1(nn.Module):
    """
    System 1: The Logical Graph Builder.
    Restored to full 896D bandwidth to process S2 states.
    """
    def __init__(self, hidden_size: int, codebook_size: int = 2000, max_prompt_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        
        # Full bandwidth codebook
        self.emb = nn.Parameter(torch.empty(codebook_size, hidden_size))
        torch.nn.init.normal_(self.emb, mean=0.0, std=0.02)
        
        # Dynamic Pointer Heads
        self.num_pointers = 2
        self.pointer_heads = nn.ModuleList([
            nn.Linear(hidden_size, max_prompt_len) for _ in range(self.num_pointers)
        ])
        
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Full 896D quantization (Gumbel-Softmax used in training loop)"""
        dist = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        idx = torch.argmin(dist, dim=1)
        z_q = self.emb[idx]
        return z + (z_q - z).detach()
        
    def build_graph(self, s2_premise_state: torch.Tensor, prompt_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes S2 reasoning state and builds the Lojban AST graph.
        Returns:
            op_state: [B, H]
            ptr_logits: [B, 2, max_prompt_len]
        """
        # 1. Operator Selection (Full 896D)
        op_state = self.quantize(s2_premise_state)
        
        # 2. Hard Pointers
        ptr_logits_list = []
        for head in self.pointer_heads:
            logits = head(s2_premise_state)
            mask = torch.full_like(logits, -float('inf'))
            mask[:, :prompt_len] = 0.0
            ptr_logits_list.append(logits + mask)
            
        ptr_logits = torch.stack(ptr_logits_list, dim=1)
        return op_state, ptr_logits
