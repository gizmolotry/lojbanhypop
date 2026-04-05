from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class M18RelationalInterpreter(nn.Module):
    """
    M18-v0 Relational Interpreter: Induces directed, typed relations.
    """
    def __init__(
        self, 
        hidden_size: int = 896, 
        num_relations: int = 8, 
        ontology: str = "U"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.ontology = ontology
        
        # Edge Function Components
        # Input to edge MLP: [z_i, z_j, z_i * z_j, z_i - z_j]
        mlp_input_size = hidden_size * 4
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations)
        )

    def forward(self, salient_embs: torch.Tensor) -> torch.Tensor:
        """
        salient_embs: [B, K, H] (Hidden states at top-K positions)
        Returns: 
            adj_matrix: [B, K, K, Num_Relations]
        """
        b, k, h = salient_embs.shape
        
        # 1. Expand to pairwise combinations [B, K, K, H]
        z_i = salient_embs.unsqueeze(2).expand(-1, -1, k, -1)
        z_j = salient_embs.unsqueeze(1).expand(-1, k, -1, -1)
        
        # 2. Compute interaction terms
        hadamard = z_i * z_j
        diff = z_i - z_j
        
        # 3. Concatenate for edge MLP [B, K, K, 4H]
        interaction = torch.cat([z_i, z_j, hadamard, diff], dim=-1)
        
        # 4. Induce relations
        logits = self.edge_mlp(interaction) # [B, K, K, R]
        
        if self.ontology == "L":
            # Apply Logebonic Constraints (e.g., asymmetry, role exclusivity)
            # For v0, we use a simple Softmax over types
            # (In v1, we will implement the arity/directionality rules)
            adj_matrix = F.softmax(logits, dim=-1)
        else:
            # Unconstrained Latent Ontology
            adj_matrix = F.softmax(logits, dim=-1)
            
        return adj_matrix

class M18BiasCompiler(nn.Module):
    """
    M18-v0 Bias Compiler: Translates graph into attention logit biases.
    """
    def __init__(self, num_relations: int, num_heads: int, hidden_size: int):
        super().__init__()
        # Type-specific head weights [R, Layers, Heads]
        # For v0, we implement vector gates over relation types per head
        self.relation_head_weights = nn.Parameter(torch.randn(num_relations, num_heads) * 0.02)

    def compile(self, adj_matrix: torch.Tensor, top_k_indices: torch.Tensor, seq_l: int) -> torch.Tensor:
        """
        adj_matrix: [B, K, K, R]
        top_k_indices: [B, K]
        Returns: 
            bias_tensor: [B, Heads, Seq_L, Seq_L]
        """
        b, k, _, r = adj_matrix.shape
        num_heads = self.relation_head_weights.shape[1]
        
        # 1. Project relations to heads [B, Heads, K, K]
        head_adj = torch.einsum("bijk,kh->bhij", adj_matrix, self.relation_head_weights)
        
        # 2. Expand to full sequence dimensions [B, Heads, Seq_L, Seq_L]
        full_bias = torch.zeros(b, num_heads, seq_l, seq_l, device=adj_matrix.device, dtype=adj_matrix.dtype)
        
        # PHYSICAL GROUNDING: 
        # We don't just bias KxK. We bias FROM the entire sequence TO the K salient tokens.
        # This tells the model: "Everyone, look at these logical anchors."
        
        # Simple projection for v0: average the head_adj across query dimension
        # resulting in [B, Heads, K] (Relative importance of each anchor)
        anchor_salience = head_adj.mean(dim=2) 
        
        for batch_idx in range(b):
            indices = top_k_indices[batch_idx]
            # Set columns corresponding to salient indices for ALL rows
            # full_bias: [Heads, Seq_L, K_cols]
            full_bias[batch_idx, :, :, indices] = anchor_salience[batch_idx].unsqueeze(1)
            
        return full_bias
