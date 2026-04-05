from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class M18SalienceSelector(nn.Module):
    """
    M18-v0 Salience Selector: Identifies Top-K structural positions.
    """
    def __init__(self, hidden_size: int = 896, top_k: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        
        self.salience_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, Seq_L, H]
        Returns: 
            scores: [B, Seq_L]
            top_k_indices: [B, K]
            top_k_mask: [B, Seq_L] (binary mask of top-k positions)
        """
        b, l, h = hidden_states.shape
        scores = self.salience_head(hidden_states).squeeze(-1) # [B, Seq_L]
        
        # Softmax over sequence length to identify relative importance
        # probabilities = F.softmax(scores, dim=-1)
        
        # Top-K selection
        top_k_vals, top_k_indices = torch.topk(scores, k=min(self.top_k, l), dim=-1)
        
        # Create binary mask
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_k_indices, 1.0)
        
        return scores, top_k_indices, mask

def compute_salience_metrics(pred_scores: torch.Tensor, gold_indices: torch.Tensor) -> Dict[str, float]:
    """
    Mandatory Separate Tracking for Gold Supervision.
    pred_scores: [B, Seq_L]
    gold_indices: [B, K_gold]
    """
    b, l = pred_scores.shape
    _, pred_indices = torch.topk(pred_scores, k=gold_indices.shape[1], dim=-1)
    
    # Calculate Precision/Recall
    # (Since pred and gold have same count, P=R)
    correct = 0
    total = gold_indices.numel()
    
    for i in range(b):
        p_set = set(pred_indices[i].tolist())
        g_set = set(gold_indices[i].tolist())
        correct += len(p_set.intersection(g_set))
        
    return {
        "precision": correct / total,
        "recall": correct / total
    }
