import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class M10bTranslationAdapter(nn.Module):
    """
    M10b: Generative Cross-Attention Adapter.
    Translates frozen 10-slot Lojban logic into English answer tokens.
    Input:
        h_english: [B, L, H] (System 2 hidden states)
        h_lojban:  [B, 10, H] (System 1 logic tensors)
    Output:
        h_final: [B, L, H] (Injected hidden states)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Gating parameter alpha: starts at 0 to prevent initial shock
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_english: torch.Tensor, h_lojban: torch.Tensor) -> torch.Tensor:
        b, l, h = h_english.shape
        n_slots = h_lojban.shape[1]
        
        # 1. Project Q, K, V
        q = self.w_q(h_english) # [B, L, H]
        k = self.w_k(h_lojban)  # [B, 10, H]
        v = self.w_v(h_lojban)  # [B, 10, H]
        
        # 2. Scaled Dot-Product Cross-Attention
        # Queries are English, Keys/Values are Lojban
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(h) # [B, L, 10]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_probs, v) # [B, L, H]
        
        # 3. Differentiable Handoff
        delta = self.out_proj(context)
        h_final = h_english + (self.alpha * delta)
        
        return h_final
