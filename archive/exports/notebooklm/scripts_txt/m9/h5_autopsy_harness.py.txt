import torch
import torch.nn as nn
import math
from pathlib import Path
import sys

# Define the exact H5 class identified from the legacy codebase
class AdvisorCrossAttentionAdapter(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, advisor_states: torch.Tensor, advisor_ids: torch.Tensor) -> torch.Tensor:
        b, l, h = advisor_states.shape
        if l % 3 != 0:
            q = self.q_proj(hidden_states)
            k = self.k_proj(advisor_states)
            v = self.v_proj(advisor_states)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(h))
            attn = torch.softmax(scores, dim=-1)
            return self.out_proj(torch.matmul(attn, v))

        triplets_state = advisor_states.view(b, l // 3, 3, h)
        triplets_ids = advisor_ids.view(b, l // 3, 3)
        v_projected = self.v_proj(triplets_state)
        
        effective_v = []
        for i in range(l // 3):
            rel_id = triplets_ids[:, i, 0]
            v_var1 = v_projected[:, i, 1]
            v_var2 = v_projected[:, i, 2]
            v_rel = v_projected[:, i, 0]

            is_and = (rel_id == 0).float().view(-1, 1)
            is_or = (rel_id == 1).float().view(-1, 1)
            is_not = (rel_id == 2).float().view(-1, 1)
            is_implies = (rel_id == 3).float().view(-1, 1)
            is_xor = (rel_id == 4).float().view(-1, 1)
            is_learned = (rel_id >= 5).float().view(-1, 1)

            v_and = torch.min(v_var1, v_var2)
            v_or = torch.max(v_var1, v_var2)
            v_not = -v_var1
            v_implies = torch.max(-v_var1, v_var2)
            v_xor = torch.abs(v_var1 - v_var2)

            res = (is_and * v_and + is_or * v_or + is_not * v_not + is_implies * v_implies + is_xor * v_xor + is_learned * v_rel)
            effective_v.append(res.unsqueeze(1))

        v_final = torch.cat(effective_v, dim=1)
        k_final = self.k_proj(triplets_state[:, :, 0, :])
        q = self.q_proj(hidden_states)
        scores = torch.matmul(q, k_final.transpose(-1, -2)) / math.sqrt(float(h))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_final)
        return self.out_proj(context)

def perform_autopsy():
    print("\n--- H5 AUTOPSY HARNESS ---")
    
    ckpt_path = Path("runs/i_series/20260302_172603/h5_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        return
        
    print(f"1. Loading Checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    
    print("\n2. Inspecting Checkpoint Keys:")
    print(f"   Top-level keys: {list(sd.keys())}")
    print(f"   Advisor Adapter keys: {list(sd['advisor_adapter_state'].keys())}")
    
    print("\n3. Instantiating Legacy AdvisorCrossAttentionAdapter...")
    hidden_size = 896
    adapter = AdvisorCrossAttentionAdapter(hidden_size=hidden_size)
    
    print("\n4. Loading Weights (Strict Mode)...")
    try:
        adapter.load_state_dict(sd['advisor_adapter_state'], strict=True)
        print("   SUCCESS: Weights loaded cleanly without mismatch.")
    except Exception as e:
        print(f"   FAIL: {e}")
        return
        
    print("\n5. Testing Tensor Contract (Fake Forward Pass)...")
    b, t_seq, l_seq = 1, 10, 3 # Batch=1, Text_Len=10, Logic_Len=3
    
    # hidden_states: The continuous English reasoning stream
    hidden_states = torch.randn(b, t_seq, hidden_size)
    
    # advisor_states: The 896D embeddings of the Lojban logic
    advisor_states = torch.randn(b, l_seq, hidden_size)
    
    # advisor_ids: The discrete token IDs of the Lojban logic
    advisor_ids = torch.randint(0, 2000, (b, l_seq))
    
    print(f"   Input 'hidden_states' shape: {hidden_states.shape}")
    print(f"   Input 'advisor_states' shape: {advisor_states.shape}")
    print(f"   Input 'advisor_ids' shape: {advisor_ids.shape}")
    
    try:
        output = adapter(hidden_states, advisor_states, advisor_ids)
        print(f"\n   SUCCESS: Output shape: {output.shape}")
        print("   Contract Verified: The H5 adapter is a Cross-Attention mechanism that takes")
        print("   continuous English states as Queries and continuous Lojban embeddings as Keys/Values.")
    except Exception as e:
        print(f"\n   FAIL: Forward pass crashed: {e}")

if __name__ == "__main__":
    perform_autopsy()
