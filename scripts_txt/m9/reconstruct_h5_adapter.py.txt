from __future__ import annotations
import torch
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

class M9CouncilAdapter(torch.nn.Module):
    """
    M9.8 Restoration: Reconstructs the H5.3 Multi-Head Council.
    Uses the trained Intuitor weights to resolve Lojbanic tensors.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gain = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states: torch.Tensor, logic_states: torch.Tensor) -> torch.Tensor:
        # Cross-attention over the injected logic
        q = self.q_proj(hidden_states) # [B, L, H]
        k = self.k_proj(logic_states)  # [B, N, H]
        v = self.v_proj(logic_states)  # [B, N, H]
        
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hidden_states.shape[-1])
        probs = torch.softmax(attn, dim=-1)
        context = torch.matmul(probs, v)
        
        return self.out_proj(context) * self.gain

def reconstruct_full_body(ckpt_path, base_model_path, output_dir):
    print(f"\n--- M9.8 FULL-BODY TRANSPLANT INITIATED ---")
    device = "cpu"
    sd = torch.load(ckpt_path, map_location=device)
    hidden_size = 896
    
    # Initialize the Council
    council = M9CouncilAdapter(hidden_size=hidden_size)
    h5_sd = sd['advisor_adapter_state']
    
    # Surgical Mapping
    with torch.no_grad():
        council.q_proj.weight.copy_(h5_sd['q_proj.weight'])
        council.k_proj.weight.copy_(h5_sd['k_proj.weight'])
        council.v_proj.weight.copy_(h5_sd['v_proj.weight'])
        council.out_proj.weight.copy_(h5_sd['out_proj.weight'])
        print("Council Minds (Intuitor Heads) mapped successfully.")

    # 3. Physically Sync the 2000-token Lojban Dictionary
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    backbone = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")
    backbone.resize_token_embeddings(151701)
    tokenizer.add_tokens([f"<loj_{i}>" for i in range(2256)])
    backbone.resize_token_embeddings(len(tokenizer))
    
    with torch.no_grad():
        h5_emb = sd['codebook_state']['emb']
        backbone.get_input_embeddings().weight[151701:151701+2000] = h5_emb
        backbone.get_output_embeddings().weight[151701:151701+2000] = h5_emb
        print("Dictionary Sync: 2000 joint-optimized vectors hard-copied.")

    # 4. Save components
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(council.state_dict(), output_dir / "council_adapter.pt")
    backbone.save_pretrained(output_dir / "backbone")
    tokenizer.save_pretrained(output_dir / "backbone")
    print(f"\nFull-Body Transplant Components Saved to {output_dir}")

if __name__ == "__main__":
    src = "runs/i_series/20260302_172603/h5_checkpoint.pt"
    base = "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct"
    dst = "archive/results/m9/active/RESULTS_M9_SYNCED/full_body_dark_transplant"
    reconstruct_full_body(src, base, dst)
