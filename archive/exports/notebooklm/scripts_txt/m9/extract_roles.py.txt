import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path

def extract_role_anchors(model_path, adapter_path):
    print(f"Extracting role anchors from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # STAGE THE BACKBONE (151701)
    backbone = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    backbone.resize_token_embeddings(151701)
    
    model = PeftModel.from_pretrained(backbone, adapter_path)
    
    # We look for the learned vectors of our positional tokens
    role_tokens = [
        "BIND_E1", "BIND_E2", "BIND_OBJ", "BIND_LOC",
        "BIND_AGENT_E1", "BIND_AGENT_E2", "BIND_A", "BIND_B", "BIND_C",
        "ANS_E1", "ANS_E2", "ANS_LOC1", "ANS_LOC2", "ANS_YES", "ANS_NO"
    ]
    
    anchors = []
    embeddings = model.get_input_embeddings().weight.data
    
    for token in role_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            # Take the mean of the tokens (usually just one for our special tokens)
            vec = embeddings[ids].mean(dim=0)
            anchors.append(vec.tolist())
            
    output_path = Path("src/lojban_evolution/m9/role_anchors.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(anchors, f)
    print(f"Extracted {len(anchors)} positional role anchors to {output_path}")

if __name__ == "__main__":
    extract_role_anchors(
        "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
        "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5"
    )
