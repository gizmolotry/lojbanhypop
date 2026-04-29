import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def extract_cmavo_anchors(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    embeddings = model.get_input_embeddings().weight.data

    # Rigid logical English counterparts for cmavo grounding
    cmavo_list = [
        "AND", "OR", "XOR", "NOT", "FALSE", "TRUE", 
        "START", "END", "GROUP", "IF", "THEN", "ELSE",
        "ALL", "SOME", "NONE", "EXISTS", "EVERY",
        "(", ")", "[", "]", "{", "}",
        "EQUALS", "IN", "CONTAINS", "CAUSES", "LEADS"
    ]

    anchors = []
    for word in cmavo_list:
        ids = tokenizer.encode(word, add_special_tokens=False)
        vec = embeddings[ids].mean(dim=0)
        anchors.append(vec.tolist())

    output_path = Path("src/lojban_evolution/m9/cmavo_anchors.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(anchors, f)
    print(f"Extracted {len(anchors)} cmavo anchors to {output_path}")

if __name__ == "__main__":
    extract_cmavo_anchors("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
