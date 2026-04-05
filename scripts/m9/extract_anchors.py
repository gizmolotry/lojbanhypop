import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def extract_gismu_anchors(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    embeddings = model.get_input_embeddings().weight.data

    # Core Lojban concepts and logical primitives for grounding
    gismu_list = [
        "nenri", "rinka", "logji", "fatra", "klama", "citka", "nelci", "prami",
        "stidi", "krinu", "nixli", "nanmu", "gerku", "mlatu", "plise", "zdani",
        "cukta", "skami", "tivni", "karce", "blanu", "xunre", "pelxu", "crino",
        "skari", "clani", "tcmila", "cmene", "valsi", "gismu", "lujvo", "cmavo",
        "du'u", "su'u", "ka", "ni", "li", "lo", "le", "la", "ku", "vau",
        "and", "or", "not", "if", "then", "true", "false", "implies"
    ]

    anchors = []
    for word in gismu_list:
        ids = tokenizer.encode(word, add_special_tokens=False)
        # Mean pool the embeddings for the word
        vec = embeddings[ids].mean(dim=0)
        anchors.append(vec.tolist())

    output_path = Path("src/lojban_evolution/m9/gismu_anchors.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(anchors, f)
    print(f"Extracted {len(anchors)} anchors to {output_path}")

if __name__ == "__main__":
    extract_gismu_anchors("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
