import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def resize_tokenizer(model_path: str, output_path: str):
    print(f"Loading base model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")

    # Add 2,000 new tokens <loj_0> through <loj_1999> and the <symbiote> token
    new_tokens = [f"<loj_{i}>" for i in range(2000)] + ["<symbiote>"]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens to the vocabulary.")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

    # Save the resized model and tokenizer
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    print(f"Resized model and tokenizer saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize Qwen tokenizer and embeddings for Discrete SFT baseline.")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the base Qwen model.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the resized model.")
    args = parser.parse_args()

    resize_tokenizer(args.base_model, args.output_path)
