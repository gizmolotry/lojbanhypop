from __future__ import annotations

import argparse
import torch
import zmq
import json
import time
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M9 Harvester: Extracts English Semantic Premises.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize ZeroMQ PUSH socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://127.0.0.1:{args.port}")
    print(f"Harvester connected to Forge on port {args.port}")

    # 2. Load Base Model (System 2)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    model.eval()

    # 3. Load Diverse Dataset
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M9 HARVESTER ACTIVE ---")
    
    for item in train_ds:
        # Generate English Reasoning Workspace
        cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # The 'Premise': Last layer hidden state at the last prompt token
            premise_state = outputs.hidden_states[-1][:, -1, :].cpu() # [1, 896]
            
        # Payload for the Forge
        payload = {
            "premise": premise_state.tolist(),
            "prompt_len": inputs.input_ids.shape[1],
            "target": item.answer,
            "metadata": {"prompt": item.prompt}
        }
        
        socket.send_json(payload)
        print(f"Pushed premise for puzzle: {item.prompt[:40]}...")
        
        # Rate limit the harvester to prevent overwhelming the Forge during localhost test
        time.sleep(0.1)

if __name__ == "__main__":
    main()
