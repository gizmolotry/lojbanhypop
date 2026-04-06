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
from peft import PeftModel
from src.lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M9 Harvester: Extracts English Semantic Premises.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize ZeroMQ sockets
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://127.0.0.1:{args.port}")
    
    # PULL socket for the logic injection (The Final Circuit)
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect(f"tcp://127.0.0.1:{args.port + 1}")
    
    # 2. Load Base Model & Adapter (System 2)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    
    # Check if this is an adapter directory
    adapter_config_path = Path(args.base_model) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            config = json.load(f)
            base_model_path = config["base_model_name_or_path"]
        
        print(f"Harvester: Loading base model {base_model_path} and adapter {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=args.local_files_only, device_map="auto")
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, args.base_model)
    else:
        print(f"Harvester: Loading monolithic model {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    
    model.eval()

    # 3. Incremental Curriculum Generation
    # We generate items one-by-one to prevent initialization hangs
    print(f"\n--- M9 HARVESTER: CURRICULUM MODE ---")

    total_steps = 0
    for tier in ["easy", "medium", "hard"]:
        print(f"Harvester: Starting {tier.upper()} curriculum...")
        
        # We process in large chunks to ensure generate_dataset works efficiently
        for chunk_idx in range(34): # ~3400 samples total
            print(f"Harvester: Generating {tier.upper()} chunk {chunk_idx+1}...")
            ds = generate_dataset(size=300, seed=42 + chunk_idx, difficulty_tier=tier, profile="diverse_v3")
            if not ds: continue
            
            for item in ds:
                # THE SEMANTIC SCAN
                cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
                inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    premise_state = outputs.hidden_states[-1][:, -1, :].detach().cpu()
                    
                # PUSH to Forge
                payload = {
                    "premise": premise_state.tolist(),
                    "prompt_len": inputs.input_ids.shape[1],
                    "trace": list(item.trace),
                    "target": item.answer,
                    "tier": tier
                }
                push_socket.send_json(payload)
                
                if (total_steps + 1) % 10 == 0:
                    print(f"Harvester: Pushed step {total_steps+1} ({tier.upper()})")
                
                # Non-blocking wait for injection
                try:
                    if pull_socket.poll(timeout=10):
                        pull_socket.recv_json()
                except zmq.ZMQError:
                    pass
                    
                total_steps += 1
                if args.max_steps and total_steps >= args.max_steps:
                    print(f"Harvester: Reached max steps ({args.max_steps}). Exiting.")
                    return
if __name__ == "__main__":
    main()
