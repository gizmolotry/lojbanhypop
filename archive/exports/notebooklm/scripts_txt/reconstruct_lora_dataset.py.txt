from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.experiment import generate_dataset

def format_training_text(prompt: str, symbolic_trace: list[str], answer: str) -> str:
    trace_line = " ".join(symbolic_trace)
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
        f"QUESTION: {prompt}\n"
        f"TRACE: {trace_line}\n"
        f"ANSWER: {answer}"
    )

def main():
    output_path = Path("runs/lora_sft_dataset.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Seeds used in the audit report for training
    seeds = [7, 11]
    dataset_size = 1000
    
    rows = []
    for seed in seeds:
        problems = generate_dataset(size=dataset_size, seed=seed)
        for p in problems:
            text = format_training_text(p.prompt, p.trace, p.answer)
            rows.append({
                "text": text,
                "problem_id": p.problem_id,
                "mode": "phase3_fewshot" 
            })
            
    # Deduplicate by text
    seen = set()
    unique_rows = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_rows.append(r)
            
    with output_path.open("w", encoding="utf-8") as f:
        for row in unique_rows:
            f.write(json.dumps(row) + "\n")
            
    print(f"Wrote {len(unique_rows)} examples to {output_path}")

if __name__ == "__main__":
    main()
