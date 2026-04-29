import json
import argparse
from pathlib import Path

def process_dataset(input_jsonl: str, output_jsonl: str):
    print(f"Flattening logic traces from {input_jsonl} to {output_jsonl}...")
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            # Track 1: Use the pre-forged target_logic string from the unified corpus
            row["flattened_logic"] = row.get("target_logic", "").strip()
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Flattening complete. Processed {count} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten logic traces from the unified M14.5 corpus.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()
    process_dataset(args.input_jsonl, args.output_jsonl)
