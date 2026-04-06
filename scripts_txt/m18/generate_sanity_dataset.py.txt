import json
from pathlib import Path

def generate_sanity_dataset():
    samples = [
        {"prompt": "The apple is on the table. The pear is on the floor. Question: Where is the apple?", "answer": "table"},
        {"prompt": "The book is in the bag. The pen is on the desk. Question: Where is the book?", "answer": "bag"},
        {"prompt": "The cat is under the chair. The dog is on the sofa. Question: Where is the cat?", "answer": "chair"},
        {"prompt": "The key is in the drawer. The wallet is on the shelf. Question: Where is the key?", "answer": "drawer"},
        {"prompt": "The milk is in the fridge. The bread is on the counter. Question: Where is the milk?", "answer": "fridge"},
        {"prompt": "The car is in the garage. The bike is on the driveway. Question: Where is the car?", "answer": "garage"},
        {"prompt": "The phone is on the charger. The laptop is on the table. Question: Where is the phone?", "answer": "charger"},
        {"prompt": "The shoes are in the box. The socks are on the bed. Question: Where are the shoes?", "answer": "box"},
        {"prompt": "The bird is in the cage. The fish is in the tank. Question: Where is the bird?", "answer": "cage"},
        {"prompt": "The painting is on the wall. The mirror is on the door. Question: Where is the painting?", "answer": "wall"}
    ]
    
    output_path = Path("artifacts/datasets/sanity_check_v1.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"Sanity dataset generated at {output_path}")

if __name__ == "__main__":
    generate_sanity_dataset()
