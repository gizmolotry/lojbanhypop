import argparse
import json
import torch
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def run_sapir_whorf_audit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    configs = {
        "en": {
            "data": "artifacts/datasets/sanity_check_v1.jsonl",
            "instruction": "\nThink step-by-step before answering. The final answer should be a single word after 'Answer: '.",
            "answer_marker": "answer:",
            "label": "English"
        },
        "zh": {
            "data": "artifacts/datasets/sanity_check_zh_v1.jsonl",
            "instruction": "\n请逐步思考后再回答。最终答案应在“答案：”之后的一个词。",
            "answer_marker": "答案：",
            "label": "Chinese"
        }
    }

    results = {}

    for lang, cfg in configs.items():
        print(f"\n--- RUNNING {cfg['label'].upper()} SAPIR-WHORF BASELINE ---")
        with open(cfg["data"], "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        correct = 0
        total_tokens = 0
        
        for item in tqdm(samples, desc=lang):
            full_prompt = item["prompt"] + cfg["instruction"]
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = output_ids[0][prompt_len:]
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            total_tokens += len(generated_ids)

            # Accuracy Check
            gen_text_lower = gen_text.lower().strip()
            marker = cfg["answer_marker"]
            if marker in gen_text_lower:
                pred = gen_text_lower.split(marker)[-1].strip()
                # Remove punctuation
                pred = "".join(c for c in pred if c.isalnum())
                target = item["answer"].lower().strip()
                if target in pred:
                    correct += 1
            
            if total_tokens < 500: # Only print a few
                safe_prompt = str(item["prompt"]).encode("unicode_escape").decode("ascii")
                safe_gen = str(gen_text).encode("unicode_escape").decode("ascii")
                print(f"\nPrompt: {safe_prompt}")
                print(f"Gen: {safe_gen}")

        results[lang] = {
            "accuracy": correct / len(samples),
            "avg_tokens": total_tokens / len(samples)
        }

    # Final Summary
    print("\n" + "="*40)
    print("SAPIR-WHORF BASELINE AUDIT SUMMARY")
    print("="*40)
    en = results["en"]
    zh = results["zh"]
    
    print(f"English CoT Accuracy: {en['accuracy']:.4f}")
    print(f"Chinese CoT Accuracy: {zh['accuracy']:.4f}")
    print(f"English Avg Tokens:   {en['avg_tokens']:.2f}")
    print(f"Chinese Avg Tokens:   {zh['avg_tokens']:.2f}")
    
    delta = en['avg_tokens'] - zh['avg_tokens']
    efficiency = (en['avg_tokens'] / zh['avg_tokens']) if zh['avg_tokens'] > 0 else 0
    
    print(f"\nLanguage Tax (EN - ZH): {delta:.2f} tokens per sample")
    print(f"Token Inflation Factor:  {efficiency:.2f}x (EN is {efficiency:.2f}x larger)")
    print("="*40)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": "M18",
        "series": {
            "series_id": "M",
            "track": "M18",
            "script": "scripts/m18/run_sapir_whorf_audit.py",
        },
        "surface": "sapir_whorf_audit",
        "config": {
            "base_model": str(args.base_model),
        },
        "results": results,
        "metrics": {
            "english_accuracy": en["accuracy"],
            "chinese_accuracy": zh["accuracy"],
            "english_avg_tokens": en["avg_tokens"],
            "chinese_avg_tokens": zh["avg_tokens"],
            "language_tax_tokens": delta,
            "token_inflation_factor": efficiency,
        },
    }
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"M18 sapir-whorf audit report written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()
    run_sapir_whorf_audit(args)
