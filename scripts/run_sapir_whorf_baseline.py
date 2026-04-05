from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 20 Logic Puzzles with English and Chinese versions
PUZZLES = [
    {
        "id": 1,
        "en": "The trophy does not fit in the suitcase because it is too big. What is too big?",
        "zh": "奖杯放不进手提箱，因为它太大了。什么太大了？",
        "ans": "trophy",
        "ans_zh": "奖杯"
    },
    {
        "id": 2,
        "en": "The trophy does not fit in the suitcase because it is too small. What is too small?",
        "zh": "奖杯放不进手提箱，因为它太小了。什么太小了？",
        "ans": "suitcase",
        "ans_zh": "手提箱"
    },
    {
        "id": 3,
        "en": "The city councilmen refused the demonstrators a permit because they feared violence. Who feared violence?",
        "zh": "市议员拒绝给示威者许可证，因为他们害怕暴力。谁害怕暴力？",
        "ans": "councilmen",
        "ans_zh": "市议员"
    },
    {
        "id": 4,
        "en": "The city councilmen refused the demonstrators a permit because they advocated revolution. Who advocated revolution?",
        "zh": "市议员拒绝给示威者许可证，因为他们提倡革命。谁提倡革命？",
        "ans": "demonstrators",
        "ans_zh": "示威者"
    },
    {
        "id": 5,
        "en": "The blue box is inside the red box. The red box is inside the green crate. Is the blue box inside the green crate?",
        "zh": "蓝盒子在红盒子里面。红盒子在绿箱子里面。蓝盒子在绿箱子里面吗？",
        "ans": "yes",
        "ans_zh": "是"
    },
    {
        "id": 6,
        "en": "The silver case is left of the amber chest. The amber chest is left of the red box. Is the red box left of the silver case?",
        "zh": "银色箱子在琥珀色箱子的左边。琥珀色箱子在红色盒子的左边。红色盒子在银色箱子的左边吗？",
        "ans": "no",
        "ans_zh": "不是"
    },
    {
        "id": 7,
        "en": "The blue box is right of the green crate. The green crate is right of the silver case. Is the blue box right of the silver case?",
        "zh": "蓝盒子在绿箱子的右边。绿箱子在银色箱子的右边。蓝盒子在银色箱子的右边吗？",
        "ans": "yes",
        "ans_zh": "是"
    },
    {
        "id": 8,
        "en": "Alice puts the red ball in the box. Bob moves it to the drawer while Alice is watching. Where does Alice think the red ball is?",
        "zh": "爱丽丝把红球放在盒子里。鲍勃在爱丽丝看着的时候把它移到了抽屉里。爱丽丝认为红球在哪里？",
        "ans": "drawer",
        "ans_zh": "抽屉"
    },
    {
        "id": 9,
        "en": "Alice puts the red ball in the box. Bob moves it to the drawer while Alice is outside. Where does Alice think the red ball is?",
        "zh": "爱丽丝把红球放在盒子里。鲍勃在爱丽丝不在场的时候把它移到了抽屉里。爱丽丝认为红球在哪里？",
        "ans": "box",
        "ans_zh": "盒子"
    },
    {
        "id": 10,
        "en": "The crane could not lift the beam because it was too heavy. What was too heavy?",
        "zh": "起重机无法吊起梁，因为它太重了。什么太重了？",
        "ans": "beam",
        "ans_zh": "梁"
    },
    {
        "id": 11,
        "en": "The crane could not lift the beam because it was too weak. What was too weak?",
        "zh": "起重机无法吊起梁，因为它太弱了。什么太弱了？",
        "ans": "crane",
        "ans_zh": "起重机"
    },
    {
        "id": 12,
        "en": "The delivery truck zoomed by the school bus because it was going so fast. What was going fast?",
        "zh": "送货卡车飞驰过校车，因为它开得很快。什么开得很快？",
        "ans": "delivery truck",
        "ans_zh": "送货卡车"
    },
    {
        "id": 13,
        "en": "The delivery truck passed the school bus because it was going so slow. What was going slow?",
        "zh": "送货卡车超过了校车，因为它开得很慢。什么开得很慢？",
        "ans": "school bus",
        "ans_zh": "校车"
    },
    {
        "id": 14,
        "en": "Sam tried to call Jordan on the phone, but they were not available. Who was not available?",
        "zh": "山姆试图给乔丹打电话，但他们没空。谁没空？",
        "ans": "Jordan",
        "ans_zh": "乔丹"
    },
    {
        "id": 15,
        "en": "Sam tried to call Jordan on the phone, but they had no signal. Who had no signal?",
        "zh": "山姆试图给乔丹打电话，但他们没有信号。谁没有信号？",
        "ans": "Sam",
        "ans_zh": "山姆"
    },
    {
        "id": 16,
        "en": "If A > B and B > C, with A=8, B=5, C=2, is A > C?",
        "zh": "如果 A > B 且 B > C，且 A=8, B=5, C=2，那么 A > C 吗？",
        "ans": "yes",
        "ans_zh": "是"
    },
    {
        "id": 17,
        "en": "Design happens before Test. Test happens before Deploy. Does Design happen before Deploy?",
        "zh": "设计在测试之前发生。测试在部署之前发生。设计在部署之前发生吗？",
        "ans": "yes",
        "ans_zh": "是"
    },
    {
        "id": 18,
        "en": "Design happens before Test. Test happens before Deploy. Does Deploy happen before Design?",
        "zh": "设计在测试之前发生。测试在部署之前发生。部署在设计之前发生吗？",
        "ans": "no",
        "ans_zh": "不是"
    },
    {
        "id": 19,
        "en": "No tester missed a build. Every tester did not miss a build. Are these equivalent?",
        "zh": "没有测试员错过版本。每个测试员都没有错过版本。这两者等价吗？",
        "ans": "yes",
        "ans_zh": "是"
    },
    {
        "id": 20,
        "en": "Policy: If an action handles personal data, approval is required unless an emergency waiver exists. Action P handles personal data and has no waiver. Is P permitted without approval?",
        "zh": "政策：如果某项操作处理个人数据，除非存在紧急豁免，否则需要批准。操作 P 处理个人数据且没有豁免。P 在未经批准的情况下被允许吗？",
        "ans": "no",
        "ans_zh": "不是"
    }
]

def run_eval(model, tokenizer, puzzles, lang='en'):
    results = []
    device = model.device
    
    cot_instr = "Think step-by-step before answering." if lang == 'en' else "在回答之前请逐步思考。"
    
    for p in puzzles:
        question = p[lang]
        prompt = f"Question: {question}\n{cot_instr}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_tokens = output[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Simple accuracy check
        gold = p['ans'] if lang == 'en' else p['ans_zh']
        correct = gold.lower() in response.lower()
        
        results.append({
            "id": p["id"],
            "response": response,
            "tokens": len(generated_tokens),
            "correct": correct
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    
    print(f"Loading frozen base model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto")
    
    print("\nStarting Sapir-Whorf Baseline Audit...")
    
    print("Running English CoT split...")
    en_results = run_eval(model, tokenizer, PUZZLES, lang='en')
    
    print("Running Chinese CoT split...")
    zh_results = run_eval(model, tokenizer, PUZZLES, lang='zh')
    
    # Calculate Metrics
    def get_metrics(results):
        acc = sum(1 for r in results if r['correct']) / len(results)
        correct_tokens = [r['tokens'] for r in results if r['correct']]
        avg_tokens = sum(correct_tokens) / len(correct_tokens) if correct_tokens else 0
        return acc, avg_tokens

    en_acc, en_tokens = get_metrics(en_results)
    zh_acc, zh_tokens = get_metrics(zh_results)
    
    delta = en_tokens - zh_tokens
    delta_pct = (delta / en_tokens * 100) if en_tokens > 0 else 0
    
    print("\n--- SAPIR-WHORF BASELINE AUDIT REPORT ---")
    print(f"Model: {args.model_path}")
    print(f"Sample Size: {len(PUZZLES)} puzzles")
    print("-" * 40)
    print(f"Cell A (English CoT):")
    print(f"  Accuracy: {en_acc:.2%}")
    print(f"  Avg Tokens (Correct): {en_tokens:.2f}")
    print("-" * 40)
    print(f"Cell B (Chinese CoT):")
    print(f"  Accuracy: {zh_acc:.2%}")
    print(f"  Avg Tokens (Correct): {zh_tokens:.2f}")
    print("-" * 40)
    print(f"Language Tax Delta: {delta:.2f} tokens ({delta_pct:.2f}% reduction in Chinese)")
    print(f"The Floor: Established at {zh_tokens:.2f} tokens.")
    print("-" * 40)
    
    # Save to file
    out_dir = Path("archive/reports/sapir_whorf")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "en": en_results,
        "zh": zh_results,
        "metrics": {
            "en_acc": en_acc,
            "en_tokens": en_tokens,
            "zh_acc": zh_acc,
            "zh_tokens": zh_tokens,
            "delta": delta,
            "delta_pct": delta_pct
        }
    }
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {out_dir / 'audit_report.json'}")

if __name__ == "__main__":
    main()
