import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def run_sft(model_path: str, dataset_path: str, output_dir: str, train_steps: int):
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    def tokenize_function(examples):
        # Format: Prompt + \n <logic> logic_str </logic> \n Answer: answer
        texts = [
            f"Question: {p}\nReasoning: Let's think step by step.\n<logic> {l} </logic>\nAnswer: {a}"
            for p, l, a in zip(examples["prompt"], examples["flattened_logic"], examples["answer"])
        ]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=train_steps,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Discrete SFT baseline on flattened logic strings.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-steps", type=int, default=500)
    args = parser.parse_args()
    run_sft(args.model_path, args.dataset_path, args.output_dir, args.train_steps)
