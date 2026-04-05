from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('archive/results/m9/active/RESULTS_M9_SYNCED/synced_model')
loj_0_id = tokenizer.convert_tokens_to_ids("<loj_0>")
print(f"Lojban Start Index: {loj_0_id}")
print(f"Total Vocab Size:   {len(tokenizer)}")
