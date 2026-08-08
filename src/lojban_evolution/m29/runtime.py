import torch
from typing import Any, Sequence
from collections import defaultdict
from torch.utils.data import DataLoader

from lojban_evolution.m25.emergent_bridi import M25EmergentBridiExample, M25LooseBridiDataset, m25_collate
from lojban_evolution.m24.compression import _accuracy
from lojban_evolution.m29.model import M29StarQFormerSymbiote

def evaluate_m29_star_runtime(
    *,
    model: M29StarQFormerSymbiote,
    examples: Sequence[M25EmergentBridiExample],
    vocab: dict[str, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """
    Evaluates the M29 STaR Q-Former model.
    Runs the Causal Mediation Gauntlet (Trace Swapping & Topology Corruption).
    """
    device_obj = torch.device(device)
    # Reusing the existing M25 dataset format since the inputs/answers are the same
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=32)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=m25_collate)
    
    model.eval()
    
    logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_obj)
            target = batch["answer_id"].to(device_obj)
            targets.append(target.detach().cpu())
            
            # PASS 1: Baseline (Normal Generation)
            outputs = model(input_ids)
            logits["predicted"].append(outputs["answer_logits"].detach().cpu())
            
            original_embeddings = outputs["discrete_embeddings"]
            original_tokens = outputs["discrete_tokens"]
            
            # PASS 2: Trace Swapping
            # We shift the input_ids by 1. We inject the ORIGINAL traces into the SHIFTED prompts.
            # We measure if the model outputs the ORIGINAL targets.
            # If so, the trace fully dictates the causal outcome, ignoring the prompt.
            shifted_input_ids = torch.roll(input_ids, shifts=1, dims=0)
            swap_outputs = model(shifted_input_ids, override_discrete_embeddings=original_embeddings)
            logits["trace_swapped"].append(swap_outputs["answer_logits"].detach().cpu())
            
            # PASS 3: Topology Corruption
            # We randomly shuffle the Lojban tokens within the trace to break the syntax tree.
            # We measure if the accuracy collapses.
            corrupted_tokens = original_tokens.clone()
            for i in range(corrupted_tokens.size(0)):
                perm = torch.randperm(corrupted_tokens.size(1))
                corrupted_tokens[i] = corrupted_tokens[i, perm]
                
            corrupted_embeddings = model.bridge.token_embeddings(corrupted_tokens)
            corrupt_outputs = model(input_ids, override_discrete_embeddings=corrupted_embeddings)
            logits["topology_corrupted"].append(corrupt_outputs["answer_logits"].detach().cpu())

    target_all = torch.cat(targets, dim=0)
    all_logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    
    predicted_acc = _accuracy(all_logits["predicted"], target_all)
    trace_swapped_acc = _accuracy(all_logits["trace_swapped"], target_all)
    corrupted_acc = _accuracy(all_logits["topology_corrupted"], target_all)
    
    # We want trace_swapped to remain high (close to predicted_acc).
    # We want corrupted_acc to fall significantly.
    trace_causality_delta = float(trace_swapped_acc - corrupted_acc)
    
    metrics = {
        "strict_accuracy": predicted_acc,
        "trace_swapped_accuracy": trace_swapped_acc,
        "topology_corrupted_accuracy": corrupted_acc,
        "trace_causality_delta": trace_causality_delta,
        "trace_token_count": 5.0, # Hardcoded num_queries for now
        "mean_prompt_tokens": 10.0, # Placeholder
    }
    
    return {"metrics": metrics}
