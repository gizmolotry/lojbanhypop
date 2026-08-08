import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Sequence, Any, List, Dict
import wandb
import random

from lojban_evolution.m29.model import M29StarQFormerSymbiote
from lojban_evolution.m29.star_qformer import generate_star_traces

def train_m29_star_symbiote(
    model: M29StarQFormerSymbiote,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    samples_per_prompt: int = 10,
    device: str | torch.device = "cpu"
) -> List[Dict[str, float]]:
    """
    Core STaR Offline Rejection Sampling Optimization Loop.
    Generates N traces per prompt, keeps the successful ones, and trains via teacher forcing.
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_stats = {
            "total_generated": 0,
            "total_accepted": 0,
            "loss": 0.0
        }
        
        batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device_obj)
            expected_answers = batch["answer_id"].to(device_obj)
            
            # 1. Get continuous English representations
            with torch.no_grad():
                language_outputs = model.language_backbone(input_ids)
                prompt_hidden_states = language_outputs["token_hidden_states"]
            
            # 2. GENERATION PHASE (STaR Rejection Sampling)
            # We turn off gradients here because we just want to harvest successful discrete tokens
            with torch.no_grad():
                model.eval()  # Eval mode to avoid batchnorm/dropout issues during sampling
                accepted_traces = generate_star_traces(
                    model, 
                    prompt_hidden_states, 
                    expected_answers, 
                    num_samples=samples_per_prompt
                )
            
            # 3. IMITATION PHASE
            if not accepted_traces:
                # No traces succeeded (random chance failed). We skip this batch and keep grinding.
                continue
                
            model.train()
            
            # Batch up the successful traces
            gold_prompt_hidden = torch.stack([t["prompt_hidden"] for t in accepted_traces])
            gold_trace_tokens = torch.stack([t["trace_tokens"] for t in accepted_traces])
            gold_answers = torch.stack([t["answer_id"] for t in accepted_traces])
            
            # Re-run forward pass with gradients on the successful inputs
            _, discrete_embeddings, logits = model.bridge(gold_prompt_hidden)
            
            # 3a. Force Q-Former to emit the successful traces
            loss_generator = F.cross_entropy(
                logits.view(-1, model.bridge.vocab_size),
                gold_trace_tokens.view(-1)
            )
            
            # 3b. Force downstream LLM to predict the correct answer from the discrete embeddings
            trace_state = discrete_embeddings.sum(dim=1) / max(1, discrete_embeddings.shape[1])
            downstream_logits = model.answer_head(trace_state)
            loss_answer = F.cross_entropy(downstream_logits, gold_answers)
            
            # Combine and backpropagate
            loss = loss_generator + loss_answer
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_stats["loss"] += loss.item()
            epoch_stats["total_generated"] += len(input_ids) * samples_per_prompt
            epoch_stats["total_accepted"] += len(accepted_traces)
            batches += 1
            
        if batches > 0:
            epoch_stats["loss"] /= batches
            
        acceptance_rate = epoch_stats["total_accepted"] / max(1, epoch_stats["total_generated"])
        print(f"Epoch {epoch+1}/{epochs} | Acc. Rate: {acceptance_rate:.4f} | Loss: {epoch_stats['loss']:.4f}")
        history.append(epoch_stats)
        
    return history
