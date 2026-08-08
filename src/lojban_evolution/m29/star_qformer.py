import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import PreTrainedModel, AutoModelForCausalLM
from typing import Optional, Tuple, Dict, Any, List

class InvertedQFormerBridge(nn.Module):
    """
    M29 Inverted Q-Former.
    Reads continuous English context via cross-attention, then acts as an autoregressive decoder.
    Output is strictly discrete Lojban symbols.
    """
    def __init__(self, hidden_size: int, num_queries: int, vocab_size: int = 7):
        super().__init__()
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # The learned query embeddings (autoregressive start token / position embeddings)
        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        
        # We need a Transformer Decoder layer to properly handle causal masking and cross-attention
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Token Emitter (No Gumbel-Softmax, we rely on STaR discrete sampling)
        self.emitter_head = nn.Linear(hidden_size, vocab_size)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal mask to prevent peeking ahead."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, prompt_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        prompt_hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size = prompt_hidden_states.shape[0]
        device = prompt_hidden_states.device
        
        # Expand queries for the batch
        queries = self.query_embeddings.expand(batch_size, -1, -1)
        
        # Create causal mask for autoregressive generation
        causal_mask = self.generate_square_subsequent_mask(self.num_queries, device)
        
        # Pass through the Decoder (Self-Attention with causal mask + Cross-Attention to prompt)
        decoder_out = self.decoder(
            tgt=queries, 
            memory=prompt_hidden_states, 
            tgt_mask=causal_mask
        )
        
        # Emit logits for each slot
        logits = self.emitter_head(decoder_out)
        
        # Discrete Sampling (STaR offline generation)
        probs = F.softmax(logits, dim=-1)
        # Sample discrete tokens sequentially or via multinomial over the sequence
        discrete_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, self.num_queries)
        
        # Embed the discrete tokens to pass to downstream LLM
        discrete_embeddings = self.token_embeddings(discrete_tokens)
        
        return discrete_tokens, discrete_embeddings, logits

def generate_star_traces(model, prompt_hidden_states, expected_answers, num_samples=10):
    """
    Offline Rejection Sampling (STaR).
    Generates multiple discrete traces, runs them downstream, and filters for correctness.
    """
    batch_size = prompt_hidden_states.shape[0]
    accepted_traces = []
    
    total_generated = 0
    total_accepted = 0

    for _ in range(num_samples):
        # 1. Generate candidate trace
        discrete_tokens, discrete_embeddings, logits = model.bridge(prompt_hidden_states)
        
        # 2. Run through downstream LLM (M29 Answer Head)
        trace_state = discrete_embeddings.sum(dim=1) / max(1, discrete_embeddings.shape[1])
        downstream_logits = model.answer_head(trace_state)
        predicted_answers = downstream_logits.argmax(dim=-1)
        
        # 3. Filter for correct answers
        correct_mask = (predicted_answers == expected_answers)
        
        total_generated += batch_size
        total_accepted += correct_mask.sum().item()
        
        for i in range(batch_size):
            if correct_mask[i]:
                accepted_traces.append({
                    'trace_tokens': discrete_tokens[i],
                    'logits': logits[i],
                    'prompt_hidden': prompt_hidden_states[i],
                    'answer_id': expected_answers[i]
                })

    # Telemetry
    acceptance_rate = total_accepted / max(1, total_generated)
    if wandb.run is not None:
        wandb.log({"star/trace_acceptance_rate": acceptance_rate})
        
    return accepted_traces
