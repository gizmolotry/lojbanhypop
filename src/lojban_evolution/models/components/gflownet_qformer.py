import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

class M29GFlowNetBridge(nn.Module):
    """
    M29 GFlowNet Bridge.
    Acts as the Forward Policy (P_F) for Contrastive Trajectory Balance (CTB).
    Reads continuous English context via cross-attention, then acts as an autoregressive decoder.
    Output is exactly two strictly discrete Lojban traces and their log-probabilities.
    """
    def __init__(self, hidden_size: int, num_queries: int, vocab_size: int = 7):
        super().__init__()
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # The learned query embeddings (autoregressive start token / position embeddings)
        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        self.emitter_head = nn.Linear(hidden_size, vocab_size)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal mask to prevent peeking ahead."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def sample_trace(self, prompt_hidden_states: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a single trace from the policy.
        Returns:
            discrete_tokens: [batch, seq_len]
            discrete_embeddings: [batch, seq_len, hidden_size]
            sum_log_pf: [batch] (The sum of log P_F for this trace, computed using TRUE un-tempered logits)
        """
        batch_size = prompt_hidden_states.shape[0]
        device = prompt_hidden_states.device
        
        # Expand queries for the batch
        queries = self.query_embeddings.expand(batch_size, -1, -1)
        causal_mask = self.generate_square_subsequent_mask(self.num_queries, device)
        
        # Pass through the Decoder
        decoder_out = self.decoder(
            tgt=queries, 
            memory=prompt_hidden_states, 
            tgt_mask=causal_mask
        )
        
        # Emit raw un-tempered logits
        logits = self.emitter_head(decoder_out) # [batch, seq_len, vocab_size]
        
        # Calculate true un-tempered log probabilities for the loss equation
        log_probs_full = F.log_softmax(logits, dim=-1) # [batch, seq_len, vocab_size]
        
        # Tempered Sampling Distribution for physical action selection
        if temperature != 1.0:
            sampling_logits = logits / temperature
        else:
            sampling_logits = logits
            
        sampling_probs = F.softmax(sampling_logits, dim=-1)
        
        # Sample discrete tokens across the sequence
        # Note: torch.multinomial takes 2D [batch*seq_len, vocab_size] and returns 1D indices
        discrete_tokens = torch.multinomial(sampling_probs.view(-1, self.vocab_size), 1).view(batch_size, self.num_queries)
        
        # Gather the true log_pf of the selected tokens
        # discrete_tokens is [batch, seq_len], we need to gather from log_probs_full along vocab_size
        selected_log_probs = log_probs_full.gather(dim=2, index=discrete_tokens.unsqueeze(2)).squeeze(2) # [batch, seq_len]
        
        # Sum over the sequence length to get the total log P_F(tau)
        sum_log_pf = selected_log_probs.sum(dim=1) # [batch]
        
        discrete_embeddings = self.token_embeddings(discrete_tokens)
        return discrete_tokens, discrete_embeddings, sum_log_pf

    def forward(self, prompt_hidden_states: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Samples two independent traces for Contrastive Trajectory Balance.
        """
        # Sample Trace 1
        tokens_1, emb_1, log_pf_1 = self.sample_trace(prompt_hidden_states, temperature)
        
        # Sample Trace 2
        tokens_2, emb_2, log_pf_2 = self.sample_trace(prompt_hidden_states, temperature)
        
        return {
            "tau_1": {
                "tokens": tokens_1,
                "embeddings": emb_1,
                "log_pf": log_pf_1
            },
            "tau_2": {
                "tokens": tokens_2,
                "embeddings": emb_2,
                "log_pf": log_pf_2
            }
        }
