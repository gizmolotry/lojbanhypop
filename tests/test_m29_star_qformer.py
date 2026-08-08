import pytest
import torch
from lojban_evolution.m29.star_qformer import InvertedQFormerBridge, generate_star_traces

def test_inverted_qformer_bridge_forward():
    batch_size = 2
    seq_len = 10
    hidden_size = 16
    num_queries = 5
    vocab_size = 7
    
    bridge = InvertedQFormerBridge(hidden_size=hidden_size, num_queries=num_queries, vocab_size=vocab_size)
    prompt_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    discrete_tokens, discrete_embeddings, logits = bridge(prompt_hidden_states)
    
    assert discrete_tokens.shape == (batch_size, num_queries)
    assert discrete_embeddings.shape == (batch_size, num_queries, hidden_size)
    assert logits.shape == (batch_size, num_queries, vocab_size)

def test_causal_mask_generation():
    bridge = InvertedQFormerBridge(hidden_size=16, num_queries=5, vocab_size=7)
    mask = bridge.generate_square_subsequent_mask(5, torch.device('cpu'))
    
    # Check that diagonal is 0, above diagonal is -inf (for PyTorch Transformer)
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == float('-inf')
    assert mask[4, 4] == 0.0
    assert mask[4, 0] == 0.0
