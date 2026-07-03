import torch
import pytest
from transformers import AutoConfig
from lojban_evolution.m27.modeling_lojban_symbiote import LojbanSymbioteConfig, LojbanSymbioteCausalLM

@pytest.fixture
def dummy_config():
    return LojbanSymbioteConfig(
        base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        tap_layer=2,  # Shorten for testing
        bottleneck_dim=16,
        max_scratchpad_steps=4,
        geometry_mode="hyperbolic",
        enable_symbiote_routing=True,
        enable_prompt_bypass_choke=True,
        poincare_curvature=1.0,
    )

def test_symbiote_forward_pass_gradient_flow(dummy_config):
    model = LojbanSymbioteCausalLM(dummy_config)
    
    # Freeze the base Qwen to simulate just training the symbiote
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze only the custom symbiote parts
    for name, param in model.named_parameters():
        if "compress" in name or "expand" in name or "coconut_cell" in name or "emitter" in name:
            param.requires_grad = True

    batch_size = 2
    # The input must contain prompt + scratchpad tokens for M19 architecture
    prompt_len = 10
    scratchpad_len = dummy_config.max_scratchpad_steps
    seq_len = prompt_len + scratchpad_len
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Run the forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # Unpack outputs
    logits = outputs["logits"]
    tier_a_constraints = outputs["tier_a_constraints"]
    
    # Assert dimensions
    assert logits.shape == (batch_size, seq_len, model.backbone.config.vocab_size)
    
    # Check Tier A constraints
    assert "arity_violation" in tier_a_constraints
    assert "scope_violation" in tier_a_constraints
    assert "identity_violation" in tier_a_constraints
    
    # Create a dummy loss (e.g. language modeling loss on the final token + Lagrangian multipliers)
    lm_loss = logits.mean()
    # Mock lagrangian multipliers
    lambda_arity = 0.5
    lambda_scope = 0.5
    lambda_identity = 0.5
    
    total_loss = lm_loss + \
                 lambda_arity * tier_a_constraints["arity_violation"] + \
                 lambda_scope * tier_a_constraints["scope_violation"] + \
                 lambda_identity * tier_a_constraints["identity_violation"]
                 
    # Backpropagate
    total_loss.backward()
    
    # Verify gradients reach the symbiote organs
    # This verifies the "M26 Spinal Cord" claim
    assert model.compress.weight.grad is not None, "Gradient did not reach the compressor"
    assert model.expand.weight.grad is not None, "Gradient did not reach the expander"
    assert model.coconut_cell.gru_cell.weight_ih.grad is not None, "Gradient did not reach the coconut cell"
    assert model.emitter.action_head.weight.grad is not None, "Gradient did not reach the action emitter"
    
def test_prompt_bypass_choke_activation(dummy_config):
    # Verify that if the choke is active, the prompt hidden states are truly severed.
    model = LojbanSymbioteCausalLM(dummy_config)
    
    batch_size = 1
    prompt_len = 5
    scratchpad_len = dummy_config.max_scratchpad_steps
    seq_len = prompt_len + scratchpad_len
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model(input_ids)
    
    # If the bypass is blocked, the model should still be able to output logits 
    # relying *only* on the injected trace. The previous test checks gradient flow,
    # this just ensures it runs without error when bypass is blocked.
    assert outputs["logits"] is not None
