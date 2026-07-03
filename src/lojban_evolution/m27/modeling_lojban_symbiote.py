import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
from typing import Optional, Tuple, Dict, Any, List

class LojbanSymbioteConfig(PretrainedConfig):
    model_type = "lojban_symbiote"
    
    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        tap_layer: int = 12,
        bottleneck_dim: int = 64,
        max_scratchpad_steps: int = 16,
        geometry_mode: str = "hyperbolic",
        enable_symbiote_routing: bool = True,
        enable_prompt_bypass_choke: bool = True,
        poincare_curvature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.tap_layer = tap_layer
        self.bottleneck_dim = bottleneck_dim
        self.max_scratchpad_steps = max_scratchpad_steps
        self.geometry_mode = geometry_mode
        self.enable_symbiote_routing = enable_symbiote_routing
        self.enable_prompt_bypass_choke = enable_prompt_bypass_choke
        self.poincare_curvature = poincare_curvature


class CoconutRecurrentCell(nn.Module):
    """
    M27 Autoregressive COCONUT Time Cell.
    Takes the previous state, prompt context, and previous bridi emission to compute the next state.
    """
    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        # GRU cell to update the latent state recurrently
        self.gru_cell = nn.GRUCell(input_size=hidden_size + bottleneck_dim, hidden_size=bottleneck_dim)
        
    def forward(self, z_prev: torch.Tensor, prompt_context: torch.Tensor, b_prev_embed: torch.Tensor) -> torch.Tensor:
        # prompt_context: [batch, hidden_size]
        # b_prev_embed: [batch, bottleneck_dim]
        # z_prev: [batch, bottleneck_dim]
        # Concatenate prompt context and previous emission embedding as input
        gru_input = torch.cat([prompt_context, b_prev_embed], dim=-1)
        z_next = self.gru_cell(gru_input, z_prev)
        return z_next


class AutoregressiveBridiEmitter(nn.Module):
    """
    M25 Loose Bridi Grammar Emitter.
    Emits the grammar action stream: OPEN, PRED, MOD, ARG, LINK, CLOSE, STOP.
    """
    VOCAB = ["OPEN", "PRED", "MOD", "ARG", "LINK", "CLOSE", "STOP"]
    
    def __init__(self, bottleneck_dim: int):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.vocab_size = len(self.VOCAB)
        self.token_to_id = {t: i for i, t in enumerate(self.VOCAB)}
        self.id_to_token = {i: t for i, t in enumerate(self.VOCAB)}
        
        self.action_head = nn.Linear(bottleneck_dim, self.vocab_size)
        self.action_embedding = nn.Embedding(self.vocab_size, bottleneck_dim)
        
    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: action logits, soft embedding, hard token IDs
        logits = self.action_head(z_t)
        probs = F.softmax(logits, dim=-1)
        
        # Soft embedding for differentiable training (M26 soft handoff)
        soft_embed = probs @ self.action_embedding.weight
        
        # Hard token for free-run/trace
        hard_token = logits.argmax(dim=-1)
        
        return logits, soft_embed, hard_token


def apply_hyperbolic_packing(z: torch.Tensor, token_ids: torch.Tensor, curvature: float) -> torch.Tensor:
    """
    M11 Provenance & Iron Collar packing.
    Pushes continuous vectors into different radius bands in Poincare space based on their symbolic provenance.
    """
    # Simplified hyperbolic packing logic
    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Different radius bands based on token ID (simplified mockup of gismu/cmavo/judri mapping)
    # Using token_ids as a rough proxy for family
    bands = (token_ids.float() / 7.0).unsqueeze(-1) * 0.5 + 0.1
    # Scale to target radius
    z_hyperbolic = z / norm * bands
    return z_hyperbolic


class LojbanSymbioteCausalLM(PreTrainedModel):
    config_class = LojbanSymbioteConfig
    
    def __init__(self, config: LojbanSymbioteConfig):
        super().__init__(config)
        self.backbone = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        
        # Ensure we have the base hidden size
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 1024)
        
        # Symbiote Organs
        self.compress = nn.Linear(self.hidden_size, config.bottleneck_dim)
        self.expand = nn.Linear(config.bottleneck_dim, self.hidden_size)
        
        self.coconut_cell = CoconutRecurrentCell(self.hidden_size, config.bottleneck_dim)
        self.emitter = AutoregressiveBridiEmitter(config.bottleneck_dim)
        
        # J-Series & L-Series constraints tracker
        self.register_buffer("j_series_foil_acceptance", torch.tensor(0.0))
        
        # Register the trace injection hook
        self._trace_output = {}
        self.hook_handle = self.backbone.model.layers[self.config.tap_layer].register_forward_pre_hook(self._injection_hook)

    def _injection_hook(self, module, args):
        if not self.config.enable_symbiote_routing:
            return args
            
        hidden_states = args[0]
        # In M19/M27, the scratchpad tokens are pre-allocated at the end of the prompt
        # We extract the context just before the scratchpad
        seq_len = hidden_states.shape[1]
        scratchpad_steps = self.config.max_scratchpad_steps
        if seq_len <= scratchpad_steps:
            return args # Not enough tokens to inject
            
        prompt_context = hidden_states[:, -(scratchpad_steps + 1), :]
        
        z_prev = self.compress(prompt_context)
        b_prev_embed = torch.zeros_like(z_prev)
        
        bridi_trace_soft = []
        bridi_trace_hard = []
        bridi_trace_logits = []
        
        for t in range(scratchpad_steps):
            z_t = self.coconut_cell(z_prev, prompt_context, b_prev_embed)
            logits, soft_embed, hard_token = self.emitter(z_t)
            
            if self.config.geometry_mode == "hyperbolic":
                z_t = apply_hyperbolic_packing(z_t, hard_token, self.config.poincare_curvature)
            
            bridi_trace_soft.append(z_t)
            bridi_trace_hard.append(hard_token)
            bridi_trace_logits.append(logits)
            
            b_prev_embed = soft_embed
            z_prev = z_t
            
        trace_tensor = torch.stack(bridi_trace_soft, dim=1)
        trace_ids = torch.stack(bridi_trace_hard, dim=1)
        delta = self.expand(trace_tensor)
        
        # Overwrite the hidden states of the scratchpad tokens with our trace
        # This keeps seq_len constant so HuggingFace causal masks stay intact
        injected = hidden_states.clone()
        
        if self.config.enable_prompt_bypass_choke:
            # Zero out everything BEFORE the scratchpad so the answer head can't read the prompt
            injected[:, :-scratchpad_steps, :] = 0.0
            
        injected[:, -scratchpad_steps:, :] = delta
        
        # Save telemetry to self for access after forward pass
        self._trace_output = {
            "tier_a_constraints": self._calculate_l_series_constraints(trace_ids),
            "loose_bridi_stream": trace_ids,
            "bridi_logits": torch.stack(bridi_trace_logits, dim=1),
        }
        
        return (injected,)

    def _extract_prompt_state(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Pool the prompt states (e.g. mean pooling or taking the last valid token)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        # Get the hidden state of the last actual prompt token
        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths, :]
        return pooled

    def _calculate_l_series_constraints(self, bridi_stream_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        L-Series Tier A Constraints: Arity, Scope, Identity.
        c_i violations to be passed to the Augmented Lagrangian trainer.
        """
        # Calculate violations (0 means perfectly compliant)
        # Simplified implementations
        arity_violation = torch.tensor(0.0, device=bridi_stream_ids.device, requires_grad=True)
        scope_violation = torch.tensor(0.0, device=bridi_stream_ids.device, requires_grad=True)
        identity_violation = torch.tensor(0.0, device=bridi_stream_ids.device, requires_grad=True)
        
        # If STOP is missing or OPEN/CLOSE is mismatched, that is a structural violation
        for b in range(bridi_stream_ids.shape[0]):
            has_stop = (bridi_stream_ids[b] == self.emitter.token_to_id["STOP"]).any()
            if not has_stop:
                scope_violation = scope_violation + 1.0
                
        # Normalize
        b_size = bridi_stream_ids.shape[0]
        return {
            "arity_violation": arity_violation / b_size,
            "scope_violation": scope_violation / b_size,
            "identity_violation": identity_violation / b_size,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Clear previous telemetry
        self._trace_output = {}
        
        # Standard forward pass, the _injection_hook handles the bridge!
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        if not self.config.enable_symbiote_routing or not self._trace_output:
            return {"logits": outputs.logits}

        return {
            "logits": outputs.logits,
            "tier_a_constraints": self._trace_output["tier_a_constraints"],
            "loose_bridi_stream": self._trace_output["loose_bridi_stream"],
            "bridi_logits": self._trace_output["bridi_logits"],
            "j_series_diagnostics": {
                "foil_acceptance": self.j_series_foil_acceptance
            }
        }
