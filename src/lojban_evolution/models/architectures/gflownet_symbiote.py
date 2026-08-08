import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from lojban_evolution.m21.bridi import ANSWER_LABELS
from lojban_evolution.m26.end_to_end import M26TinyLanguageBackbone
from lojban_evolution.m29.star_qformer import InvertedQFormerBridge
from lojban_evolution.m29.gflownet_qformer import M29GFlowNetBridge

class M29StarQFormerSymbiote(nn.Module):
    """
    The definitive M29 model wrapper. 
    Connects the English continuous prompt -> STaR Q-Former -> Discrete Bottleneck -> Downstream LLM.
    """
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 128,
        num_queries: int = 5,
        target_vocab_size: int = 7,
        max_prompt_length: int = 128,
        language_layers: int = 1,
        language_heads: int = 2,
    ) -> None:
        super().__init__()
        self.vocab = {}  # Set later
        
        # 1. English context reader
        self.language_backbone = M26TinyLanguageBackbone(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_prompt_length=max_prompt_length,
            num_layers=language_layers,
            num_heads=language_heads,
        )
        
        # 2. Inverted Q-Former (Trace Generator)
        self.bridge = InvertedQFormerBridge(
            hidden_size=hidden_dim,
            num_queries=num_queries,
            vocab_size=target_vocab_size
        )
        
        # 3. Downstream LLM (A simple Answer Head for the Gauntlet tests)
        self.answer_head_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.answer_head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(ANSWER_LABELS))
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        override_discrete_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        If override_discrete_embeddings is provided, we maliciously inject them into the answer head
        instead of what the Q-former generated. This is the Causal Mediation probe mechanism.
        """
        # Read English context
        language_outputs = self.language_backbone(input_ids)
        prompt_hidden_states = language_outputs["token_hidden_states"]
        
        # Generate strict discrete trace
        discrete_tokens, discrete_embeddings, qformer_logits = self.bridge(prompt_hidden_states)
        
        # CAUSAL PROBE INJECTION POINT:
        if override_discrete_embeddings is not None:
            final_embeddings = override_discrete_embeddings
        else:
            final_embeddings = discrete_embeddings
            
        # Pool the discrete trace for the downstream Answer Head
        _, (h_n, _) = self.answer_head_rnn(final_embeddings)
        trace_state = h_n[-1]
        answer_logits = self.answer_head_mlp(trace_state)
        
        return {
            "answer_logits": answer_logits,
            "discrete_tokens": discrete_tokens,
            "discrete_embeddings": discrete_embeddings,
            "qformer_logits": qformer_logits,
            "prompt_state": language_outputs["prompt_state"]
        }

    @property
    def core(self):
        """Mock for M27 runtime compatibility if needed."""
        return self

class M29GFlowNetSymbiote(nn.Module):
    """
    The M29 GFlowNet wrapper. 
    Uses Contrastive Trajectory Balance (CTB) + Reward Shaping.
    """
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 128,
        num_queries: int = 12,
        target_vocab_size: int = 7,
        max_prompt_length: int = 128,
        language_layers: int = 1,
        language_heads: int = 2,
    ) -> None:
        super().__init__()
        self.vocab = {}
        
        # 1. English context reader
        self.language_backbone = M26TinyLanguageBackbone(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_prompt_length=max_prompt_length,
            num_layers=language_layers,
            num_heads=language_heads,
        )
        
        # 2. GFlowNet Q-Former (Trace Generator)
        self.bridge = M29GFlowNetBridge(
            hidden_size=hidden_dim,
            num_queries=num_queries,
            vocab_size=target_vocab_size
        )
        
        # 3. Downstream LLM (Answer Head)
        self.answer_head_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.answer_head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(ANSWER_LABELS))
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        target_answers: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass tailored for Contrastive Trajectory Balance.
        """
        # Read English context
        language_outputs = self.language_backbone(input_ids)
        prompt_hidden_states = language_outputs["token_hidden_states"]
        
        # Generate two traces via GFlowNet Bridge
        traces = self.bridge(prompt_hidden_states, temperature=temperature)
        
        # Evaluate Trace 1
        emb_1 = traces["tau_1"]["embeddings"]
        _, (h_n_1, _) = self.answer_head_rnn(emb_1)
        trace_state_1 = h_n_1[-1]
        logits_1 = self.answer_head_mlp(trace_state_1)
        
        # Evaluate Trace 2
        emb_2 = traces["tau_2"]["embeddings"]
        _, (h_n_2, _) = self.answer_head_rnn(emb_2)
        trace_state_2 = h_n_2[-1]
        logits_2 = self.answer_head_mlp(trace_state_2)
        
        output = {
            "tau_1": traces["tau_1"],
            "tau_2": traces["tau_2"],
            "logits_1": logits_1,
            "logits_2": logits_2,
            "prompt_state": language_outputs["prompt_state"]
        }
        
        # Calculate CTB components if target provided
        if target_answers is not None:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            
            # Cross entropy (loss) for each trace
            ce_1 = loss_fn(logits_1, target_answers)
            ce_2 = loss_fn(logits_2, target_answers)
            
            # Reward shaping: log R = -beta * L_CE
            log_r_1 = -beta * ce_1
            log_r_2 = -beta * ce_2
            
            # Trajectory Balance equations for each trace
            tb_1 = traces["tau_1"]["log_pf"] - log_r_1
            tb_2 = traces["tau_2"]["log_pf"] - log_r_2
            
            # CTB Loss: (TB1 - TB2)^2
            ctb_loss = (tb_1 - tb_2).pow(2).mean()
            
            # Answer head is trained to minimize standard cross entropy
            # We average the CE over both traces to keep Answer Head stable
            answer_head_loss = (ce_1 + ce_2).mean() / 2.0
            
            output["ctb_loss"] = ctb_loss
            output["answer_head_loss"] = answer_head_loss
            
        return output

    @property
    def core(self):
        return self
