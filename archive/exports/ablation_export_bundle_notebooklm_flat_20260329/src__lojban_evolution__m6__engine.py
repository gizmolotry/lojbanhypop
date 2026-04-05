from __future__ import annotations
import torch
import torch.nn as nn
from .matrix_core import M6MatrixCore

class HardPointerDictionary:
    """
    Physical Law #2: The Hard Pointers.
    System 1 reaches down into System 2a and copies exact English word tensors.
    """
    def __init__(self, s2a_embeddings: torch.Tensor):
        # s2a_embeddings: [B, Prompt_Len, H]
        self.registry = s2a_embeddings

    def borrow(self, batch_idx: int, word_index: int) -> torch.Tensor:
        # word_index is clamped to prompt length
        idx = min(word_index, self.registry.shape[1] - 1)
        return self.registry[batch_idx, idx, :]

class System2aEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model.get_input_embeddings()
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Returns the continuous noun embeddings for Hard Pointers
        return self.encoder(input_ids)

class System1LoRA(nn.Module):
    def __init__(self, lora_model, matrix_core: M6MatrixCore):
        super().__init__()
        self.engine = lora_model
        self.matrix_core = matrix_core
        self.hidden_size = matrix_core.hidden_size
        
        # Integration head to combine internal hidden state with borrowed pointers
        # for the final resolution stream.
        self.resolution_head = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, s2a_embeddings: torch.Tensor, max_steps: int = 10, use_iron_collar: bool = True) -> torch.Tensor:
        """
        Physical Law #3: Auto-Regressive CoT.
        Chains logic until [<STOP>] is emitted.
        """
        batch_size = s2a_embeddings.shape[0]
        pointers = HardPointerDictionary(s2a_embeddings)
        
        # Initial hidden state (placeholder for LoRA start state)
        current_hidden = torch.zeros(batch_size, self.hidden_size, device=s2a_embeddings.device, dtype=s2a_embeddings.dtype)
        
        final_h = current_hidden
        for step in range(max_steps):
            # 1. Internal State Stream (KV cache passing simulation)
            current_hidden = current_hidden + torch.randn_like(current_hidden) * 0.01 
            
            # 2. Logic Matrix Generation
            op_logits, x_logits = self.matrix_core(current_hidden, use_iron_collar=use_iron_collar)
            
            # Hardcoded stop check for simulation
            # (In training, the model must learn to emit this)
            op_idx = torch.argmax(op_logits, dim=-1)
            
            # 3. Resolve Pointers (COCONUT Stream 2)
            # For each slot, grab the highest probability location in System 2a
            # We take slot 1 as an example
            ptr_indices = torch.argmax(x_logits[:, 0, :], dim=-1) # [B]
            borrowed_vectors = []
            for b in range(batch_size):
                borrowed_vectors.append(pointers.borrow(b, ptr_indices[b].item()))
            borrowed_v = torch.stack(borrowed_vectors) # [B, H]
            
            # Update hidden state with grounded entities
            current_hidden = self.resolution_head(torch.cat([current_hidden, borrowed_v], dim=-1))
            
            final_h = current_hidden
            if (op_idx == self.matrix_core.OP_STOP_IDX).all():
                break
                
        # Returns the final 'Resolution' tensor (COCONUT Stream 3)
        return final_h

class System2bDecoder(nn.Module):
    """
    System 2b (The Decoder): Mathematically lobotomized.
    """
    def __init__(self, base_lm_head):
        super().__init__()
        self.lm_head = base_lm_head
        
    def forward(self, resolution_payload: torch.Tensor) -> torch.Tensor:
        """
        Physical Law #4: The Choked Read Path.
        Only allowed to see System 1's final mathematical state.
        """
        # Predicts the English word from the logic payload
        return self.lm_head(resolution_payload)
