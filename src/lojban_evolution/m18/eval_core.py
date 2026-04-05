from __future__ import annotations
import torch
import json
from pathlib import Path
from typing import Dict, Any, List

from .salience import M18SalienceSelector
from .graph_induction import M18RelationalInterpreter, M18BiasCompiler
from .attention_patch import apply_m18_bias

class M18TwoPassOrchestrator:
    """
    M18-v0 Orchestrator: Executes the Clean Analysis and Biased Execution passes.
    """
    def __init__(
        self, 
        model, 
        tokenizer, 
        salience_selector: M18SalienceSelector,
        interpreter: M18RelationalInterpreter,
        bias_compiler: M18BiasCompiler,
        config: Dict[str, Any]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.salience = salience_selector
        self.interpreter = interpreter
        self.compiler = bias_compiler
        self.config = config
        self.device = next(model.parameters()).device

    def execute(self, prompt: str, intervention_active: bool = True, max_new_tokens: int = 20) -> Dict[str, Any]:
        """
        Executes the Two-Pass Protocol.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_l = inputs.input_ids.shape[1]
        
        # --- PASS 1: CLEAN ANALYSIS ---
        with torch.no_grad():
            outputs_p1 = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            # Extract from tap layer
            h_tap = outputs_p1.hidden_states[self.config["tap_layer"]] # [1, L, H]
            
        # --- SYMBIOTE: GRAPH INDUCTION ---
        scores, top_k_indices, mask = self.salience(h_tap)
        
        # Extract embeddings at salient positions
        # salient_embs: [1, K, H]
        salient_embs = torch.gather(
            h_tap, 1, 
            top_k_indices.unsqueeze(-1).expand(-1, -1, h_tap.shape[-1])
        )
        
        adj_matrix = self.interpreter(salient_embs)
        bias_tensor = self.compiler.compile(adj_matrix, top_k_indices, seq_l)
        
        if not intervention_active:
            # CLEAN PASS 2 (No bias)
            with torch.no_grad():
                out_p2 = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
                prediction = self.tokenizer.decode(out_p2.sequences[0][seq_l:], skip_special_tokens=True).strip()
            return {
                "prediction": prediction,
                "top_k_indices": top_k_indices.tolist(),
                "top_k_tokens": [self.tokenizer.decode([inputs.input_ids[0, idx]]) for idx in top_k_indices[0]],
                "telemetry": [{"bias_active": False}],
                "token_ids": out_p2.sequences[0][seq_l:].tolist()
            }

        # --- PASS 2: BIASED EXECUTION ---
        intervention_config = {
            layer_idx: list(range(self.model.config.num_attention_heads))
            for layer_idx in self.config["intervention_layers"]
        }
        
        # Prepare per-layer biases (v0: same bias for all intervention layers)
        layer_biases = {
            layer_idx: bias_tensor 
            for layer_idx in self.config["intervention_layers"]
        }
        
        with apply_m18_bias(self.model, self.config["intervention_layers"], intervention_config, layer_biases) as intervenors:
            with torch.no_grad():
                # Autoregressive generation of the answer
                # MANDATORY: output_attentions=True to activate the hooks
                out_p2 = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
                prediction = self.tokenizer.decode(out_p2.sequences[0][seq_l:], skip_special_tokens=True).strip()
                
        # --- TELEMETRY ---
        telemetry = [inv.telemetry for inv in intervenors]
        
        return {
            "prediction": prediction,
            "top_k_indices": top_k_indices.tolist(),
            "top_k_tokens": [self.tokenizer.decode([inputs.input_ids[0, idx]]) for idx in top_k_indices[0]],
            "telemetry": telemetry,
            "token_ids": out_p2.sequences[0][seq_l:].tolist()
        }
