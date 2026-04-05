from __future__ import annotations
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from ..experiment import generate_dataset, Problem

@dataclass
class M6GroundedStep:
    op_idx: int
    pointers: List[int] # Indices into prompt tokens

class M6DataEngine:
    """
    M6 J-Series Data Engine.
    Wraps existing logic puzzles and decomposes them into grounded M6 traces.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # OP_QUOTE=0, OP_STOP=1, PAD=2 from matrix_core.py spec
        self.OP_QUOTE = 0
        self.OP_STOP = 1
        self.PAD = 2
        
        # Mapping high-level concepts to emergent gismu range (5-1999)
        # We start by grounding basic causal/logical states.
        self.GISMU_MAP = {
            "LINK_CAUSAL": 10,
            "PRONOUN_REF": 11,
            "RESOLVE_PRON_E1": 12,
            "RESOLVE_PRON_E2": 13,
            "VERIFY_ID": 14,
            "ANS_E1": 15,
            "ANS_E2": 16,
            "STATE_OBJ_LOC1": 17,
            "EVENT_MOVE_LOC2": 18,
            "MENTAL_STATE_E1_LOC1": 19,
            "MENTAL_STATE_E1_LOC2": 20,
        }

    def _find_entity_index(self, prompt: str, entity: str) -> int:
        """Finds the token index of a specific entity in the prompt."""
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        # Simple heuristic: re-tokenize entity and find position
        e_ids = self.tokenizer(entity, add_special_tokens=False).input_ids
        for i in range(len(ids) - len(e_ids) + 1):
            if ids[i : i + len(e_ids)] == e_ids:
                return i
        return 0

    def ground_problem(self, prob: Problem) -> List[M6GroundedStep]:
        """
        Decomposes a high-level Problem into a sequence of M6 Grounded Steps.
        """
        grounded_trace = []
        
        # 1. Borrow key entities from the prompt (OP_QUOTE)
        # We extract nouns based on the problem type
        entities = []
        if "TASK_WINOGRAD" in prob.trace:
            # Winograd usually has two main entities
            words = prob.prompt.split()
            # Extremely simple heuristic for demo
            entities = [prob.answer] 
        
        for ent in entities:
            idx = self._find_entity_index(prob.prompt, ent)
            grounded_trace.append(M6GroundedStep(op_idx=self.OP_QUOTE, pointers=[idx]))

        # 2. Map logical task tokens to gismu slots
        for t in prob.trace:
            if t in self.GISMU_MAP:
                grounded_trace.append(M6GroundedStep(op_idx=self.GISMU_MAP[t], pointers=[]))
        
        # 3. Explicit Termination
        grounded_trace.append(M6GroundedStep(op_idx=self.OP_STOP, pointers=[]))
        
        return grounded_trace

def get_m6_dataloader(size: int, tokenizer, seed: int = 7):
    probs = generate_dataset(size=size, seed=seed)
    engine = M6DataEngine(tokenizer)
    
    dataset = []
    for p in probs:
        m6_trace = engine.ground_problem(p)
        dataset.append({
            "problem": p,
            "m6_trace": m6_trace
        })
    return dataset
