from __future__ import annotations
from typing import Dict, List

# M11 Arity Registry: Maps Gismu ID to the number of allowed place structure slots (m_g)
# Default is 5 (Lojban standard), but many gismu use fewer in practice or for specific tasks.
ARITY_REGISTRY: Dict[int, int] = {
    # Winograd / Physical Commonsense
    0: 2, # barda (big): x1 is big in dimension x2
    1: 2, # cmalu (small): x1 is small in dimension x2
    2: 1, # titnan (weak - approximate): x1 is weak
    3: 1, # tsani (strong - approximate): x1 is strong
    4: 2, # terpa (fear): x1 fears x2
    5: 2, # sruma (assume): x1 assumes x2
    6: 2, # krici (believe): x1 believes x2
    
    # Spatial / Relational
    10: 2, # nenri (inside): x1 is inside x2
    11: 2, # zunle (left): x1 is to the left of x2
    12: 2, # pritu (right): x1 is to the right of x2
}

def get_arity(gismu_id: int) -> int:
    return ARITY_REGISTRY.get(gismu_id, 2) # Default to 2 for the current task set

# Role Labels for interpretability and cross-predicate transfer (Optional Phase 2)
ROLE_LABELS: Dict[int, List[str]] = {
    4: ["feared_by", "fear_source"],
    10: ["contained", "container"],
}
