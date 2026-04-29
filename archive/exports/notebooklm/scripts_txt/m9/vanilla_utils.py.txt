from __future__ import annotations
import torch
from typing import Tuple, Sequence

def trace_to_english(trace: Sequence[str]) -> str:
    """
    Converts a symbolic logic trace to a simple English CoT.
    """
    mapping = {
        "TASK_WINOGRAD": "This is a Winograd schema problem.",
        "TASK_WINOGRAD_BENCH": "This is a Winograd benchmark problem.",
        "TASK_MULTI_AGENT": "This is a multi-agent theory of mind puzzle.",
        "TASK_KNIGHTS": "This is a knights and knaves logic puzzle.",
        "TASK_SPATIAL": "This is a spatial relationship puzzle.",
        "TASK_TEMPORAL": "This is a temporal ordering puzzle.",
        "TASK_QUANTIFIER": "This is a quantifier equivalence puzzle.",
        "TASK_COMPARATIVE": "This is a comparative logic puzzle.",
        "TASK_NESTED_SCOPE": "This is a nested scope logic puzzle.",
        "TASK_DEONTIC": "This is a deontic policy puzzle.",
        "TASK_COUNTERFACTUAL": "This is a counterfactual reasoning puzzle.",
        
        "BIND_E1": "First, let's identify the primary entity.",
        "BIND_E2": "Next, let's identify the secondary entity.",
        "BIND_AGENT_E1": "Let's identify the first agent.",
        "BIND_AGENT_E2": "Let's identify the second agent.",
        "BIND_OBJ": "Let's identify the object being moved.",
        "BIND_LOC": "Let's identify the location.",
        "BIND_A": "Let's identify agent A.",
        "BIND_B": "Let's identify agent B.",
        "BIND_C": "Let's identify agent C.",
        
        "LINK_CAUSAL": "Let's analyze the causal relationship.",
        "LINK_SPATIAL": "Let's analyze the spatial relationship.",
        "LINK_TEMPORAL": "Let's analyze the temporal sequence.",
        
        "PRONOUN_REF": "We need to resolve the pronoun reference.",
        "RESOLVE_PRON_E1": "The pronoun refers to the first entity.",
        "RESOLVE_PRON_E2": "The pronoun refers to the second entity.",
        "RESOLVE_REFERENT": "Let's find the correct referent.",
        
        "STATE_OBJ_LOC1": "The object is initially at the first location.",
        "EVENT_MOVE_LOC2": "An event moves the object to a second location.",
        "OBS_PRESENT": "The observer is present during the move.",
        "OBS_ABSENT": "The observer is absent during the move.",
        "MENTAL_STATE_E1_LOC1": "The agent believes the object is at the first location.",
        "MENTAL_STATE_E1_LOC2": "The agent believes the object is at the second location.",
        
        "CLAIM_GRAPH": "Let's map the claims made by the agents.",
        "ASSUME_BRANCH": "Let's assume one of the claims is true.",
        "CONSISTENCY_CHECK": "Checking for logical consistency.",
        
        "CHAIN_TRANSITIVE": "Applying transitive logic.",
        "SCOPE_BIND": "Binding the scope of the quantifiers.",
        "CHECK_EQUIV": "Checking if the statements are equivalent.",
        
        "VERIFY_ID": "Verifying the identity of the target.",
        "VERIFY_TRUE": "The condition is true.",
        "VERIFY_FALSE": "The condition is false.",
        
        "ANS_YES": "Therefore, the answer is yes.",
        "ANS_NO": "Therefore, the answer is no.",
        "ANS_E1": "Therefore, the answer is the first entity.",
        "ANS_E2": "Therefore, the answer is the second entity.",
    }
    
    steps = [mapping.get(t, f"Analyzing {t}...") for t in trace]
    return " ".join(steps)

if __name__ == "__main__":
    # Test a few
    print(trace_to_english(("TASK_WINOGRAD", "BIND_E1", "LINK_CAUSAL", "ANS_E1")))
