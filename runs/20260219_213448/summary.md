# Language Evolution Run Summary

- Dataset size: 1000 (train=600, val=200, test=200)
- Iterations: 6
- Final vocabulary: 32 tokens (21 base + 11 macros)
- Test accuracy: 1.0000
- Test avg tokens: 2.9850
- Test parse success: 1.0000

## Iteration Snapshots

- Iteration 0: avg_tokens=7.5650, accuracy=1.0000, accepted=2, language_size=23
- Iteration 1: avg_tokens=5.4500, accuracy=1.0000, accepted=3, language_size=26
- Iteration 2: avg_tokens=4.5150, accuracy=1.0000, accepted=2, language_size=28
- Iteration 3: avg_tokens=3.3550, accuracy=1.0000, accepted=3, language_size=31
- Iteration 4: avg_tokens=3.0250, accuracy=1.0000, accepted=1, language_size=32
- Iteration 5: avg_tokens=3.0250, accuracy=1.0000, accepted=0, language_size=32

## Accepted Macros

- M001: TASK_WINOGRAD BIND_E1 BIND_E2 LINK_CAUSAL
- M010: TASK_KNIGHTS BIND_A BIND_B BIND_C
- M011: CLAIM_GRAPH ASSUME_BRANCH CONSISTENCY_CHECK
- M014: VERIFY_ID ANS_ROLE_MAP
- M015: TASK_MULTI_AGENT BIND_AGENT_E1 BIND_AGENT_E2 BIND_OBJ_O1
- M021: STATE_OBJ_LOC1 EVENT_MOVE_LOC2
- M022: STATE_OBJ_LOC1 EVENT_MOVE_LOC2 OBS_ABSENT MENTAL_STATE_E1_LOC1
- M031: VERIFY_ID ANS_LOC1
- M032: PRONOUN_REF RESOLVE_PRON_E1 VERIFY_ID ANS_E1
- M038: PRONOUN_REF RESOLVE_PRON_E2 VERIFY_ID ANS_E2
- M041: OBS_PRESENT MENTAL_STATE_E1_LOC2 VERIFY_ID ANS_LOC2
