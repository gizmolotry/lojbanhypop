# M5 Auto-Formalization

- baseline_id: `M_BASE_20260310_REFAC`
- checkpoint_in: `runs/l_series/l6_ablation/refactor_clean/20260310_221033/l6-c/20260310_222234/l_series_checkpoint.pt`
- dataset_profile: `semantic_bench_v1`
- difficulty_tier: `all`
- relation_vocab: `16`

| Run | Name | Status | Op Entropy | Top1 | Family Score | Lex Probe | Lex Adv Loss | Lex Adv Acc | Scope | Identity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M5.A | Reuse-oriented control | ok | 2.7677 | 1.0000 | 0.8216 | 0.1136 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| M5.B | Selective lexical adversary + reuse | ok | 2.7672 | 0.9491 | 0.7737 | 0.1364 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| M5.C | Selective lexical adversary + family clustering | ok | 2.7481 | 1.0000 | 0.8216 | 0.1136 | 4.1082 | 0.0000 | 0.0000 | 0.0000 |