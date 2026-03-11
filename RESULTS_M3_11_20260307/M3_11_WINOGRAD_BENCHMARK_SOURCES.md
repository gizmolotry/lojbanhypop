# M3.11 Benchmark Sources

M3.11 corpus expansion is benchmark-inspired and grounded in these references:

- Winograd Schema Challenge (official site): https://www.winograd-schema-challenge.org/
- WinoGrande (Sakaguchi et al., 2020, AAAI): https://arxiv.org/abs/1907.10641
- Definite Pronoun Resolution Dataset (Rahman and Ng, 2012): https://aclanthology.org/P12-1093/
- SuperGLUE task suite (includes WSC-style pronoun resolution): https://super.gluebenchmark.com/tasks

Notes:

- We do not copy benchmark items verbatim at scale; we use schema-level patterns (causal inversion, role assignment, adjective/property inversion, lexical paraphrase variants).
- The generated corpus is intended for internal ablations and failure anatomy, not benchmark score reporting.
