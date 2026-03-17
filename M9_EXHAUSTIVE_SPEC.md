# THE FULL M9 ARCHITECTURE: THE CONTRASTIVE NLI ENGINE

**Status:** INITIATED ON BRANCH `m9_contrastive_nli_engine`

## 1. The Death of the Metric Illusion
- **The Reality:** The true zero-shot capability of the 0.5B Qwen model on complex, multi-hop logic is hovering around 5%.
- **The Implication:** We are trying to cure a mathematically blind model. We strictly use zero-fallback, exact-extraction grading.

## 2. The Architectural Inversion (The True Symbiote)
- **The Fix:** The logic must be born of the English reasoning.
- **The Execution:** System 2 reads the prompt and generates a continuous semantic workspace first. System 1 then reads those active hidden states and translates the LLM's intent into a strict, formal Lojbanic AST graph, injecting it back to keep System 2 from derailing.

## 3. The 896D Bottleneck Restoration
- **The Physics:** The discrete selection (Gumbel-Softmax over K=2000 tokens) is the true bottleneck, not the embedding dimension.
- **The Fix:** System 1 is restored to the full $d=896$ dimensions. It has full geometric bandwidth but cannot smuggle semantic cheats because a discrete token embedding is globally static.

## 4. The NLI Contrastive Engine (The Mathematical Forge)
- **The Fix:** Training is reframed as Latent Natural Language Inference (NLI).
- **The Math (InfoNCE):**
  - **Premise:** System 2's continuous English reasoning.
  - **Hypothesis:** System 1's generated Lojban graph.
  - The InfoNCE Contrastive Loss pulls valid logical graphs closer to the English semantic states and repels hallucinated graphs (e.g., pointing at punctuation).

## 5. The ZeroMQ Decoupled Pipeline
- **The Fix:** The codebase is split into two decoupled microservices communicating via asynchronous ZeroMQ sockets.
- **Process 1 (The Harvester):** Loads the base model, generates English reasoning, extracts 896-dimensional hidden states, and `PUSH`es them to the socket.
- **Process 2 (The Forge):** `PULL`s the tensors, generates positive/negative Lojban graphs, and runs the NLI Contrastive backpropagation on the LoRA.
