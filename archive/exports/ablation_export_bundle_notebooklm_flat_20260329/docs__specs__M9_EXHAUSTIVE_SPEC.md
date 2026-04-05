# THE M9.1 MASTER BLUEPRINT: THE LOJBAN SYMBIOTE

**Status:** ACTIVE - MANDATORY ARCHITECTURE
**Revision:** 9.1 (The AdaHessian & Taxonomy Update)

## 1. The Differential Physics (AdaHessian)
We replace standard AdamW with **AdaHessian** in the Forge.
- **Hutchinson's Trace Approximation:** We compute the Hessian diagonal approximation using randomized Rademacher sampling.
- **Memory Optimization:** We isolate the second-order derivative calculation strictly to the `nn.Embedding(2256, 896)` projection head. This keeps memory overhead linear ($O(d)$) and prevents the "Memory Trap."
- **Geometric Topology:** The optimizer restricts step sizes in the "steep ravines" of initialized gismu/cmavo while taking aggressive steps in the "flat plateaus" of the Gaussian noise.

## 2. The Symbiote Taxonomy (2256-Token Partition)
The vocabulary matrix is a rigorously partitioned `2256 x 896` tensor:
- **Indices 0-49 (gismu):** Semantic relations (e.g., nenri, rinka) - Seeded from Qwen.
- **Indices 50-99 (cmavo):** Pure logic gates (AND, NOT) and scope brackets - Seeded from Qwen.
- **Indices 100-1999 (Emergent Playground):** Pure Gaussian noise for operator evolution.
- **Indices 2000-2255 (Ptr):** Learned positional routing cables mapping to System 2's context window.

## 3. The Tokenizer Sync (The Semantic Bridge)
To fix the 25% inference plateau, we physically sync the vocabularies:
- **Resizing:** Expand System 2's AutoTokenizer and Embedding layers to include the new Lojban tokens.
- **Hard-Copy:** Direct copy of System 1's trained 896D vectors into System 2's **Input Embeddings** AND **LM Head (Output Head)** to ensure non-zero prediction probability for logic tokens.

## 4. The ZeroMQ Concurrency (Batching N=16)
To prevent GPU idling, the Harvester and Forge use non-blocking, batched communication.
- **Batch Size:** $N=16$.
- **Flow:** Harvester fires a `[16, seq_len, 896]` tensor. Forge processes the entire batch via matrix multiplication and returns 16 AST graphs simultaneously.
