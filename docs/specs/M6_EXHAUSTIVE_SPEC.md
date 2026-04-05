# THE FULL M6 ARCHITECTURE: EXHAUSTIVE SPECIFICATION

**Status:** INITIATED ON BRANCH `m6_neuro_symbolic_engine`
**Goal:** Clean separation. Do not spoil the legacy H5/M5 repo.

## 1. The Strict Neuro-Symbolic Bottleneck (The Severed Bridge)
To permanently kill the "Decorative Side-Car" cheat, System 2 is violently severed into two mathematically isolated halves.

*   **System 2a (The Encoder):** Reads the English prompt, embeds it, and serves as the continuous lookup dictionary. It then completely shuts off. It cannot process logic.
*   **System 1 (The Lojban LoRA):** The exclusive reasoning engine. It takes the context from System 2a and runs the logic in an isolated autoregressive void.
*   **System 2b (The Decoder):** Mathematically lobotomized. Blind to the original English prompt's causal actions. It only reads System 1's final emitted `<STOP>` matrix to guess the heavily masked target word.

## 2. The Topological Bedrock (LC6 Constraints)
All backprop band-aids (GRL, Uniformity, Sparsity, Brivi Gates) are DELETED. The only regulatory mathematics surviving are the LC6 Structural Legality Constraints ($L_{struct}$):

*   **The Padded N-Ary Matrix:** System 1 must output tensors in a strictly fixed 10-slot width: `[OP, x1, x2, x3... x9]`.
*   **Dynamic Arity via Decay:** The network is not told how many variables to use. It fills what it needs and forces unused $x$ slots to decay to `<PAD>`.
*   **Termination:** System 1 must physically emit the `[<STOP>]` operator matrix to hand control back to System 2b.

## 3. The 3 COCONUT Streams (The Continuous Physics)
The architecture operates three simultaneous continuous (latent) data streams:

1.  **Internal State:** As System 1 chains its matrices, it passes its continuous hidden state back into its own key-value cache.
2.  **The Hard Pointers:** System 1 uses $x$ slots as Hard Pointer Indices, copying exact continuous tensors of English words (e.g., "Alex") from System 2a directly into its variable slots.
3.  **The Resolution:** System 1’s final continuous payload (the operator tensor applied to pointer tensors) flows across the bridge to System 2b to unlock the masked target.

## 4. The Vocabulary: Auto-Formalization & Lojban Borrowing
The $K=2000$ VQ bottleneck operates with absolute linguistic precision, separating the gismu (operator) from the sumti (arguments).

*   **The Quote Operator:** A hardcoded token `[OP_QUOTE]`. Invokes the Pointer Network to borrow English nouns from System 2a without translating them.
*   **The Emergent Gismu:** The remaining 1,900+ tokens are auto-formalized purely through reconstruction pressure.
*   **Emergent Causal Geometry:** System 1 mathematically locks positional slots (e.g., $x_1$ = Giver, $x_2$ = Receiver for Transfer) to prevent reconstruction failure.

## 5. The J-Series Data Engine (Multi-Hop State Tracking)
One-shot Winograd schemas are dead. The J-Series data loader dynamically generates **Zero-Prior Logic Puzzles** (e.g., "The Glorp contains a Frazz"). System 1 is mathematically starved and must use `[OP_QUOTE]` to track artificial entities across multiple steps (`[OP_CONTAIN] -> [OP_PROPERTY] -> [OP_DAMAGE]`).

## 6. The Step-by-Step Matrix Trace (The Bridi)
A perfectly grounded, human-readable proof unrolls as follows:

```text
Prompt: The Glorp hit the Frazz. The Frazz is highly acidic. What happens to the Glorp?
Read/Borrow: [OP_QUOTE, Ptr(Loc:2 "Glorp"), Ptr(Loc:5 "Frazz"), <PAD>...]
Acknowledge Physics: [OP_PROPERTY_ACIDIC, Ptr(Loc:5 "Frazz"), <PAD>...]
Deduce Interaction: [OP_IMPACT_TRANSFER, Ptr(Loc:2 "Glorp"), Ptr(Loc:5 "Frazz"), <PAD>...]
Calculate State: [OP_CAUSAL_CORROSION, Ptr(Loc:2 "Glorp"), <PAD>...]
Terminate: [<STOP>, <PAD>, <PAD>, <PAD>...]
```
