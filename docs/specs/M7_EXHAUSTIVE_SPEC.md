# THE FULL M7 ARCHITECTURE: THE INTERLEAVED LATENT COPROCESSOR

**Status:** INITIATED ON BRANCH `m7_interleaved_coprocessor`

## 1. The Topology: The Synchronous Loop
We un-sever the bridge. System 2 (the continuous English LLM) is no longer blinded. It is the primary sequence generator. System 1 (the Lojban LoRA) becomes a synchronous co-processor.

Instead of System 1 doing all the thinking and handing a compressed ball of math to a deaf System 2, they take turns.

## 2. The M7 Forward Pass (The Physics of the Step)
- **Step 1 (Semantic Reading):** System 2 reads the prompt. It generates its continuous hidden states. It "knows" the entities involved because it holds their massive, high-dimensional semantic embeddings in its KV cache.
- **Step 2 (The Trigger):** When System 2 hits a logical bottleneck, it emits a specific latent control token: `<CALL_ADVISOR>`.
- **Step 3 (The Handoff):** The continuous embedding of `<CALL_ADVISOR>` is routed entirely out of System 2 and fed into System 1 as its input state.
- **Step 4 (The Discrete Choke):** System 1 receives the continuous state and is forced through the strict topology. It must select a Blank Slate Codebook operator (e.g., G1713) and bind it to specific Hard Pointers reaching back into System 2's prompt. Output: `[G1713, Ptr(Councilmen), Ptr(Demonstrators)]`.
- **Step 5 (The Injection):** This discrete, legally bound Lojbanic matrix is embedded back into continuous space via a bottleneck and injected directly into System 2's residual stream.
- **Step 6 (The Resolution):** System 2 attends to this newly injected rigid constraint. It updates its continuous worldview based on the strict causal relationship System 1 just defined, and outputs the final English answer.

## 3. The Semantic/Syntactic Schism (The Death of the Cheat)
In M6, System 1 had to tell System 2 the answer ("demonstrators"), so it stuffed that massive semantic vector into the `[OP]` token.
In M7, System 2 already has the word "demonstrators" in its prompt. It doesn't need System 1 to retrieve the dictionary definition. It needs System 1 to define the physics of the interaction.

Because System 2 retains the semantic payload, System 1 is mathematically starved of any incentive to encode noun-meaning. The reconstruction loss organically forces System 1's Codebook tokens to represent pure, abstract syntax—causality, friction, containment, negation.

## 4. The Engineering Mandates
- **The Interleaved Router:** PyTorch routing logic that pauses S2, queries S1, and injects the result.
- **The Blank Slate Initialization:** Codebook is absolutely purged of [AND], [OR], [NOT]. 
- **The Vector Choke:** Embedding dimension of Codebook tokens injected back into System 2 is bottlenecked to a low dimension ($d=16$ or $32$). It carries syntax, but is physically too small to carry semantic nouns.
