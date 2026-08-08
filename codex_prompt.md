Read the following file locally: D:\lojbanhypop\src\lojban_evolution\models\architectures\gflownet_symbiote.py

I need you to implement an architectural fix for a causal mediation probe flaw. Currently, the Answer Head uses mean pooling which is permutation invariant. We must replace it with an order-sensitive LSTM.

In both `M29StarQFormerSymbiote` and `M29GFlowNetSymbiote`:
1. In `__init__`, replace the `self.answer_head = nn.Sequential(...)` block with two distinct layers:
   ```python
   self.answer_head_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
   self.answer_head_mlp = nn.Sequential(
       nn.Linear(hidden_dim, hidden_dim),
       nn.Tanh(),
       nn.Linear(hidden_dim, len(ANSWER_LABELS))
   )
   ```
2. In the `forward` pass, find where the `trace_state` is calculated via `.sum(dim=1) / max(...)`. Replace it with:
   ```python
   # final_embeddings is the trace sequence (batch, seq_len, hidden_dim)
   _, (h_n, _) = self.answer_head_rnn(final_embeddings)
   trace_state = h_n[-1] # Extract final hidden state
   answer_logits = self.answer_head_mlp(trace_state)
   ```
   Do this anywhere the model pools the trace for the answer head (e.g., traces 1 and 2 in the GFlowNet).

Implement this change directly into `D:\lojbanhypop\src\lojban_evolution\models\architectures\gflownet_symbiote.py`.
