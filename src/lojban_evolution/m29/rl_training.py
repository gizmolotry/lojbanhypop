from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lojban_evolution.m29.model import M29StarQFormerSymbiote


def _gold_trace_from_batch(
    batch: dict[str, Any],
    *,
    num_queries: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return M29's 0..6 loose-symbol type trace, padded to num_queries."""

    if "type_targets" in batch:
        raw = batch["type_targets"].to(device=device, dtype=torch.long)
        out = torch.zeros(raw.shape[0], int(num_queries), dtype=torch.long, device=device)
        keep = min(int(num_queries), int(raw.shape[1]))
        if keep > 0:
            clipped = raw[:, :keep]
            valid = (clipped >= 0) & (clipped < int(vocab_size))
            prefix_valid = valid.long().cumprod(dim=1).bool()
            out[:, :keep] = torch.where(prefix_valid, clipped, torch.zeros_like(clipped))
        return out

    rows = batch.get("examples") or batch.get("loose_symbols")
    if rows is None:
        raise KeyError("M29 RL training requires batch['type_targets'], batch['examples'], or batch['loose_symbols'].")

    traces: list[list[int]] = []
    for row in rows:
        symbols = getattr(row, "loose_symbols", row)
        trace: list[int] = []
        for symbol in symbols:
            token = int(getattr(symbol, "type_id", symbol[0] if isinstance(symbol, (tuple, list)) else symbol))
            if token < 0 or token >= int(vocab_size):
                break
            trace.append(token)
            if len(trace) >= int(num_queries):
                break
        trace = trace[: int(num_queries)]
        trace += [0] * (int(num_queries) - len(trace))
        traces.append(trace)
    return torch.tensor(traces, dtype=torch.long, device=device)


def _trace_state_from_tokens(model: M29StarQFormerSymbiote, tokens: torch.Tensor) -> torch.Tensor:
    embeddings = model.bridge.token_embeddings(tokens.long())
    _, (h_n, _) = model.answer_head_rnn(embeddings)
    return h_n[-1]


def train_m29_rl_symbiote(
    model: M29StarQFormerSymbiote,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    samples_per_prompt: int = 10,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """
    Unified M29 cold-start training loop.

    Epochs 1-3 perform supervised teacher forcing against the M25 loose-symbol
    type trace. Later epochs switch to REINFORCE with a batch reward baseline,
    adding dense syntax rewards for matching the supervised trace.
    """

    device_obj = torch.device(device)
    model.to(device_obj)

    history: list[dict[str, float]] = []
    warmup_epochs = min(3, int(epochs))
    sample_count = max(1, int(samples_per_prompt))

    for epoch in range(int(epochs)):
        model.train()
        warmup = epoch < warmup_epochs
        epoch_stats = {
            "loss": 0.0,
            "generator_loss": 0.0,
            "answer_loss": 0.0,
            "reinforce_loss": 0.0,
            "mean_reward": 0.0,
            "answer_reward": 0.0,
            "dense_reward": 0.0,
            "accuracy": 0.0,
            "positive_trace_fraction": 0.0,
            "total_generated": 0.0,
            "total_positive": 0.0,
            "mode": 0.0 if warmup else 1.0,
        }
        batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device_obj)
            expected_answers = batch["answer_id"].to(device_obj)
            gold_trace = _gold_trace_from_batch(
                batch,
                num_queries=model.bridge.num_queries,
                vocab_size=model.bridge.vocab_size,
                device=device_obj,
            )

            with torch.no_grad():
                language_outputs = model.language_backbone(input_ids)
                prompt_hidden_states = language_outputs["token_hidden_states"]

            optimizer.zero_grad(set_to_none=True)

            if warmup:
                _, _, logits = model.bridge(prompt_hidden_states)
                generator_loss = F.cross_entropy(
                    logits.reshape(-1, model.bridge.vocab_size),
                    gold_trace.reshape(-1),
                )
                answer_logits = model.answer_head_mlp(_trace_state_from_tokens(model, gold_trace))
                answer_loss = F.cross_entropy(answer_logits, expected_answers)
                loss = generator_loss + answer_loss

                accuracy = (answer_logits.argmax(dim=-1) == expected_answers).float().mean()
                reinforce_loss = loss.new_zeros(())
                mean_reward = loss.new_zeros(())
                answer_reward = accuracy
                dense_reward = (logits.argmax(dim=-1) == gold_trace).float().sum(dim=-1).mul(0.1).mean()
                positive_fraction = loss.new_zeros(())
                total_generated = float(input_ids.shape[0])
                total_positive = float((answer_logits.argmax(dim=-1) == expected_answers).sum().detach().cpu().item())
            else:
                sample_log_probs: list[torch.Tensor] = []
                sample_rewards: list[torch.Tensor] = []
                sample_answer_rewards: list[torch.Tensor] = []
                sample_dense_rewards: list[torch.Tensor] = []
                sample_answer_logits: list[torch.Tensor] = []

                for _ in range(sample_count):
                    sampled_tokens, _, logits = model.bridge(prompt_hidden_states)
                    token_log_probs = F.log_softmax(logits, dim=-1).gather(
                        dim=-1,
                        index=sampled_tokens.unsqueeze(-1),
                    ).squeeze(-1)

                    trace_state = _trace_state_from_tokens(model, sampled_tokens)
                    answer_logits = model.answer_head_mlp(trace_state)
                    predicted_answers = answer_logits.argmax(dim=-1)

                    answer_hits = (predicted_answers == expected_answers).float()
                    # Step-level dense hits! Shape: (batch, seq_len)
                    step_dense_rewards = (sampled_tokens == gold_trace).float() * 0.1
                    step_rewards = step_dense_rewards.clone()
                    # Add answer hit to the final step of the sequence
                    step_rewards[:, -1] += answer_hits

                    sample_log_probs.append(token_log_probs)
                    sample_rewards.append(step_rewards.detach())
                    sample_answer_rewards.append(answer_hits.detach())
                    sample_dense_rewards.append(step_dense_rewards.sum(dim=-1).detach())
                    sample_answer_logits.append(answer_logits)

                # Stack dimensions: [sample_count, batch_size, seq_len]
                log_probs = torch.stack(sample_log_probs, dim=0)
                rewards = torch.stack(sample_rewards, dim=0)
                answer_rewards = torch.stack(sample_answer_rewards, dim=0)
                dense_rewards = torch.stack(sample_dense_rewards, dim=0)
                answer_logits_all = torch.stack(sample_answer_logits, dim=0)

                # Step-level baseline and advantages
                baseline = rewards.mean(dim=0, keepdim=True)
                advantages = rewards - baseline
                
                # Step-level REINFORCE loss: log_probs * advantages, sum over sequence, mean over samples/batch
                reinforce_loss = -(log_probs * advantages).sum(dim=-1).mean()

                trajectory_rewards = rewards.sum(dim=-1)
                positive_mask = trajectory_rewards > 0.0
                expanded_answers = expected_answers.unsqueeze(0).expand(sample_count, -1)
                if bool(positive_mask.any().item()):
                    answer_loss = F.cross_entropy(answer_logits_all[positive_mask], expanded_answers[positive_mask])
                else:
                    answer_loss = reinforce_loss.new_zeros(())

                loss = reinforce_loss + answer_loss

                accuracy = answer_rewards.mean()
                mean_reward = trajectory_rewards.mean()
                answer_reward = answer_rewards.mean()
                dense_reward = dense_rewards.mean()
                positive_fraction = positive_mask.float().mean()
                total_generated = float(input_ids.shape[0] * sample_count)
                total_positive = float(positive_mask.sum().detach().cpu().item())
                generator_loss = reinforce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_stats["loss"] += float(loss.detach().cpu().item())
            epoch_stats["generator_loss"] += float(generator_loss.detach().cpu().item())
            epoch_stats["answer_loss"] += float(answer_loss.detach().cpu().item())
            epoch_stats["reinforce_loss"] += float(reinforce_loss.detach().cpu().item())
            epoch_stats["mean_reward"] += float(mean_reward.detach().cpu().item())
            epoch_stats["answer_reward"] += float(answer_reward.detach().cpu().item())
            epoch_stats["dense_reward"] += float(dense_reward.detach().cpu().item())
            epoch_stats["accuracy"] += float(accuracy.detach().cpu().item())
            epoch_stats["positive_trace_fraction"] += float(positive_fraction.detach().cpu().item())
            epoch_stats["total_generated"] += total_generated
            epoch_stats["total_positive"] += total_positive
            batches += 1

        if batches > 0:
            for key in (
                "loss",
                "generator_loss",
                "answer_loss",
                "reinforce_loss",
                "mean_reward",
                "answer_reward",
                "dense_reward",
                "accuracy",
                "positive_trace_fraction",
            ):
                epoch_stats[key] /= batches

        mode = "warmup" if warmup else "rl"
        print(
            f"Epoch {epoch + 1}/{epochs} [{mode}] | "
            f"Loss: {epoch_stats['loss']:.4f} | "
            f"Reward: {epoch_stats['mean_reward']:.4f} | "
            f"Acc: {epoch_stats['accuracy']:.4f}"
        )
        history.append(epoch_stats)

    return history
