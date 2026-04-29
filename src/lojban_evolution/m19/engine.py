from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, List
from contextlib import contextmanager


def _off_diagonal_cosines(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D tensor for pairwise cosine stats, got shape {tuple(vectors.shape)}")
    count = int(vectors.shape[0])
    if count < 2:
        return torch.empty(0, device=vectors.device, dtype=vectors.dtype)
    normalized = F.normalize(vectors, dim=-1)
    cosine = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(count, device=vectors.device, dtype=torch.bool)
    return cosine.masked_select(mask)


def pairwise_cosine_stats(vectors: torch.Tensor) -> Dict[str, float]:
    off_diag = _off_diagonal_cosines(vectors)
    if off_diag.numel() == 0:
        return {
            "pairwise_cosine_mean": 0.0,
            "pairwise_cosine_max": 0.0,
            "pairwise_cosine_min": 0.0,
            "pairwise_cosine_std": 0.0,
            "anisotropy": 0.0,
        }
    return {
        "pairwise_cosine_mean": float(off_diag.mean().item()),
        "pairwise_cosine_max": float(off_diag.max().item()),
        "pairwise_cosine_min": float(off_diag.min().item()),
        "pairwise_cosine_std": float(off_diag.std(unbiased=False).item()),
        "anisotropy": float(off_diag.abs().mean().item()),
    }


def batched_pairwise_cosine_stats(trace: torch.Tensor, lengths: torch.Tensor | None = None) -> Dict[str, float]:
    if trace.ndim != 3:
        raise ValueError(f"Expected 3D trace tensor, got shape {tuple(trace.shape)}")
    if lengths is None:
        lengths = torch.full((trace.shape[0],), trace.shape[1], device=trace.device, dtype=torch.long)
    values: list[torch.Tensor] = []
    for batch_idx in range(trace.shape[0]):
        active = int(lengths[batch_idx].item())
        if active < 2:
            continue
        off_diag = _off_diagonal_cosines(trace[batch_idx, :active, :])
        if off_diag.numel() > 0:
            values.append(off_diag)
    if not values:
        return {
            "pairwise_cosine_mean": 0.0,
            "pairwise_cosine_max": 0.0,
            "pairwise_cosine_min": 0.0,
            "pairwise_cosine_std": 0.0,
            "anisotropy": 0.0,
        }
    merged = torch.cat(values, dim=0)
    return {
        "pairwise_cosine_mean": float(merged.mean().item()),
        "pairwise_cosine_max": float(merged.max().item()),
        "pairwise_cosine_min": float(merged.min().item()),
        "pairwise_cosine_std": float(merged.std(unbiased=False).item()),
        "anisotropy": float(merged.abs().mean().item()),
    }


def operator_distribution_stats(op_logits: torch.Tensor, lengths: torch.Tensor | None = None) -> Dict[str, float]:
    if op_logits.ndim != 3:
        raise ValueError(f"Expected 3D operator logits tensor, got shape {tuple(op_logits.shape)}")
    probs = F.softmax(op_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    top1 = probs.max(dim=-1).values
    if lengths is None:
        valid = torch.ones_like(entropy, dtype=torch.bool)
    else:
        valid = torch.arange(op_logits.shape[1], device=op_logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    entropy_values = entropy.masked_select(valid)
    top1_values = top1.masked_select(valid)
    if entropy_values.numel() == 0:
        return {
            "operator_entropy_mean": 0.0,
            "operator_entropy_ratio_mean": 0.0,
            "operator_top1_share_mean": 0.0,
        }
    max_entropy = math.log(op_logits.shape[-1])
    return {
        "operator_entropy_mean": float(entropy_values.mean().item()),
        "operator_entropy_ratio_mean": float((entropy_values.mean() / max_entropy).item()),
        "operator_top1_share_mean": float(top1_values.mean().item()),
    }


def compute_query_repulsion_loss(
    query_vectors: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    off_diag = _off_diagonal_cosines(query_vectors)
    if off_diag.numel() == 0:
        return torch.zeros((), device=query_vectors.device, dtype=query_vectors.dtype)
    penalty = torch.relu(off_diag.abs() - float(margin))
    return penalty.mean()


def dictionary_health_metrics(
    query_vectors: torch.Tensor,
    query_state: torch.Tensor,
    trace: torch.Tensor,
    delta: torch.Tensor,
    op_logits: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for prefix, stats in (
        ("query_embed", pairwise_cosine_stats(query_vectors)),
        ("query_state", batched_pairwise_cosine_stats(query_state)),
        ("scratch_trace", batched_pairwise_cosine_stats(trace, lengths=lengths)),
        ("scratch_delta", batched_pairwise_cosine_stats(delta, lengths=lengths)),
        ("operator", operator_distribution_stats(op_logits, lengths=lengths)),
    ):
        for key, value in stats.items():
            metrics[f"{prefix}_{key}"] = float(value)
    return metrics


class M19MagneticCollar(nn.Module):
    def __init__(self, bottleneck_dim: int, max_positions: int = 64):
        super().__init__()
        self.subject_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.relation_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.object_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.spatial_embeddings = nn.Parameter(torch.randn(max_positions, bottleneck_dim) * 0.02)

    def forward(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if trace.shape[1] < 3:
            return torch.tensor(0.0, device=trace.device, dtype=trace.dtype)
        active_positions = self.spatial_embeddings[: trace.shape[1], :].to(device=trace.device, dtype=trace.dtype)
        trace = trace + active_positions.unsqueeze(0)
        v_sub = self.subject_proj(trace[:, 0:-2, :])
        v_rel = self.relation_proj(trace[:, 1:-1, :])
        v_obj = self.object_proj(trace[:, 2:, :])
        residual = (v_sub + v_rel - v_obj) ** 2
        if lengths is None:
            return residual.mean()
        lengths = lengths.to(device=trace.device)
        triad_positions = torch.arange(trace.shape[1] - 2, device=trace.device).unsqueeze(0)
        valid = triad_positions < (lengths.unsqueeze(1) - 2).clamp_min(0)
        if not bool(valid.any()):
            return torch.tensor(0.0, device=trace.device, dtype=trace.dtype)
        weighted = residual.mean(dim=-1)
        return (weighted * valid.to(dtype=weighted.dtype)).sum() / valid.to(dtype=weighted.dtype).sum().clamp_min(1.0)

class M19SymbioteBridge(nn.Module):
    """
    M19.3 Capacity-Agnostic Bridge:
    Supports variable query counts, bottleneck dimensions, and scratchpad lengths.
    """
    def __init__(
        self,
        hidden_size: int = 896,
        bottleneck_dim: int = 64,
        scratchpad_len: int = 8,
        num_queries: int = 8,
        max_latent_steps: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.scratchpad_len = scratchpad_len
        self.num_queries = num_queries
        self.max_latent_steps = max(int(max_latent_steps or scratchpad_len), int(scratchpad_len))
        
        # 1. Latent Queries (Extraction Heads)
        self.query_embeds = nn.Parameter(torch.randn(num_queries, bottleneck_dim) * 0.02)
        
        # 2. Cross-Attention Bottleneck
        self.compress = nn.Linear(hidden_size, bottleneck_dim)
        self.cross_attn = nn.MultiheadAttention(bottleneck_dim, num_heads=max(1, bottleneck_dim // 32), batch_first=True)
        
        # 3. Output Mapping Into A Dynamic Positional Reservoir
        self.output_map = nn.Linear(num_queries, self.max_latent_steps)
        
        # 4. Projections
        self.expand = nn.Linear(bottleneck_dim, hidden_size)
        self.collar = M19MagneticCollar(bottleneck_dim, max_positions=self.max_latent_steps)
        self.op_head = nn.Linear(bottleneck_dim, 2000)
        self.register_buffer("halt_centroid", torch.zeros(bottleneck_dim), persistent=True)
        self.register_buffer("halt_centroid_samples", torch.tensor(0.0), persistent=True)

    def forward(
        self,
        h_tap: torch.Tensor,
        active_steps: int | None = None,
        lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b = h_tap.shape[0]
        target_steps = int(active_steps or self.scratchpad_len)
        target_steps = max(1, min(target_steps, self.max_latent_steps))
        h_bottleneck = self.compress(h_tap)
        
        queries = self.query_embeds.unsqueeze(0).expand(b, -1, -1)
        h_query, _ = self.cross_attn(queries, h_bottleneck, h_bottleneck)
        h_reservoir = self.output_map(h_query.transpose(1, 2)).transpose(1, 2)
        h_scratch = h_reservoir[:, :target_steps, :]

        if lengths is None:
            lengths = torch.full((b,), target_steps, device=h_tap.device, dtype=torch.long)
        l_topo = self.collar(h_scratch, lengths=lengths)
        op_logits = self.op_head(h_scratch)
        delta = self.expand(h_scratch)
        telemetry = {
            "active_steps": target_steps,
            "halt_cosine_per_step": self.halt_similarity(h_scratch, lengths=lengths).detach(),
            "query_state": h_query.detach(),
            "trace": h_scratch.detach(),
            "delta": delta.detach(),
            "dictionary_health": dictionary_health_metrics(
                self.query_embeds.detach(),
                h_query.detach(),
                h_scratch.detach(),
                delta.detach(),
                op_logits.detach(),
                lengths=lengths.detach() if lengths is not None else None,
            ),
        }

        return delta, l_topo, op_logits, telemetry

    def update_halt_centroid(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            if trace.ndim != 3 or trace.shape[1] == 0:
                return
            if lengths is None:
                lengths = torch.full((trace.shape[0],), trace.shape[1], device=trace.device, dtype=torch.long)
            lengths = lengths.to(device=trace.device)
            samples: list[torch.Tensor] = []
            for batch_idx in range(trace.shape[0]):
                final_idx = int(lengths[batch_idx].item()) - 1
                if final_idx < 0 or final_idx >= trace.shape[1]:
                    continue
                samples.append(trace[batch_idx, final_idx, :])
            if not samples:
                return
            batch_mean = torch.stack(samples, dim=0).mean(dim=0).to(device=self.halt_centroid.device, dtype=self.halt_centroid.dtype)
            current = float(self.halt_centroid_samples.item())
            batch_count = float(len(samples))
            if current <= 0:
                self.halt_centroid.copy_(batch_mean)
                self.halt_centroid_samples.fill_(batch_count)
                return
            mixed = ((self.halt_centroid * current) + (batch_mean * batch_count)) / max(current + batch_count, 1.0)
            self.halt_centroid.copy_(mixed)
            self.halt_centroid_samples.fill_(current + batch_count)

    def halt_similarity(self, trace: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if trace.ndim != 3 or trace.shape[1] == 0:
            return torch.zeros(trace.shape[:2], device=trace.device, dtype=trace.dtype)
        centroid = self.halt_centroid.to(device=trace.device, dtype=trace.dtype)
        if float(self.halt_centroid_samples.item()) <= 0 or float(centroid.norm().item()) == 0.0:
            return torch.zeros(trace.shape[:2], device=trace.device, dtype=trace.dtype)
        sims = F.cosine_similarity(trace, centroid.view(1, 1, -1), dim=-1)
        if lengths is None:
            return sims
        lengths = lengths.to(device=trace.device)
        valid_steps = torch.arange(trace.shape[1], device=trace.device).unsqueeze(0) < lengths.unsqueeze(1)
        return sims * valid_steps.to(dtype=sims.dtype)

@contextmanager
def m19_injection_hook(model, layer_idx: int, scratchpad_mask: torch.Tensor, delta: torch.Tensor):
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        if hidden.shape[1] == scratchpad_mask.shape[1]:
            mask_indices = scratchpad_mask.to(hidden.device)
            for b in range(hidden.shape[0]):
                indices = torch.where(mask_indices[b])[0]
                # Allow partial injection if runway length differs slightly
                injection_len = min(len(indices), delta.shape[1])
                if injection_len > 0:
                    hidden[b, indices[:injection_len], :] += delta[b, :injection_len].to(hidden.dtype)
            
        return (hidden, *rest) if rest is not None else hidden

    layers = model.model.layers if hasattr(model, "model") else model.layers
    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()

def compute_m19_anti_collapse(logits: torch.Tensor, min_entropy: float = 0.85) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    max_ent = math.log(logits.shape[-1])
    entropy_ratio = entropy / max_ent
    return torch.relu(min_entropy - entropy_ratio) * 5.0


def ensure_special_tokens(model, tokenizer, tokens: List[str]) -> Dict[str, int]:
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [token for token in tokens if token not in existing_vocab]
    old_size = len(tokenizer)
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
    if tokens_to_add:
        input_emb = model.get_input_embeddings().weight
        with torch.no_grad():
            mean_in = input_emb[:old_size].mean(dim=0, keepdim=True)
            input_emb[old_size : len(tokenizer)] = mean_in.to(device=input_emb.device, dtype=input_emb.dtype)
            output_emb = model.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, "weight") and output_emb.weight.shape[0] >= len(tokenizer):
                mean_out = output_emb.weight[:old_size].mean(dim=0, keepdim=True)
                output_emb.weight[old_size : len(tokenizer)] = mean_out.to(
                    device=output_emb.weight.device, dtype=output_emb.weight.dtype
                )
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}
