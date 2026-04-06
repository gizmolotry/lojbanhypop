from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return ckpt


def _extract_codebook(ckpt: Dict, path: Path) -> torch.Tensor:
    state = ckpt.get("codebook_state")
    if not isinstance(state, dict) or "emb" not in state:
        raise KeyError(f"Missing codebook_state.emb in {path}")
    emb = state["emb"]
    if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
        raise ValueError(f"Invalid codebook tensor in {path}: expected rank-2 tensor.")
    return emb.float().cpu()


def _extract_usage_proxy(ckpt: Dict, codebook_size: int) -> torch.Tensor:
    state = ckpt.get("arity_head_state", {})
    if not isinstance(state, dict):
        return torch.zeros(codebook_size, dtype=torch.float32)

    proxy = torch.zeros(codebook_size, dtype=torch.float32)

    rel_bias = state.get("head_rel.bias")
    if isinstance(rel_bias, torch.Tensor) and rel_bias.numel() > 0:
        rel_bias = rel_bias.float().flatten()
        rel_p = torch.softmax(rel_bias, dim=0)
        proxy[: min(codebook_size, rel_p.numel())] += rel_p[: min(codebook_size, rel_p.numel())]

    # Older H5 checkpoints decode var slots directly in codebook space.
    for key in ("head_var1.bias", "head_var2.bias"):
        vb = state.get(key)
        if isinstance(vb, torch.Tensor) and vb.numel() > 0:
            vb = vb.float().flatten()
            if vb.numel() == codebook_size:
                proxy += torch.softmax(vb, dim=0)

    s = float(proxy.sum().item())
    if s > 0.0:
        proxy = proxy / s
    return proxy


def _topk_rows(delta_norms: torch.Tensor, top_k: int) -> List[Dict]:
    k = min(max(1, int(top_k)), int(delta_norms.numel()))
    vals, idx = torch.topk(delta_norms, k=k, largest=True, sorted=True)
    rows = []
    for i in range(k):
        rows.append({"token_id": int(idx[i].item()), "l2_delta": float(vals[i].item())})
    return rows


def _nearest_mapping(
    source: torch.Tensor,
    target: torch.Tensor,
    usage_proxy: torch.Tensor,
    top_k: int,
) -> List[Dict]:
    source_n = F.normalize(source, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    sim = source_n @ target_n.t()
    best_sim, best_id = torch.max(sim, dim=1)

    counts = torch.bincount(best_id, minlength=target.shape[0]).float()
    counts = counts / max(1.0, float(counts.sum().item()))

    strength = best_sim + usage_proxy[best_id] + counts[best_id]
    k = min(max(1, int(top_k)), int(source.shape[0]))
    _, top_ids = torch.topk(strength, k=k, largest=True, sorted=True)

    rows = []
    for src_idx in top_ids.tolist():
        tgt_idx = int(best_id[src_idx].item())
        rows.append(
            {
                "h53_token_id": int(src_idx),
                "slice1_token_id": tgt_idx,
                "cosine_similarity": float(best_sim[src_idx].item()),
                "slice1_usage_proxy": float(usage_proxy[tgt_idx].item()),
                "slice1_inbound_map_mass": float(counts[tgt_idx].item()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trace H5.3 token provenance into Slice1 codebook space and emit usage-aware mapping JSON."
    )
    p.add_argument("--h53-checkpoint", type=Path, required=True, help="Path to H5.3 checkpoint (h5_checkpoint.pt).")
    p.add_argument("--slice1-checkpoint", type=Path, required=True, help="Path to Slice1 checkpoint.")
    p.add_argument("--top-k", type=int, default=64, help="Top-k rows for provenance map and delta highlights.")
    p.add_argument("--delta-eps", type=float, default=1e-6, help="Tolerance for exact row-match ratio.")
    p.add_argument("--output", type=Path, default=Path("runs/h5_provenance_trace.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h53_ckpt = _load_checkpoint(args.h53_checkpoint)
    s1_ckpt = _load_checkpoint(args.slice1_checkpoint)

    h53_emb = _extract_codebook(h53_ckpt, args.h53_checkpoint)
    s1_emb = _extract_codebook(s1_ckpt, args.slice1_checkpoint)

    n = min(h53_emb.shape[0], s1_emb.shape[0])
    if h53_emb.shape[1] != s1_emb.shape[1]:
        raise ValueError(
            f"Hidden-size mismatch: H5.3={h53_emb.shape[1]} vs Slice1={s1_emb.shape[1]}. "
            "Cannot compute direct token provenance."
        )

    h53_common = h53_emb[:n]
    s1_common = s1_emb[:n]
    delta = h53_common - s1_common
    delta_norms = torch.norm(delta, p=2, dim=1)

    usage_proxy = _extract_usage_proxy(s1_ckpt, codebook_size=s1_emb.shape[0])

    anchor_n = min(5, n)
    anchor_delta = delta_norms[:anchor_n]
    map_rows = _nearest_mapping(h53_emb, s1_emb, usage_proxy, top_k=args.top_k)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "h53_checkpoint": str(args.h53_checkpoint),
        "slice1_checkpoint": str(args.slice1_checkpoint),
        "shapes": {
            "h53_codebook": list(h53_emb.shape),
            "slice1_codebook": list(s1_emb.shape),
            "compared_rows": int(n),
        },
        "provenance_metrics": {
            "mean_l2_delta": float(delta_norms.mean().item()),
            "max_l2_delta": float(delta_norms.max().item()),
            "min_l2_delta": float(delta_norms.min().item()),
            "exact_match_ratio_eps": float((delta_norms <= args.delta_eps).float().mean().item()),
            "anchor_mean_l2_delta": float(anchor_delta.mean().item()) if anchor_n else 0.0,
        },
        "usage_proxy_summary": {
            "nonzero_ratio": float((usage_proxy > 0).float().mean().item()),
            "top_usage_token_ids": torch.topk(usage_proxy, k=min(10, usage_proxy.numel())).indices.tolist(),
            "top_usage_values": [float(x) for x in torch.topk(usage_proxy, k=min(10, usage_proxy.numel())).values.tolist()],
        },
        "largest_delta_tokens": _topk_rows(delta_norms, top_k=args.top_k),
        "provenance_map_topk": map_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Compared rows: {n}")
    print(f"mean_l2_delta: {payload['provenance_metrics']['mean_l2_delta']:.6f}")


if __name__ == "__main__":
    main()
