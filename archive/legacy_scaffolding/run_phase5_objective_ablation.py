from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from train_lora import (
    compositional_consistency_loss,
    compression_regularization_loss,
    coverage_regularization_loss,
    roundtrip_consistency_loss,
    semantic_unambiguity_loss,
    trajectory_balance_loss,
)


OBJECTIVE_TERM_TO_WEIGHT: Dict[str, float] = {
    "semantic_unambiguity_loss": 0.05,
    "compositional_consistency_loss": 0.05,
    "roundtrip_consistency_loss": 0.05,
    "coverage_regularization_loss": 0.01,
    "compression_regularization_loss": 0.01,
    "trajectory_balance_loss": 0.02,
}


def synthetic_batch(batch_size: int, seq_len: int, hidden_dim: int, vocab_size: int, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), generator=g)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    hidden = torch.randn((batch_size, seq_len, hidden_dim), generator=g)
    logits = torch.randn((batch_size, seq_len, vocab_size), generator=g)

    trace_start = torch.full((batch_size,), 4, dtype=torch.long)
    answer_start = torch.full((batch_size,), max(seq_len - 6, 5), dtype=torch.long)
    problem_id = torch.arange(batch_size) // 2
    mode_id = torch.tensor([(i % 2) for i in range(batch_size)], dtype=torch.long)

    # Force deterministic token coverage so ablation terms are not spuriously zero.
    sem_ids = [3, 7, 11, 13]
    comp_pairs = [(3, 7), (11, 13)]
    for b in range(batch_size):
        input_ids[b, 5] = sem_ids[b % len(sem_ids)]
        input_ids[b, 9] = sem_ids[(b + 1) % len(sem_ids)]
        pair = comp_pairs[b % len(comp_pairs)]
        input_ids[b, 12] = pair[0]
        input_ids[b, 14] = pair[1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "hidden": hidden,
        "logits": logits,
        "trace_start": trace_start,
        "answer_start": answer_start,
        "problem_id": problem_id,
        "mode_id": mode_id,
    }


def compute_terms(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    sem_ids = [3, 7, 11, 13]
    comp_pairs = [(3, 7), (11, 13)]
    cov_ids = [3, 5, 7, 11, 13]
    return {
        "semantic_unambiguity_loss": float(
            semantic_unambiguity_loss(
                batch["hidden"], batch["input_ids"], batch["attention_mask"], sem_ids
            ).item()
        ),
        "compositional_consistency_loss": float(
            compositional_consistency_loss(
                batch["hidden"], batch["input_ids"], batch["attention_mask"], comp_pairs
            ).item()
        ),
        "roundtrip_consistency_loss": float(
            roundtrip_consistency_loss(
                batch["hidden"],
                batch["problem_id"],
                batch["mode_id"],
                batch["trace_start"],
                batch["answer_start"],
                batch["attention_mask"],
            ).item()
        ),
        "coverage_regularization_loss": float(
            coverage_regularization_loss(
                batch["logits"],
                batch["input_ids"],
                batch["attention_mask"],
                cov_ids,
                batch["trace_start"],
                batch["answer_start"],
            ).item()
        ),
        "compression_regularization_loss": float(
            compression_regularization_loss(
                batch["hidden"], batch["attention_mask"], batch["trace_start"], batch["answer_start"]
            ).item()
        ),
        "trajectory_balance_loss": float(
            trajectory_balance_loss(
                batch["logits"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["trace_start"],
                batch["answer_start"],
                reward_beta=0.9,
                reward_floor=1e-6,
                air_gapped_oracle=True,
                uniform_backward_policy=True,
            ).item()
        ),
    }


def build_variants(base_weights: Dict[str, float]) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    variants.append({"name": "phase5_full", "weights": dict(base_weights)})
    variants.append({"name": "baseline_no_phase5", "weights": {k: 0.0 for k in base_weights}})
    for key in base_weights:
        w = dict(base_weights)
        w[key] = 0.0
        variants.append({"name": f"ablate_{key}", "weights": w})
    for key in base_weights:
        w = {k: 0.0 for k in base_weights}
        w[key] = base_weights[key]
        variants.append({"name": f"only_{key}", "weights": w})
    return variants


def score_variants(terms: Dict[str, float], variants: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], float]:
    full_total = sum(float(OBJECTIVE_TERM_TO_WEIGHT[k]) * float(terms[k]) for k in OBJECTIVE_TERM_TO_WEIGHT)
    for row in variants:
        w = row["weights"]
        total = sum(float(w[k]) * float(terms[k]) for k in OBJECTIVE_TERM_TO_WEIGHT)
        row["total_regularizer"] = float(total)
        row["delta_vs_full"] = float(total - full_total)
    return variants, float(full_total)


def write_summary_md(path: Path, terms: Dict[str, float], variants: List[Dict[str, object]], full_total: float) -> None:
    dead = [k for k, v in terms.items() if abs(float(v)) < 1e-8]
    lines: List[str] = []
    lines.append("# Phase-5 Objective Ablation Summary")
    lines.append("")
    lines.append(f"- full_regularizer: `{full_total:.6f}`")
    lines.append(f"- dead_terms: `{', '.join(dead) if dead else 'none'}`")
    lines.append("")
    lines.append("## Raw Terms")
    for k, v in terms.items():
        lines.append(f"- `{k}` = `{float(v):.6f}`")
    lines.append("")
    lines.append("## Variant Impact (sorted by delta_vs_full)")
    lines.append("| variant | total_regularizer | delta_vs_full |")
    lines.append("|---|---:|---:|")
    for row in sorted(variants, key=lambda r: float(r["delta_vs_full"])):
        lines.append(
            f"| `{row['name']}` | `{float(row['total_regularizer']):.6f}` | `{float(row['delta_vs_full']):+.6f}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate Phase-5 differentiable objective terms.")
    parser.add_argument("--output", type=Path, default=Path("runs/phase5_objective_ablation.json"))
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = synthetic_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    terms = compute_terms(batch)

    variants, full_total = score_variants(terms, build_variants(OBJECTIVE_TERM_TO_WEIGHT))

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "vocab_size": args.vocab_size,
            "seed": args.seed,
        },
        "weights": OBJECTIVE_TERM_TO_WEIGHT,
        "terms": terms,
        "dead_terms": [k for k, v in terms.items() if abs(float(v)) < 1e-8],
        "full_total_regularizer": full_total,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md = args.summary_md if args.summary_md is not None else args.output.with_suffix(".md")
    write_summary_md(summary_md, terms, variants, full_total)
    print(f"Wrote: {args.output}")
    print(f"Wrote: {summary_md}")
    for v in variants:
        print(f"{v['name']}: regularizer={v['total_regularizer']:.6f} delta_vs_full={v['delta_vs_full']:+.6f}")


if __name__ == "__main__":
    main()
