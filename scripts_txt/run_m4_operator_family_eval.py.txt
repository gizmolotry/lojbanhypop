from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    extract_trace_hidden_states,
)
from train_l_series_mvs import _decode_with_arity_bias  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4 operator family instrumentation eval.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--samples-per-family", type=int, default=20)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def _families() -> list[dict[str, Any]]:
    return [
        {
            "family": "equality",
            "domain": "logical",
            "symmetric": True,
            "templates": [
                "{a} is the same as {b}.",
                "{a} equals {b}.",
                "{a} and {b} are identical.",
            ],
            "lex_label": "equal",
        },
        {
            "family": "comparison",
            "domain": "ordering",
            "symmetric": False,
            "templates": [
                "{a} is greater than {b}.",
                "{a} outranks {b}.",
                "{a} is above {b} in order.",
            ],
            "lex_label": "compare",
        },
        {
            "family": "containment",
            "domain": "spatial",
            "symmetric": False,
            "templates": [
                "{a} is inside {b}.",
                "{b} contains {a}.",
                "{a} sits within {b}.",
            ],
            "lex_label": "contain",
        },
        {
            "family": "causal",
            "domain": "event",
            "symmetric": False,
            "templates": [
                "{a} causes {b}.",
                "{b} happens because of {a}.",
                "{a} leads to {b}.",
            ],
            "lex_label": "cause",
        },
        {
            "family": "transfer",
            "domain": "social",
            "symmetric": False,
            "templates": [
                "{a} gives the item to {b}.",
                "{a} transfers the object to {b}.",
                "{a} hands the package to {b}.",
            ],
            "lex_label": "transfer",
        },
        {
            "family": "connectivity",
            "domain": "graph",
            "symmetric": True,
            "templates": [
                "{a} is connected to {b}.",
                "{a} links with {b}.",
                "{a} and {b} are connected.",
            ],
            "lex_label": "connect",
        },
    ]


def _swap_entities(text: str, a: str, b: str) -> str:
    x = text.replace(a, "__A__").replace(b, "__B__")
    x = x.replace("__A__", b).replace("__B__", a)
    return x


def _predict_relation_ids(
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    prompt: str,
    max_logic_new_tokens: int,
) -> list[int]:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(max_logic_new_tokens)).to(model.dtype)
    z_st, _idx, _cb, _cm = codebook.quantize(h_t)
    tokens, _logits, _proxy = _decode_with_arity_bias(
        arity_head=arity_head,
        latent=z_st,
        relation_vocab=5,
        var_min_id=5,
        mask_bias=0.0,
    )
    flat = [int(t[0].detach().item()) for t in tokens]
    rels = [x for i, x in enumerate(flat) if i % 3 == 0]
    return rels


def _mode(xs: list[int]) -> int:
    if not xs:
        return -1
    c = Counter(xs)
    return int(max(c.items(), key=lambda kv: kv[1])[0])


def _cluster_ops(behavior: dict[int, list[float]], cos_threshold: float = 0.90) -> list[list[int]]:
    ops = sorted(behavior.keys())
    clusters: list[list[int]] = []
    used: set[int] = set()
    for op in ops:
        if op in used:
            continue
        base = behavior[op]
        norm_b = math.sqrt(sum(v * v for v in base)) or 1.0
        cur = [op]
        used.add(op)
        for op2 in ops:
            if op2 in used:
                continue
            vec = behavior[op2]
            norm_v = math.sqrt(sum(v * v for v in vec)) or 1.0
            dot = sum(a * b for a, b in zip(base, vec))
            cos = dot / (norm_b * norm_v)
            if cos >= cos_threshold:
                cur.append(op2)
                used.add(op2)
        clusters.append(cur)
    return clusters


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        device_map="auto",
    )
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    arity_head = AdvisorArityHead(hidden_size, 2000).to(model.device, dtype=model.dtype)
    ckpt = torch.load(args.checkpoint, map_location=model.device)
    codebook.load_state_dict(ckpt["codebook_state"])
    cs, ps = arity_head.state_dict(), ckpt["arity_head_state"]
    for n, p in ps.items():
        if n in cs and cs[n].shape == p.shape:
            cs[n].copy_(p)
    arity_head.load_state_dict(cs)

    entities = ["Alice", "Bob", "Carol", "Dave", "Eve", "Mallory"]

    rows: list[dict[str, Any]] = []
    family_to_ops: dict[str, list[int]] = defaultdict(list)
    domain_to_ops: dict[str, Counter[int]] = defaultdict(Counter)
    op_to_lex_labels: dict[int, list[str]] = defaultdict(list)
    op_swap_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"inv_total": 0, "inv_match": 0, "foil_total": 0, "foil_change": 0})

    for fam in _families():
        family = str(fam["family"])
        domain = str(fam["domain"])
        symmetric = bool(fam["symmetric"])
        templates = list(fam["templates"])
        lex_label = str(fam["lex_label"])
        for _ in range(int(args.samples_per_family)):
            a, b = rng.sample(entities, 2)
            for tmpl in templates:
                prompt = tmpl.format(a=a, b=b)
                rels = _predict_relation_ids(model, tokenizer, codebook, arity_head, prompt, args.max_logic_new_tokens)
                op = _mode(rels)
                family_to_ops[family].append(op)
                domain_to_ops[domain][op] += 1
                op_to_lex_labels[op].append(lex_label)

                swap_prompt = _swap_entities(prompt, a, b)
                rels_sw = _predict_relation_ids(model, tokenizer, codebook, arity_head, swap_prompt, args.max_logic_new_tokens)
                op_sw = _mode(rels_sw)
                s = op_swap_stats[op]
                if symmetric:
                    s["inv_total"] += 1
                    if op == op_sw:
                        s["inv_match"] += 1
                else:
                    s["foil_total"] += 1
                    if op != op_sw:
                        s["foil_change"] += 1

                rows.append(
                    {
                        "family": family,
                        "domain": domain,
                        "symmetric": symmetric,
                        "prompt": prompt,
                        "swap_prompt": swap_prompt,
                        "op": op,
                        "op_swapped": op_sw,
                        "lex_label": lex_label,
                    }
                )

    # 1) paraphrase operator consistency
    paraphrase_consistency_by_family: dict[str, float] = {}
    for fam, ops in family_to_ops.items():
        if not ops:
            paraphrase_consistency_by_family[fam] = 0.0
            continue
        c = Counter(ops)
        paraphrase_consistency_by_family[fam] = float(max(c.values())) / float(len(ops))
    paraphrase_operator_consistency = (
        sum(paraphrase_consistency_by_family.values()) / float(max(1, len(paraphrase_consistency_by_family)))
    )

    # 2) lexical leakage probe (simple train/test mapping op->majority label)
    rng.shuffle(rows)
    split = int(0.8 * len(rows))
    tr, te = rows[:split], rows[split:]
    map_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for r in tr:
        map_counts[int(r["op"])][str(r["lex_label"])] += 1
    op_to_label: dict[int, str] = {}
    for op, c in map_counts.items():
        op_to_label[op] = max(c.items(), key=lambda kv: kv[1])[0]
    correct = 0
    for r in te:
        pred = op_to_label.get(int(r["op"]), "")
        if pred == str(r["lex_label"]):
            correct += 1
    operator_lexical_probe_acc = float(correct) / float(max(1, len(te)))
    majority = Counter([str(x["lex_label"]) for x in te])
    majority_label = max(majority.items(), key=lambda kv: kv[1])[0] if majority else ""
    majority_baseline_acc = float(majority.get(majority_label, 0)) / float(max(1, len(te)))

    # 3) per-op swap invariant rate
    per_op_swap_invariant_rate: dict[str, float] = {}
    per_op_swap_foil_change_rate: dict[str, float] = {}
    for op, s in op_swap_stats.items():
        inv = float(s["inv_match"]) / float(max(1, s["inv_total"]))
        foil = float(s["foil_change"]) / float(max(1, s["foil_total"]))
        per_op_swap_invariant_rate[str(op)] = inv
        per_op_swap_foil_change_rate[str(op)] = foil

    # domain matrix
    ops_sorted = sorted({int(x["op"]) for x in rows})
    op_counts = Counter(int(x["op"]) for x in rows)
    total_ops = int(sum(op_counts.values()))
    active_op_count = int(sum(1 for _op, c in op_counts.items() if c >= 5))
    top1_op_share = float(max(op_counts.values())) / float(max(1, total_ops))
    operator_domain_matrix: dict[str, dict[str, float]] = {}
    for domain, c in domain_to_ops.items():
        tot = float(sum(c.values()))
        operator_domain_matrix[domain] = {str(op): float(c.get(op, 0)) / float(max(1.0, tot)) for op in ops_sorted}

    # cluster report from behavior vectors
    behavior_vectors: dict[int, list[float]] = {}
    for op in ops_sorted:
        vec: list[float] = []
        vec.append(float(per_op_swap_invariant_rate.get(str(op), 0.0)))
        vec.append(float(per_op_swap_foil_change_rate.get(str(op), 0.0)))
        for domain in sorted(operator_domain_matrix.keys()):
            vec.append(float(operator_domain_matrix[domain].get(str(op), 0.0)))
        behavior_vectors[int(op)] = vec
    clusters = _cluster_ops(behavior_vectors, cos_threshold=0.90)

    # Predicate Family Score (dashboard scalar)
    avg_inv = sum(per_op_swap_invariant_rate.values()) / float(max(1, len(per_op_swap_invariant_rate)))
    avg_foil = sum(per_op_swap_foil_change_rate.values()) / float(max(1, len(per_op_swap_foil_change_rate)))
    predicate_family_score = (
        0.30 * paraphrase_operator_consistency
        + 0.25 * (1.0 - operator_lexical_probe_acc)
        + 0.20 * avg_inv
        + 0.15 * avg_foil
        + 0.10 * (1.0 / float(max(1, len(clusters))))
    )

    generated = datetime.now(timezone.utc).isoformat()
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    operator_family_report = {
        "generated_utc": generated,
        "report_type": "operator_family_report",
        "active_op_count": active_op_count,
        "top1_op_share": float(top1_op_share),
        "paraphrase_operator_consistency": float(paraphrase_operator_consistency),
        "paraphrase_consistency_by_family": paraphrase_consistency_by_family,
        "per_op_swap_invariant_rate": per_op_swap_invariant_rate,
        "per_op_swap_foil_change_rate": per_op_swap_foil_change_rate,
        "predicate_family_score": float(predicate_family_score),
    }
    (out_dir / "operator_family_report.json").write_text(json.dumps(operator_family_report, indent=2), encoding="utf-8")

    lexical_report = {
        "generated_utc": generated,
        "report_type": "lexical_leakage_probe",
        "operator_lexical_probe_acc": float(operator_lexical_probe_acc),
        "majority_baseline_acc": float(majority_baseline_acc),
        "test_size": int(len(te)),
        "op_to_majority_label_train": {str(k): v for k, v in op_to_label.items()},
    }
    (out_dir / "lexical_leakage_probe.json").write_text(json.dumps(lexical_report, indent=2), encoding="utf-8")

    domain_report = {
        "generated_utc": generated,
        "report_type": "operator_domain_matrix",
        "operators": [int(x) for x in ops_sorted],
        "matrix": operator_domain_matrix,
    }
    (out_dir / "operator_domain_matrix.json").write_text(json.dumps(domain_report, indent=2), encoding="utf-8")

    cluster_report = {
        "generated_utc": generated,
        "report_type": "operator_cluster_report",
        "clusters": clusters,
        "behavior_vectors": {str(k): v for k, v in behavior_vectors.items()},
    }
    (out_dir / "operator_cluster_report.json").write_text(json.dumps(cluster_report, indent=2), encoding="utf-8")

    print(f"Wrote: {out_dir / 'operator_family_report.json'}")
    print(f"Wrote: {out_dir / 'operator_domain_matrix.json'}")
    print(f"Wrote: {out_dir / 'lexical_leakage_probe.json'}")
    print(f"Wrote: {out_dir / 'operator_cluster_report.json'}")


if __name__ == "__main__":
    main()
