from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import Problem, generate_dataset
from lojban_evolution.l_series import (
    build_scope_tokens_from_triples,
    compute_identity_violation_from_ce,
    compute_scope_violation_components,
)
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", str(text).strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and (p == e or p.startswith(e))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _triples(token_ids: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for i in range(0, len(token_ids) - 2, 3):
        out.append((int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2])))
    return out


def _decode_logic_tokens(
    arity_head: AdvisorArityHead,
    z_st: torch.Tensor,
    relation_vocab: int,
    var_min_id: int,
) -> list[torch.Tensor]:
    tokens: list[torch.Tensor] = []
    for i in range(z_st.shape[1]):
        z = z_st[:, i, :]
        l_rel = arity_head.head_rel(z)
        mask_rel = torch.full_like(l_rel, -1e9)
        mask_rel[:, : int(relation_vocab)] = 0.0
        t_rel = torch.argmax(l_rel + mask_rel, dim=-1)

        l_v1 = arity_head.head_var1(z)
        mask_v = torch.full_like(l_v1, -1e9)
        mask_v[:, int(var_min_id) :] = 0.0
        t_v1 = torch.argmax(l_v1 + mask_v, dim=-1)

        l_v2 = arity_head.head_var2(z)
        t_v2 = torch.argmax(l_v2 + mask_v, dim=-1)

        tokens.extend([t_rel, t_v1, t_v2])
    return tokens


def _generate_answer_and_metrics(
    model,
    tokenizer,
    adapter_mod: CouncilCrossAttentionAdapter,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    prompt: str,
    layer_index: int,
    max_logic_new_tokens: int,
    max_final_new_tokens: int,
    relation_vocab: int,
    var_min_id: int,
    primitive_token_ids: set[int],
) -> dict[str, Any]:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=max_logic_new_tokens)
        z_st, _idx, _cb, _commit = codebook.quantize(h_t)
        tokens = _decode_logic_tokens(arity_head, z_st, relation_vocab=int(relation_vocab), var_min_id=int(var_min_id))
        advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
        advisor_ids = torch.stack(tokens, dim=1)

        prefix = build_final_prefix(prompt)
        cur_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        cur_emb = model.get_input_embeddings()(cur_ids)
        generated: list[int] = []
        pointer = 0
        for _ in range(int(max_final_new_tokens)):
            p_ids = torch.full((1, cur_emb.shape[1]), pointer, device=model.device, dtype=torch.long)
            with persistent_advisor_hook(model, layer_index, adapter_mod, advisor_states, advisor_ids, p_ids, 1.0):
                out = model(inputs_embeds=cur_emb, return_dict=True)
            logits = out.logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            next_logprob = float(torch.log_softmax(logits, dim=-1)[0, next_id].item())
            generated.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
            next_emb = model.get_input_embeddings()(torch.tensor([[next_id]], device=model.device))
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            pointer = min(pointer + 1, max(0, advisor_ids.shape[1] - 1))

        answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    token_ids = [int(t[0].detach().item()) for t in tokens]
    rel_ids = token_ids[0::3]
    rel_total = max(1, len(rel_ids))
    rel_counts: dict[int, int] = {}
    for r in rel_ids:
        rel_counts[r] = rel_counts.get(r, 0) + 1
    probs = [float(c) / float(rel_total) for c in rel_counts.values()]
    op_entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
    op_top1 = float(max(probs)) if probs else 1.0

    triples = _triples(token_ids)
    scope = compute_scope_violation_components(build_scope_tokens_from_triples(triples, var_prefix="VAR"))

    active_token_count = len(set(token_ids))
    primitive_usage = 0.0
    if token_ids and primitive_token_ids:
        primitive_hits = sum(1 for t in token_ids if int(t) in primitive_token_ids)
        primitive_usage = float(primitive_hits) / float(len(token_ids))

    return {
        "model_answer": answer_text,
        "mean_answer_logprob": float(next_logprob if generated else 0.0),
        "active_token_count": int(active_token_count),
        "scope_total": float(scope.get("scope_total", 1.0)),
        "scope_unbound": float(scope.get("unbound", 1.0)),
        "operator_entropy": float(op_entropy),
        "operator_top1_share": float(op_top1),
        "primitive_usage": float(primitive_usage),
    }


def _load_primitive_ids(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    out: set[int] = set()
    for row in payload.get("candidates", []):
        if isinstance(row, dict) and "token_id" in row:
            try:
                out.add(int(row["token_id"]))
            except (TypeError, ValueError):
                pass
    return out


def _validate_m_baseline_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("series_id", "")).strip().upper() != "M":
        raise ValueError("baseline_manifest.series_id must be 'M'")
    if not str(payload.get("baseline_id", "")).strip():
        raise ValueError("baseline_manifest.baseline_id is required")
    return payload


def _bucket_dataset(bucket: str, size: int, seed: int) -> list[Problem]:
    b = str(bucket).strip().lower()
    if b == "legacy":
        return generate_dataset(size=int(size), seed=int(seed), profile="legacy", difficulty_tier="all")
    if b in {"easy", "medium", "hard"}:
        return generate_dataset(size=int(size), seed=int(seed), profile="diverse_v2", difficulty_tier=b)
    raise ValueError(f"unsupported bucket: {bucket}")


def _infer_trained_buckets(checkpoint: Path, declared: str) -> list[str]:
    if declared.strip():
        return [b.strip().lower() for b in declared.split(",") if b.strip()]
    parts = {p.lower() for p in checkpoint.parts}
    return [b for b in ("legacy", "easy", "medium", "hard") if b in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.10 OOD Accuracy Probe (evaluation-only, frozen M_BASE).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--dataset-size-per-bucket", type=int, default=80)
    p.add_argument("--buckets", type=str, default="legacy,easy,medium,hard")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=16)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--primitive-candidates", type=Path, default=None)
    p.add_argument("--trained-buckets", type=str, default="", help="Comma-separated buckets used during checkpoint training.")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_10_ood_accuracy"))
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_manifest.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {args.baseline_manifest}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    assert_output_path_allowed("M", args.output_root)
    baseline_manifest = _validate_m_baseline_manifest(args.baseline_manifest)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=model.device)
    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    adapter_mod = CouncilCrossAttentionAdapter(hidden_size, use_boolean_surgery=True).to(model.device, dtype=model.dtype)
    adapter_mod.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    arity_head = AdvisorArityHead(hidden_size=hidden_size, codebook_size=2000).to(model.device, dtype=model.dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)

    primitive_ids = _load_primitive_ids(args.primitive_candidates)
    buckets = [b.strip().lower() for b in str(args.buckets).split(",") if b.strip()]
    trained_buckets = _infer_trained_buckets(args.checkpoint, str(args.trained_buckets))

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for bi, bucket in enumerate(buckets):
        ds = _bucket_dataset(bucket, size=int(args.dataset_size_per_bucket), seed=int(args.seed) + int(bi))
        for item in ds:
            r = _generate_answer_and_metrics(
                model=model,
                tokenizer=tokenizer,
                adapter_mod=adapter_mod,
                codebook=codebook,
                arity_head=arity_head,
                prompt=item.prompt,
                layer_index=int(args.layer_index),
                max_logic_new_tokens=int(args.max_logic_new_tokens),
                max_final_new_tokens=int(args.max_final_new_tokens),
                relation_vocab=int(args.relation_vocab),
                var_min_id=int(args.var_min_id),
                primitive_token_ids=primitive_ids,
            )
            ce = -float(r["mean_answer_logprob"])
            identity = float(compute_identity_violation_from_ce(torch.tensor(ce), None).item())
            ok = _answer_match(item.answer, str(r["model_answer"]))
            row = {
                "bucket": bucket,
                "problem_id": int(item.problem_id),
                "difficulty": str(item.difficulty),
                "prompt": item.prompt,
                "gold_answer": item.answer,
                "model_answer": str(r["model_answer"]),
                "correct": bool(ok),
                "mean_ce": float(ce),
                "mean_answer_logprob": float(r["mean_answer_logprob"]),
                "active_token_count": int(r["active_token_count"]),
                "scope_total": float(r["scope_total"]),
                "scope_unbound": float(r["scope_unbound"]),
                "identity": float(identity),
                "operator_entropy": float(r["operator_entropy"]),
                "operator_top1_share": float(r["operator_top1_share"]),
                "primitive_usage": float(r["primitive_usage"]),
            }
            by_bucket[bucket].append(row)
            all_rows.append(row)

    bucket_summary: dict[str, Any] = {}
    for bucket in buckets:
        rows = by_bucket.get(bucket, [])
        n = len(rows)
        n_correct = sum(1 for r in rows if bool(r["correct"]))
        ys = [1.0 if bool(r["correct"]) else 0.0 for r in rows]
        act = [float(r["active_token_count"]) for r in rows]
        scp = [float(r["scope_total"]) for r in rows]
        prv = [float(r["primitive_usage"]) for r in rows]

        bucket_summary[bucket] = {
            "distribution_relation": "in_distribution" if bucket in trained_buckets else "ood",
            "n_examples": int(n),
            "n_correct": int(n_correct),
            "accuracy": float(n_correct / n) if n else 0.0,
            "mean_ce": float(sum(float(r["mean_ce"]) for r in rows) / n) if n else 0.0,
            "mean_answer_logprob": float(sum(float(r["mean_answer_logprob"]) for r in rows) / n) if n else 0.0,
            "mean_active_token_count": float(sum(float(r["active_token_count"]) for r in rows) / n) if n else 0.0,
            "mean_scope": float(sum(float(r["scope_total"]) for r in rows) / n) if n else 0.0,
            "mean_scope_unbound": float(sum(float(r["scope_unbound"]) for r in rows) / n) if n else 0.0,
            "mean_identity": float(sum(float(r["identity"]) for r in rows) / n) if n else 0.0,
            "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n) if n else 0.0,
            "mean_top1_op_share": float(sum(float(r["operator_top1_share"]) for r in rows) / n) if n else 0.0,
            "mean_primitive_candidate_usage": float(sum(float(r["primitive_usage"]) for r in rows) / n) if n else 0.0,
            "corr_active_token_count_vs_correct": _corr(act, ys),
            "corr_scope_vs_correct": _corr(scp, ys),
            "corr_primitive_usage_vs_correct": _corr(prv, ys),
        }

    examples: dict[str, Any] = {}
    for bucket in buckets:
        rows = by_bucket.get(bucket, [])
        correct_rows = [r for r in rows if bool(r["correct"])]
        incorrect_rows = [r for r in rows if not bool(r["correct"])]
        examples[bucket] = {
            "correct_examples": correct_rows[:3],
            "incorrect_examples": incorrect_rows[:3],
        }

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_10_ood_accuracy_probe",
        "series": series_metadata("M", "M3.10", "scripts/run_m3_10_ood_accuracy_probe.py"),
        "track": "M3.10",
        "lineage": lineage_metadata("eval_only", checkpoint_in=str(args.checkpoint).replace("\\", "/"), dataset_profile="mixed_bucket_eval", difficulty_tier="mixed"),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": str(args.adapter).replace("\\", "/"),
            "checkpoint": str(args.checkpoint).replace("\\", "/"),
            "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
            "baseline_id": str(baseline_manifest.get("baseline_id", "")),
            "dataset_size_per_bucket": int(args.dataset_size_per_bucket),
            "buckets": buckets,
            "trained_buckets": trained_buckets,
        },
        "bucket_metrics": bucket_summary,
        "ood_bucket_metrics": {k: v for k, v in bucket_summary.items() if str(v.get("distribution_relation")) == "ood"},
        "in_distribution_bucket_metrics": {k: v for k, v in bucket_summary.items() if str(v.get("distribution_relation")) == "in_distribution"},
        "representative_examples": examples,
    }
    (out_dir / "m3_10_ood_accuracy_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# M3.10 OOD Accuracy Probe",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_utc: `{payload['generated_utc']}`",
        "",
        "## Headline Table",
        "",
        "| bucket | relation | n | accuracy | mean_ce | active_tokens | scope | identity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in buckets:
        m = bucket_summary.get(b, {})
        lines.append(
            f"| `{b}` | `{m.get('distribution_relation', 'undeclared')}` | {int(m.get('n_examples', 0))} | {float(m.get('accuracy', 0.0)):.4f} | "
            f"{float(m.get('mean_ce', 0.0)):.4f} | {float(m.get('mean_active_token_count', 0.0)):.4f} | "
            f"{float(m.get('mean_scope', 0.0)):.4f} | {float(m.get('mean_identity', 0.0)):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Structural-Performance Linkage",
            "",
        ]
    )
    for b in buckets:
        m = bucket_summary.get(b, {})
        lines.extend(
            [
                f"### `{b}`",
                f"- corr(active_token_count, correct): `{float(m.get('corr_active_token_count_vs_correct', 0.0)):.6f}`",
                f"- corr(scope, correct): `{float(m.get('corr_scope_vs_correct', 0.0)):.6f}`",
                f"- corr(primitive_usage, correct): `{float(m.get('corr_primitive_usage_vs_correct', 0.0)):.6f}`",
            ]
        )

    lines.extend(["", "## Representative Examples", ""])
    for b in buckets:
        lines.append(f"### `{b}`")
        lines.append("- correct:")
        for r in examples[b]["correct_examples"]:
            lines.append(
                f"  - id={r['problem_id']} gold=`{r['gold_answer']}` pred=`{r['model_answer']}` "
                f"active={r['active_token_count']} scope={r['scope_total']:.4f}"
            )
        lines.append("- incorrect:")
        for r in examples[b]["incorrect_examples"]:
            lines.append(
                f"  - id={r['problem_id']} gold=`{r['gold_answer']}` pred=`{r['model_answer']}` "
                f"active={r['active_token_count']} scope={r['scope_total']:.4f}"
            )
    (out_dir / "m3_10_ood_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "m3_10_bucket_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows), encoding="utf-8")

    print(f"Wrote: {out_dir / 'm3_10_ood_accuracy_report.json'}")
    print(f"Wrote: {out_dir / 'm3_10_ood_accuracy_report.md'}")
    print(f"Wrote: {out_dir / 'm3_10_bucket_predictions.jsonl'}")


if __name__ == "__main__":
    main()
