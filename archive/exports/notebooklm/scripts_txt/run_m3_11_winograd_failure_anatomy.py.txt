from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import Problem, generate_dataset
from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", str(text).strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and (p == e or p.startswith(e))


def _triples(token_ids: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for i in range(0, len(token_ids) - 2, 3):
        out.append((int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2])))
    return out


def _decode_logic_tokens(arity_head: AdvisorArityHead, z_st: torch.Tensor, relation_vocab: int, var_min_id: int) -> list[torch.Tensor]:
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


def _infer_family(prompt: str) -> str:
    p = prompt.lower()
    if "too big" in p or "too small" in p or "too heavy" in p or "too weak" in p:
        return "adjective_property"
    if "because" in p or "since" in p:
        return "causal_direction"
    if "who" in p:
        return "referent_query"
    return "other"


def _gen_controlled_variants(prompt: str) -> dict[str, str]:
    out = {
        "original": prompt,
        "lexical_paraphrase": prompt.replace("because", "since").replace("suitcase", "travel case").replace("trophy", "prize"),
        "causal_connective_swap": prompt.replace("because", "although").replace("since", "although"),
        "size_adjective_swap": prompt.replace("too big", "too small").replace("too small", "too big"),
        "noun_pair_swap": prompt.replace("councilmen", "officials").replace("demonstrators", "protesters"),
        "abstract_variable_form": re.sub(r"\b[A-Z][a-z]+\b", "Entity", prompt),
    }
    return out


def _taxonomize_failure(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).lower()
    if row.get("correct", False):
        return "correct"
    if ("too big" in prompt or "too small" in prompt or "too heavy" in prompt or "too weak" in prompt):
        return "adjective_property_inversion"
    if ("because" in prompt or "since" in prompt) and _normalize(row.get("gold_answer", "")) != _normalize(row.get("model_answer", "")):
        return "causal_direction_failure"
    if float(row.get("active_token_count", 0)) < 16 or float(row.get("operator_entropy", 0.0)) < 0.2:
        return "advisor_under_articulation"
    if float(row.get("active_token_count", 0)) > 35 and not row.get("correct", False):
        return "bridge_underuse"
    return "semantic_plausibility_failure"


def _collect_bucket(bucket: str, n: int, seed: int) -> list[Problem]:
    b = str(bucket).lower()
    if b == "legacy":
        ds = generate_dataset(size=max(4 * n, n), seed=seed, profile="legacy", difficulty_tier="all")
        out = [p for p in ds if "TASK_WINOGRAD" in p.trace[0]]
        return out[:n]
    # Force diverse Winograd coverage in non-legacy buckets.
    return generate_dataset(size=n, seed=seed, profile="winograd_bench_v1", difficulty_tier=b)


def _dataset_profile_for_bucket(bucket: str) -> str:
    return "legacy" if str(bucket).lower() == "legacy" else "winograd_bench_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.11 Winograd Failure Anatomy (analysis-first).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--dataset-size-per-bucket", type=int, default=60)
    p.add_argument("--buckets", type=str, default="legacy,easy,medium,hard")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=16)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_11_winograd_failure_anatomy"))
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    assert_output_path_allowed("M", args.output_root)

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

    rows: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    buckets = [b.strip().lower() for b in str(args.buckets).split(",") if b.strip()]
    for bi, bucket in enumerate(buckets):
        ds = _collect_bucket(bucket, n=int(args.dataset_size_per_bucket), seed=int(args.seed) + bi)
        for item in ds:
            with torch.no_grad():
                h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, max_logic_new_tokens=int(args.max_logic_new_tokens))
                z_st, _idx, _cb, _commit = codebook.quantize(h_t)
                tokens = _decode_logic_tokens(arity_head, z_st, relation_vocab=int(args.relation_vocab), var_min_id=int(args.var_min_id))
                advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
                advisor_ids = torch.stack(tokens, dim=1)
                prefix = build_final_prefix(item.prompt)
                cur_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
                cur_emb = model.get_input_embeddings()(cur_ids)
                generated: list[int] = []
                pointer = 0
                for _ in range(int(args.max_final_new_tokens)):
                    p_ids = torch.full((1, cur_emb.shape[1]), pointer, device=model.device, dtype=torch.long)
                    with persistent_advisor_hook(model, int(args.layer_index), adapter_mod, advisor_states, advisor_ids, p_ids, 1.0):
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

            model_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            token_ids = [int(t[0].detach().item()) for t in tokens]
            rel_ids = token_ids[0::3]
            rel_counts = Counter(rel_ids)
            rel_total = max(1, len(rel_ids))
            probs = [float(c) / float(rel_total) for c in rel_counts.values()]
            op_entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in probs)) if probs else 0.0
            op_top1 = float(max(probs)) if probs else 1.0
            scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
            correct = _answer_match(item.answer, model_answer)
            articulated = bool(len(set(token_ids)) >= 20 and op_entropy >= 0.35)
            row = {
                "bucket": bucket,
                "dataset_profile": _dataset_profile_for_bucket(bucket),
                "problem_id": int(item.problem_id),
                "prompt": item.prompt,
                "family": _infer_family(item.prompt),
                "gold_answer": item.answer,
                "model_answer": model_answer,
                "correct": bool(correct),
                "mean_ce": float(-next_logprob if generated else 0.0),
                "mean_answer_logprob": float(next_logprob if generated else 0.0),
                "active_token_count": int(len(set(token_ids))),
                "operator_entropy": float(op_entropy),
                "operator_top1_share": float(op_top1),
                "scope": float(scope.get("scope_total", 1.0)),
                "scope_unbound": float(scope.get("scope_unbound", 1.0)),
                "advisor_trace_state": "articulated" if articulated else "collapsed",
            }
            row["failure_taxonomy"] = _taxonomize_failure(row)
            rows.append(row)
            variants.append(
                {
                    "bucket": bucket,
                    "problem_id": int(item.problem_id),
                    "family": row["family"],
                    "variants": _gen_controlled_variants(item.prompt),
                }
            )

    by_bucket: dict[str, Any] = {}
    for b in buckets:
        r = [x for x in rows if x["bucket"] == b]
        n = max(1, len(r))
        ok = [x for x in r if x["correct"]]
        bad = [x for x in r if not x["correct"]]
        by_bucket[b] = {
            "n_examples": len(r),
            "accuracy": float(len(ok)) / float(n),
            "mean_active_token_count": float(sum(float(x["active_token_count"]) for x in r) / n),
            "mean_operator_entropy": float(sum(float(x["operator_entropy"]) for x in r) / n),
            "mean_scope": float(sum(float(x["scope"]) for x in r) / n),
            "trace_articulated_rate": float(sum(1 for x in r if x["advisor_trace_state"] == "articulated")) / float(n),
            "failure_taxonomy_counts": dict(Counter(str(x["failure_taxonomy"]) for x in bad)),
            "correct_structural_means": {
                "active_token_count": float(sum(float(x["active_token_count"]) for x in ok) / max(1, len(ok))),
                "operator_entropy": float(sum(float(x["operator_entropy"]) for x in ok) / max(1, len(ok))),
                "scope": float(sum(float(x["scope"]) for x in ok) / max(1, len(ok))),
            },
            "incorrect_structural_means": {
                "active_token_count": float(sum(float(x["active_token_count"]) for x in bad) / max(1, len(bad))),
                "operator_entropy": float(sum(float(x["operator_entropy"]) for x in bad) / max(1, len(bad))),
                "scope": float(sum(float(x["scope"]) for x in bad) / max(1, len(bad))),
            },
            "examples": {
                "correct": ok[:4],
                "incorrect": bad[:4],
            },
        }

    overall_taxonomy = dict(Counter(str(x["failure_taxonomy"]) for x in rows if not x["correct"]))
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_11_winograd_failure_anatomy",
        "series": series_metadata("M", "M3.11", "scripts/run_m3_11_winograd_failure_anatomy.py"),
        "track": "M3.11",
        "lineage": lineage_metadata("eval_only", checkpoint_in=str(args.checkpoint).replace("\\", "/"), dataset_profile="mixed_winograd_eval", difficulty_tier="mixed"),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": str(args.adapter).replace("\\", "/"),
            "checkpoint": str(args.checkpoint).replace("\\", "/"),
            "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
            "baseline_id": str(baseline_manifest.get("baseline_id", "")),
            "dataset_size_per_bucket": int(args.dataset_size_per_bucket),
            "buckets": buckets,
            "dataset_profiles": {
                "legacy": "legacy",
                "easy": "winograd_bench_v1",
                "medium": "winograd_bench_v1",
                "hard": "winograd_bench_v1",
            },
        },
        "comparison_contract": {
            "mixed_dataset_families": True,
            "cross_bucket_comparable": False,
            "reason": "legacy bucket is drawn from legacy profile while easy/medium/hard use winograd_bench_v1.",
        },
        "bucket_metrics": by_bucket,
        "overall_failure_taxonomy": overall_taxonomy,
    }
    (out_dir / "m3_11_winograd_failure_anatomy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "m3_11_winograd_slice.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (out_dir / "m3_11_winograd_controlled_variants.json").write_text(json.dumps({"items": variants}, indent=2), encoding="utf-8")

    md = [
        "# M3.11 Winograd Failure Anatomy",
        "",
        "## Bucket Summary",
        "",
        "| bucket | n | accuracy | active_tokens | op_entropy | scope | articulated_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in buckets:
        m = by_bucket[b]
        md.append(
            f"| `{b}` | {m['n_examples']} | {m['accuracy']:.4f} | {m['mean_active_token_count']:.2f} | "
            f"{m['mean_operator_entropy']:.4f} | {m['mean_scope']:.4f} | {m['trace_articulated_rate']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Comparison Contract",
            "",
            "- cross_bucket_comparable: `False`",
            "- reason: `legacy uses legacy profile; easy/medium/hard use winograd_bench_v1.`",
        ]
    )
    md.extend(["", "## Failure Taxonomy (Overall)", ""])
    for k, v in sorted(overall_taxonomy.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {v}")
    (out_dir / "m3_11_winograd_failure_anatomy_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_dir / 'm3_11_winograd_failure_anatomy_report.json'}")
    print(f"Wrote: {out_dir / 'm3_11_winograd_failure_anatomy_report.md'}")
    print(f"Wrote: {out_dir / 'm3_11_winograd_slice.jsonl'}")
    print(f"Wrote: {out_dir / 'm3_11_winograd_controlled_variants.json'}")


if __name__ == "__main__":
    main()
