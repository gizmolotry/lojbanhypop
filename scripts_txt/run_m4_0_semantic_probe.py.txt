from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_15d_answer_path_forcing import (  # type: ignore
    _answer_match,
    _build_winograd_split_packs,
    _score_candidate_first_token_with_adapter_tensor,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    extract_trace_hidden_states,
)


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _decode_rel_ids(model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, prompt: str, relation_vocab: int, max_logic_new_tokens: int) -> tuple[list[int], list[int]]:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(max_logic_new_tokens)).to(model.dtype)
    z_st, _idx, _cb, _commit = codebook.quantize(h_t)
    token_ids: list[int] = []
    rel_ids: list[int] = []
    for i in range(z_st.shape[1]):
        z = z_st[:, i, :]
        rel = int(torch.argmax(arity_head.head_rel(z)[:, : int(relation_vocab)], dim=-1)[0].item())
        v1 = int(torch.argmax(arity_head.head_var1(z), dim=-1)[0].item())
        v2 = int(torch.argmax(arity_head.head_var2(z), dim=-1)[0].item())
        token_ids.extend([rel, v1, v2])
        rel_ids.append(rel)
    return token_ids, rel_ids


def _probe_features(rel_ids: list[int], relation_vocab: int, max_slots: int) -> tuple[list[float], list[float]]:
    bag = [0.0 for _ in range(int(relation_vocab))]
    for rid in rel_ids:
        if 0 <= int(rid) < int(relation_vocab):
            bag[int(rid)] += 1.0
    total = max(1.0, sum(bag))
    bag = [x / total for x in bag]

    pos = [0.0 for _ in range(int(relation_vocab) * int(max_slots))]
    for i, rid in enumerate(rel_ids[: int(max_slots)]):
        if 0 <= int(rid) < int(relation_vocab):
            pos[i * int(relation_vocab) + int(rid)] = 1.0
    return bag, pos


def _fit_probe(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, seed: int) -> tuple[LinearProbe, dict[str, float]]:
    torch.manual_seed(int(seed))
    probe = LinearProbe(int(train_x.shape[-1])).to(train_x.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=0.05, weight_decay=0.01)
    best_state = None
    best_val = -1.0
    for _ in range(200):
        opt.zero_grad()
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_logits = probe(val_x)
            val_acc = float((torch.argmax(val_logits, dim=-1) == val_y).float().mean().item())
            if val_acc >= best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    if best_state is not None:
        probe.load_state_dict(best_state)
    with torch.no_grad():
        train_acc = float((torch.argmax(probe(train_x), dim=-1) == train_y).float().mean().item())
        val_acc = float((torch.argmax(probe(val_x), dim=-1) == val_y).float().mean().item())
    return probe, {"train_accuracy": train_acc, "val_accuracy": val_acc}


def _eval_probe(probe: LinearProbe, x: torch.Tensor, y: torch.Tensor, rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    with torch.no_grad():
        logits = probe(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out = dict(row)
        out["probe_pred"] = int(pred[i].item())
        out["probe_correct"] = bool(int(pred[i].item()) == int(y[i].item()))
        out["probe_p0"] = float(probs[i, 0].item())
        out["probe_p1"] = float(probs[i, 1].item())
        out_rows.append(out)
    acc = float((pred == y).float().mean().item())
    return {"accuracy": acc}, out_rows


def _build_feature_rows(pack: list[dict[str, Any]], cache: dict[str, dict[str, Any]], relation_vocab: int, max_slots: int) -> tuple[list[list[float]], list[list[float]], list[int], list[dict[str, Any]]]:
    bag_x: list[list[float]] = []
    pos_x: list[list[float]] = []
    y: list[int] = []
    rows: list[dict[str, Any]] = []
    for item in pack:
        prompt = str(item["prompt"])
        cached = cache[prompt]
        bag_x.append(list(cached["bag_feat"]))
        pos_x.append(list(cached["pos_feat"]))
        yi = int(item["gold_index"])
        y.append(yi)
        rows.append(
            {
                "item_id": item.get("item_id", ""),
                "pair_id": item.get("pair_id", ""),
                "family": item.get("family", ""),
                "polarity": item.get("polarity", ""),
                "prompt": prompt,
                "candidates": list(item["candidates"]),
                "gold_index": yi,
                "gold_answer": str(item["candidates"][yi]),
                "relation_ids": list(cached["rel_ids"]),
                "token_ids": list(cached["token_ids"]),
            }
        )
    return bag_x, pos_x, y, rows


def _states_from_token_ids(token_ids: list[int], arity_head: AdvisorArityHead, codebook: BooleanAnchorTable) -> tuple[torch.Tensor, torch.Tensor]:
    toks = [torch.tensor([int(t)], device=codebook.emb.device, dtype=torch.long) for t in token_ids]
    states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in toks], dim=1)
    ids = torch.tensor(token_ids, device=codebook.emb.device, dtype=torch.long).unsqueeze(0)
    return states, ids


def _dominant_ops(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {0: Counter(), 1: Counter()}
    for row in train_rows:
        counts[int(row["gold_index"])] += Counter(int(x) for x in row["relation_ids"])
    op0, c0 = max(counts[0].items(), key=lambda kv: kv[1]) if counts[0] else (-1, 0)
    op1, c1 = max(counts[1].items(), key=lambda kv: kv[1]) if counts[1] else (-1, 0)
    total0 = max(1, sum(counts[0].values()))
    total1 = max(1, sum(counts[1].values()))
    return {
        "label0_op": int(op0),
        "label1_op": int(op1),
        "label0_purity": float(c0) / float(total0),
        "label1_purity": float(c1) / float(total1),
        "distinct": int(op0) != int(op1),
    }


def _run_causal_scrub(eval_pack: list[dict[str, Any]], cache: dict[str, dict[str, Any]], model, tokenizer, base_adapter: CouncilCrossAttentionAdapter, arity_head: AdvisorArityHead, codebook: BooleanAnchorTable, layer_index: int, op0: int, op1: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    flips = 0
    causal_hits = 0
    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        cached = cache[prompt]
        base_states, base_ids = _states_from_token_ids(list(cached["token_ids"]), arity_head, codebook)
        patched_token_ids = list(cached["token_ids"])
        for i in range(0, len(patched_token_ids), 3):
            if patched_token_ids[i] == int(op0):
                patched_token_ids[i] = int(op1)
            elif patched_token_ids[i] == int(op1):
                patched_token_ids[i] = int(op0)
        patched_states, patched_ids = _states_from_token_ids(patched_token_ids, arity_head, codebook)
        with torch.no_grad():
            base_gold = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, gold, base_adapter, base_states, base_ids, int(layer_index))
            base_foil = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, foil, base_adapter, base_states, base_ids, int(layer_index))
            patched_gold = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, gold, base_adapter, patched_states, patched_ids, int(layer_index))
            patched_foil = _score_candidate_first_token_with_adapter_tensor(model, tokenizer, prompt, foil, base_adapter, patched_states, patched_ids, int(layer_index))
        base_pred = gi if float(base_gold.item()) >= float(base_foil.item()) else fi
        patched_pred = gi if float(patched_gold.item()) >= float(patched_foil.item()) else fi
        flipped = int(base_pred != patched_pred)
        if flipped:
            flips += 1
        if patched_pred == fi:
            causal_hits += 1
        rows.append(
            {
                "prompt": prompt,
                "pair_id": item.get("pair_id", ""),
                "gold_index": gi,
                "base_pred": int(base_pred),
                "patched_pred": int(patched_pred),
                "flipped": bool(flipped),
                "base_delta": float((base_gold - base_foil).item()),
                "patched_delta": float((patched_gold - patched_foil).item()),
            }
        )
    n = max(1, len(rows))
    return {
        "executed": True,
        "swap_ops": [int(op0), int(op1)],
        "answer_flip_rate": float(flips) / float(n),
        "causal_alignment_rate": float(causal_hits) / float(n),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4.0 semantic probe on frozen System 1, with conditional M4.1 causal scrubbing.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--dataset-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--max-slots", type=int, default=8)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--causal-threshold", type=float, default=0.90)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m4_0_semantic_probe"))
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    assert_output_path_allowed("M", args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    target_device = torch.device("cuda" if torch.cuda.is_available() else next(model.parameters()).device)
    model = model.to(target_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.checkpoint, map_location=target_device)
    hidden = int(model.config.hidden_size)
    module_dtype = next(model.parameters()).dtype
    codebook = BooleanAnchorTable(2000, hidden).to(target_device, dtype=module_dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    codebook.eval()
    arity_head = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    arity_head.eval()
    base_adapter = CouncilCrossAttentionAdapter(hidden, use_boolean_surgery=True).to(target_device, dtype=module_dtype)
    base_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    base_adapter.eval()
    for module in (codebook, arity_head, base_adapter):
        for p in module.parameters():
            p.requires_grad = False

    train_pack, val_pack, eval_pack, split_meta = _build_winograd_split_packs(size=int(args.dataset_size), seed=int(args.seed), strict_balance=True)
    _t2, _v2, eval_seed2, split_meta_seed2 = _build_winograd_split_packs(size=int(args.dataset_size), seed=int(args.seed) + 1, strict_balance=True)

    all_prompts = {str(r["prompt"]) for r in train_pack + val_pack + eval_pack + eval_seed2}
    cache: dict[str, dict[str, Any]] = {}
    for prompt in sorted(all_prompts):
        token_ids, rel_ids = _decode_rel_ids(model, tokenizer, codebook, arity_head, prompt, int(args.relation_vocab), int(args.max_logic_new_tokens))
        bag_feat, pos_feat = _probe_features(rel_ids, int(args.relation_vocab), int(args.max_slots))
        cache[prompt] = {
            "token_ids": token_ids,
            "rel_ids": rel_ids,
            "bag_feat": bag_feat,
            "pos_feat": pos_feat,
        }

    train_bag, train_pos, train_y, train_rows = _build_feature_rows(train_pack, cache, int(args.relation_vocab), int(args.max_slots))
    val_bag, val_pos, val_y, val_rows = _build_feature_rows(val_pack, cache, int(args.relation_vocab), int(args.max_slots))
    eval_bag, eval_pos, eval_y, eval_rows = _build_feature_rows(eval_pack, cache, int(args.relation_vocab), int(args.max_slots))
    seed2_bag, seed2_pos, seed2_y, seed2_rows = _build_feature_rows(eval_seed2, cache, int(args.relation_vocab), int(args.max_slots))

    dev = target_device
    train_y_t = torch.tensor(train_y, device=dev, dtype=torch.long)
    val_y_t = torch.tensor(val_y, device=dev, dtype=torch.long)
    eval_y_t = torch.tensor(eval_y, device=dev, dtype=torch.long)
    seed2_y_t = torch.tensor(seed2_y, device=dev, dtype=torch.long)

    bag_probe, bag_fit = _fit_probe(torch.tensor(train_bag, device=dev, dtype=torch.float32), train_y_t, torch.tensor(val_bag, device=dev, dtype=torch.float32), val_y_t, int(args.seed))
    pos_probe, pos_fit = _fit_probe(torch.tensor(train_pos, device=dev, dtype=torch.float32), train_y_t, torch.tensor(val_pos, device=dev, dtype=torch.float32), val_y_t, int(args.seed) + 1)

    bag_eval_metrics, bag_eval_rows = _eval_probe(bag_probe, torch.tensor(eval_bag, device=dev, dtype=torch.float32), eval_y_t, eval_rows)
    pos_eval_metrics, pos_eval_rows = _eval_probe(pos_probe, torch.tensor(eval_pos, device=dev, dtype=torch.float32), eval_y_t, eval_rows)
    bag_seed2_metrics, _ = _eval_probe(bag_probe, torch.tensor(seed2_bag, device=dev, dtype=torch.float32), seed2_y_t, seed2_rows)
    pos_seed2_metrics, _ = _eval_probe(pos_probe, torch.tensor(seed2_pos, device=dev, dtype=torch.float32), seed2_y_t, seed2_rows)

    dominant = _dominant_ops(train_rows)
    causal_scrub: dict[str, Any]
    if max(float(bag_eval_metrics["accuracy"]), float(pos_eval_metrics["accuracy"])) >= float(args.causal_threshold) and bool(dominant["distinct"]):
        causal_scrub = _run_causal_scrub(eval_pack, cache, model, tokenizer, base_adapter, arity_head, codebook, int(args.layer_index), int(dominant["label0_op"]), int(dominant["label1_op"]))
    else:
        causal_scrub = {
            "executed": False,
            "reason": "probe_below_threshold_or_non_distinct_ops",
            "dominant_ops": dominant,
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m4_0_semantic_probe",
        "series": series_metadata("M", "M4.0", "scripts/run_m4_0_semantic_probe.py"),
        "lineage": lineage_metadata("eval_only", checkpoint_in=str(args.checkpoint).replace("\\", "/"), dataset_profile="winograd_family_split", difficulty_tier="mixed"),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {k: str(v) for k, v in vars(args).items()},
        "data_split": {**split_meta, "seed2_eval_meta": split_meta_seed2},
        "probe": {
            "bag": {**bag_fit, "eval_accuracy": float(bag_eval_metrics["accuracy"]), "seed2_accuracy": float(bag_seed2_metrics["accuracy"]), "input_dim": int(args.relation_vocab)},
            "positional": {**pos_fit, "eval_accuracy": float(pos_eval_metrics["accuracy"]), "seed2_accuracy": float(pos_seed2_metrics["accuracy"]), "input_dim": int(args.relation_vocab) * int(args.max_slots)},
            "dominant_ops": dominant,
        },
        "m4_1_causal_scrub": causal_scrub,
        "m4_2_recommendation": "implement_predicate_grounding" if max(float(bag_eval_metrics["accuracy"]), float(pos_eval_metrics["accuracy"])) < float(args.causal_threshold) else "bridge_is_reader_bottleneck",
    }

    (out_dir / "m4_0_semantic_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "m4_0_semantic_probe_rows_bag.json").write_text(json.dumps(bag_eval_rows, indent=2), encoding="utf-8")
    (out_dir / "m4_0_semantic_probe_rows_positional.json").write_text(json.dumps(pos_eval_rows, indent=2), encoding="utf-8")
    if bool(causal_scrub.get("executed")):
        (out_dir / "m4_1_causal_scrub_report.json").write_text(json.dumps(causal_scrub, indent=2), encoding="utf-8")

    md = [
        "# M4.0 Semantic Probe",
        "",
        f"- bag eval accuracy: `{float(report['probe']['bag']['eval_accuracy']):.4f}`",
        f"- positional eval accuracy: `{float(report['probe']['positional']['eval_accuracy']):.4f}`",
        f"- bag seed2 accuracy: `{float(report['probe']['bag']['seed2_accuracy']):.4f}`",
        f"- positional seed2 accuracy: `{float(report['probe']['positional']['seed2_accuracy']):.4f}`",
        f"- dominant ops: `{report['probe']['dominant_ops']}`",
        f"- m4_1 executed: `{bool(causal_scrub.get('executed', False))}`",
        f"- m4_2 recommendation: `{report['m4_2_recommendation']}`",
    ]
    if bool(causal_scrub.get("executed")):
        md.extend([
            "",
            "## M4.1 Causal Scrub",
            f"- swap_ops: `{causal_scrub['swap_ops']}`",
            f"- answer_flip_rate: `{float(causal_scrub['answer_flip_rate']):.4f}`",
            f"- causal_alignment_rate: `{float(causal_scrub['causal_alignment_rate']):.4f}`",
        ])
    else:
        md.extend(["", "## M4.1 Causal Scrub", f"- skipped: `{causal_scrub.get('reason', 'unknown')}`"])
    (out_dir / "m4_0_semantic_probe_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_dir / 'm4_0_semantic_probe_report.json'}")


if __name__ == "__main__":
    main()
