from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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
from run_m3_15d_answer_path_forcing import _build_winograd_split_packs  # type: ignore
from run_m4_0_semantic_probe import (  # type: ignore
    _build_feature_rows,
    _decode_rel_ids,
    _dominant_ops,
    _eval_probe,
    _fit_probe,
    _probe_features,
)
from train_h5_persistent_vq_advisor import AdvisorArityHead, BooleanAnchorTable, extract_trace_hidden_states  # type: ignore


def _pooled_relation_probs(arity_head: AdvisorArityHead, z_st: torch.Tensor, relation_vocab: int) -> torch.Tensor:
    probs = []
    for i in range(z_st.shape[1]):
        logits = arity_head.head_rel(z_st[:, i, :])[:, : int(relation_vocab)]
        probs.append(torch.softmax(logits, dim=-1))
    return torch.stack(probs, dim=0).mean(dim=(0, 1))


def _cache_latents(model, tokenizer, codebook: BooleanAnchorTable, prompts: list[str], max_logic_new_tokens: int) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for prompt in prompts:
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(max_logic_new_tokens)).to(model.dtype)
        z_st, _idx, _cb, _commit = codebook.quantize(h_t)
        cache[prompt] = z_st.detach()
    return cache


def _build_pair_groups(pack: list[dict[str, Any]]) -> dict[str, dict[int, list[str]]]:
    groups: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in pack:
        groups[str(item["pair_id"])][int(item["gold_index"])].append(str(item["prompt"]))
    return groups


def _train_grounding(arity_head: AdvisorArityHead, latent_cache: dict[str, torch.Tensor], groups: dict[str, dict[int, list[str]]], relation_vocab: int, train_steps: int, seed: int, lr: float, margin: float) -> dict[str, float]:
    for p in arity_head.parameters():
        p.requires_grad = False
    for p in arity_head.head_rel.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(arity_head.head_rel.parameters(), lr=float(lr), weight_decay=0.01)
    rng = random.Random(int(seed))
    same_hist, diff_hist, sharp_hist = [], [], []
    pair_ids = [pid for pid, by_label in groups.items() if by_label.get(0) and by_label.get(1)]
    for _ in range(int(train_steps)):
        pid = rng.choice(pair_ids)
        pos0 = rng.sample(groups[pid][0], 2) if len(groups[pid][0]) >= 2 else [groups[pid][0][0], groups[pid][0][0]]
        pos1 = rng.sample(groups[pid][1], 2) if len(groups[pid][1]) >= 2 else [groups[pid][1][0], groups[pid][1][0]]
        p00 = _pooled_relation_probs(arity_head, latent_cache[pos0[0]], int(relation_vocab))
        p01 = _pooled_relation_probs(arity_head, latent_cache[pos0[1]], int(relation_vocab))
        p10 = _pooled_relation_probs(arity_head, latent_cache[pos1[0]], int(relation_vocab))
        p11 = _pooled_relation_probs(arity_head, latent_cache[pos1[1]], int(relation_vocab))
        same_loss = F.mse_loss(p00, p01) + F.mse_loss(p10, p11)
        mean0 = 0.5 * (p00 + p01)
        mean1 = 0.5 * (p10 + p11)
        sep = torch.norm(mean0 - mean1, p=2)
        diff_loss = torch.relu(torch.tensor(float(margin), device=sep.device, dtype=sep.dtype) - sep)
        sharp_loss = (1.0 - torch.max(p00)) + (1.0 - torch.max(p01)) + (1.0 - torch.max(p10)) + (1.0 - torch.max(p11))
        loss = same_loss + diff_loss + 0.1 * sharp_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        same_hist.append(float(same_loss.detach().item()))
        diff_hist.append(float(diff_loss.detach().item()))
        sharp_hist.append(float(sharp_loss.detach().item()))
    return {
        "same_loss": float(sum(same_hist) / max(1, len(same_hist))),
        "diff_loss": float(sum(diff_hist) / max(1, len(diff_hist))),
        "sharp_loss": float(sum(sharp_hist) / max(1, len(sharp_hist))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4.2 predicate grounding on System 1 relation head, with post-grounding semantic probe.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--dataset-size", type=int, default=128)
    p.add_argument("--train-steps", type=int, default=160)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--max-slots", type=int, default=8)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m4_2_predicate_grounding"))
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
    arity_head_base = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head_base.load_state_dict(ckpt["arity_head_state"], strict=False)
    arity_head_base.eval()
    arity_head_grounded = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head_grounded.load_state_dict(ckpt["arity_head_state"], strict=False)

    train_pack, val_pack, eval_pack, split_meta = _build_winograd_split_packs(size=int(args.dataset_size), seed=int(args.seed), strict_balance=True)
    _t2, _v2, eval_seed2, split_meta_seed2 = _build_winograd_split_packs(size=int(args.dataset_size), seed=int(args.seed) + 1, strict_balance=True)
    prompts = sorted({str(r["prompt"]) for r in train_pack + val_pack + eval_pack + eval_seed2})
    latent_cache = _cache_latents(model, tokenizer, codebook, prompts, int(args.max_logic_new_tokens))

    train_groups = _build_pair_groups(train_pack)
    grounding_train = _train_grounding(arity_head_grounded, latent_cache, train_groups, int(args.relation_vocab), int(args.train_steps), int(args.seed), float(args.lr), float(args.margin))

    def build_cache_for_head(head: AdvisorArityHead) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        for prompt in prompts:
            token_ids, rel_ids = _decode_rel_ids(model, tokenizer, codebook, head, prompt, int(args.relation_vocab), int(args.max_logic_new_tokens))
            bag_feat, pos_feat = _probe_features(rel_ids, int(args.relation_vocab), int(args.max_slots))
            cache[prompt] = {"token_ids": token_ids, "rel_ids": rel_ids, "bag_feat": bag_feat, "pos_feat": pos_feat}
        return cache

    cache_before = build_cache_for_head(arity_head_base)
    cache_after = build_cache_for_head(arity_head_grounded)

    def run_probe(cache: dict[str, dict[str, Any]], probe_seed: int) -> dict[str, Any]:
        train_bag, train_pos, train_y, train_rows = _build_feature_rows(train_pack, cache, int(args.relation_vocab), int(args.max_slots))
        val_bag, val_pos, val_y, _ = _build_feature_rows(val_pack, cache, int(args.relation_vocab), int(args.max_slots))
        eval_bag, eval_pos, eval_y, eval_rows = _build_feature_rows(eval_pack, cache, int(args.relation_vocab), int(args.max_slots))
        seed2_bag, seed2_pos, seed2_y, seed2_rows = _build_feature_rows(eval_seed2, cache, int(args.relation_vocab), int(args.max_slots))
        dev = target_device
        train_y_t = torch.tensor(train_y, device=dev, dtype=torch.long)
        val_y_t = torch.tensor(val_y, device=dev, dtype=torch.long)
        eval_y_t = torch.tensor(eval_y, device=dev, dtype=torch.long)
        seed2_y_t = torch.tensor(seed2_y, device=dev, dtype=torch.long)
        bag_probe, bag_fit = _fit_probe(torch.tensor(train_bag, device=dev, dtype=torch.float32), train_y_t, torch.tensor(val_bag, device=dev, dtype=torch.float32), val_y_t, int(probe_seed))
        pos_probe, pos_fit = _fit_probe(torch.tensor(train_pos, device=dev, dtype=torch.float32), train_y_t, torch.tensor(val_pos, device=dev, dtype=torch.float32), val_y_t, int(probe_seed) + 1)
        bag_eval, _bag_rows = _eval_probe(bag_probe, torch.tensor(eval_bag, device=dev, dtype=torch.float32), eval_y_t, eval_rows)
        pos_eval, _pos_rows = _eval_probe(pos_probe, torch.tensor(eval_pos, device=dev, dtype=torch.float32), eval_y_t, eval_rows)
        bag_seed2, _ = _eval_probe(bag_probe, torch.tensor(seed2_bag, device=dev, dtype=torch.float32), seed2_y_t, seed2_rows)
        pos_seed2, _ = _eval_probe(pos_probe, torch.tensor(seed2_pos, device=dev, dtype=torch.float32), seed2_y_t, seed2_rows)
        return {
            "bag": {**bag_fit, "eval_accuracy": float(bag_eval["accuracy"]), "seed2_accuracy": float(bag_seed2["accuracy"])},
            "positional": {**pos_fit, "eval_accuracy": float(pos_eval["accuracy"]), "seed2_accuracy": float(pos_seed2["accuracy"])},
            "dominant_ops": _dominant_ops(train_rows),
        }

    probe_before = run_probe(cache_before, int(args.seed))
    probe_after = run_probe(cache_after, int(args.seed) + 17)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m4_2_predicate_grounding",
        "series": series_metadata("M", "M4.2", "scripts/run_m4_2_predicate_grounding.py"),
        "lineage": lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed"),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {k: str(v) for k, v in vars(args).items()},
        "data_split": {**split_meta, "seed2_eval_meta": split_meta_seed2},
        "grounding_train": grounding_train,
        "probe_before": probe_before,
        "probe_after": probe_after,
        "recommendation": "advance_to_causal_patch" if max(float(probe_after["bag"]["eval_accuracy"]), float(probe_after["positional"]["eval_accuracy"])) > max(float(probe_before["bag"]["eval_accuracy"]), float(probe_before["positional"]["eval_accuracy"])) + 0.05 else "grounding_did_not_rescue_system1",
    }

    (out_dir / "m4_2_predicate_grounding_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# M4.2 Predicate Grounding",
        "",
        f"- grounding same_loss: `{float(grounding_train['same_loss']):.4f}`",
        f"- grounding diff_loss: `{float(grounding_train['diff_loss']):.4f}`",
        f"- grounding sharp_loss: `{float(grounding_train['sharp_loss']):.4f}`",
        "",
        "## Probe Before",
        f"- bag eval: `{float(probe_before['bag']['eval_accuracy']):.4f}`",
        f"- positional eval: `{float(probe_before['positional']['eval_accuracy']):.4f}`",
        f"- dominant_ops: `{probe_before['dominant_ops']}`",
        "",
        "## Probe After",
        f"- bag eval: `{float(probe_after['bag']['eval_accuracy']):.4f}`",
        f"- positional eval: `{float(probe_after['positional']['eval_accuracy']):.4f}`",
        f"- dominant_ops: `{probe_after['dominant_ops']}`",
        "",
        f"- recommendation: `{report['recommendation']}`",
    ]
    (out_dir / "m4_2_predicate_grounding_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_dir / 'm4_2_predicate_grounding_report.json'}")


if __name__ == "__main__":
    main()
