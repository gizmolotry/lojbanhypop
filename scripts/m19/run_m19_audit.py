from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from lojban_evolution.m19.engine import M19SymbioteBridge, m19_injection_hook
from lojban_evolution.m19.family import M19_HIDDEN_SIZE, M19_REGISTRY, M19_SCRATCHPAD_TOKEN
from lojban_evolution.m19.typed_physics import parse_typed_slot_layout
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)

import re


def _dtype_for_runtime(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _ensure_symbiote_token(model, tokenizer, scratchpad_token: str) -> int:
    vocab = tokenizer.get_vocab()
    if scratchpad_token in vocab:
        token_id = int(tokenizer.convert_tokens_to_ids(scratchpad_token))
        model.resize_token_embeddings(len(tokenizer))
        return token_id
    old_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": [scratchpad_token]})
    model.resize_token_embeddings(len(tokenizer))
    token_id = int(tokenizer.convert_tokens_to_ids(scratchpad_token))
    input_emb = model.get_input_embeddings().weight
    with torch.no_grad():
        mean_in = input_emb[:old_size].mean(dim=0, keepdim=True)
        input_emb[old_size : len(tokenizer)] = mean_in.to(device=input_emb.device, dtype=input_emb.dtype)
    return token_id


def _normalize_answer(text: str) -> str:
    cleaned = text.strip()
    for marker in ("Final answer:", "Answer:", "Final Answer:", "答案：", "答案:", "最终答案：", "最终答案:"):
        if marker.lower() in cleaned.lower():
            idx = cleaned.lower().rfind(marker.lower())
            cleaned = cleaned[idx + len(marker) :].strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    cleaned = re.sub(r"^[\s:：;；\-\.\,，。!！\?？\)\(（）]+|[\s:：;；\-\.\,，。!！\?？\)\(（）]+$", "", cleaned.lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _strip_leading_phrase_tokens(text: str) -> str:
    stripped = text
    while True:
        updated = re.sub(r"^(?:the|a|an|to|in|on|at|under|inside|into|from|of)\s+", "", stripped).strip()
        if updated == stripped:
            return stripped
        stripped = updated


def scoring_fn(prediction: str, target: str) -> bool:
    pred = _normalize_answer(prediction)
    gold = _normalize_answer(target)
    if "," in gold:
        pred_parts = [part.strip() for part in pred.split(",") if part.strip()]
        gold_parts = [part.strip() for part in gold.split(",") if part.strip()]
        return pred_parts == gold_parts
    return pred == gold


def phrase_scoring_fn(prediction: str, target: str) -> bool:
    pred = _normalize_answer(prediction)
    gold = _normalize_answer(target)
    if not pred or not gold:
        return False
    if scoring_fn(prediction, target):
        return True
    if gold in {"yes", "no", "true", "false"} or "," in gold:
        return False
    pred_relaxed = _strip_leading_phrase_tokens(pred)
    gold_relaxed = _strip_leading_phrase_tokens(gold)
    if pred_relaxed == gold_relaxed:
        return True
    if len(gold_relaxed.split()) == 1:
        pred_tokens = pred_relaxed.split()
        if 1 <= len(pred_tokens) <= 4 and pred_tokens[-1] == gold_relaxed:
            return True
    return False


def _resolve_report_path(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / args.run_id
    report_path = Path(args.output_path) if args.output_path else (run_dir / M19_REGISTRY["M19"]["report_names"]["audit"])
    validate_series_outputs("M", [output_root], [run_dir])
    assert_output_path_allowed("M", report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _infer_max_latent_steps(bridge_state: dict[str, Any], configured_steps: int) -> int:
    configured = max(1, int(configured_steps))
    candidates: list[int] = []
    for key in ("output_map.bias", "collar.spatial_embeddings"):
        tensor = bridge_state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            candidates.append(int(tensor.shape[0]))
    return max([configured, *candidates])


def run_m19_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = _dtype_for_runtime(device)
    _set_seed(int(args.seed))

    report_path = _resolve_report_path(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if device != "cuda":
        backbone = backbone.to(device)
    backbone.eval()
    symbiote_id = _ensure_symbiote_token(backbone, tokenizer, str(args.scratchpad_token))

    bridge_state = torch.load(args.bridge_path, map_location=device)
    max_latent_steps = _infer_max_latent_steps(bridge_state, int(args.max_latent_steps))
    bridge = M19SymbioteBridge(
        hidden_size=int(args.hidden_size),
        bottleneck_dim=int(args.bottleneck_dim),
        scratchpad_len=int(args.scratchpad_length),
        num_queries=int(args.num_queries),
        max_latent_steps=max_latent_steps,
        typed_slot_layout=parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else None,
        geometry_mode=str(args.geometry_mode),
        arity_router_mode=str(args.arity_router_mode),
        gumbel_hard=bool(args.gumbel_hard),
        poincare_curvature=float(args.poincare_curvature),
    ).to(device=device, dtype=model_dtype)
    bridge_load = bridge.load_state_dict(bridge_state, strict=False)
    bridge.eval()

    with Path(args.dataset_path).open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()][: int(args.eval_size)]

    results: dict[str, dict[str, float]] = {}
    previews: dict[str, list[dict[str, Any]]] = {}
    regimes = [
        "BASE-NO-SCRATCHPAD",
        "SCRATCHPAD-ONLY",
        "RANDOM",
        "Q-FORMER",
    ]

    for cid in regimes:
        print(f"\n--- RUNNING CELL: {cid} ---")
        correct = 0
        phrase_correct = 0
        total_tokens = 0
        typed_family_values: list[float] = []
        masked_pointer_values: list[float] = []
        family_entropy_values: list[float] = []
        radial_gap_values: list[float] = []
        radial_violation_values: list[float] = []
        geodesic_margin_values: list[float] = []
        rows: list[dict[str, Any]] = []
        for item in tqdm(samples, desc=cid):
            use_scratchpad = cid != "BASE-NO-SCRATCHPAD"
            prompt = "Solve the logic question.\n\n" f"Question: {item['prompt']}\n"
            if use_scratchpad:
                prompt += " ".join([str(args.scratchpad_token)] * int(args.scratchpad_length)) + "\n"
            prompt += "Final answer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            delta = None
            if cid == "Q-FORMER":
                with torch.no_grad():
                    out_prompt = backbone(**inputs, output_hidden_states=True)
                    h_tap = out_prompt.hidden_states[int(args.tap_layer)]
                    delta, _, _, telemetry = bridge(h_tap, gumbel_temperature=float(args.gumbel_temp_end))
                slot_family_ids = telemetry.get("slot_family_ids") or []
                slot_family_logits = telemetry.get("slot_family_logits")
                if slot_family_logits is not None and slot_family_ids:
                    family_targets = torch.tensor(slot_family_ids, device=slot_family_logits.device, dtype=torch.long)
                    typed_family_values.append(
                        float((slot_family_logits[0].argmax(dim=-1) == family_targets).float().mean().item())
                    )
                masked_pointer_zero_rate = telemetry.get("masked_pointer_zero_rate")
                if masked_pointer_zero_rate is not None:
                    masked_pointer_values.append(float(masked_pointer_zero_rate))
                family_entropy_values.append(float(telemetry.get("slot_family_entropy", 0.0)))
                hyper_metrics = telemetry.get("hyperbolic_metrics", {})
                radial_gap_values.append(float(hyper_metrics.get("predicate_pointer_radial_gap", 0.0)))
                radial_violation_values.append(float(hyper_metrics.get("family_radius_violation_rate", 0.0)))
                geodesic_margin_values.append(float(hyper_metrics.get("hyperbolic_geodesic_margin", 0.0)))
            elif cid == "RANDOM":
                delta = torch.randn(
                    1,
                    int(args.scratchpad_length),
                    int(args.hidden_size),
                    device=device,
                    dtype=model_dtype,
                ) * float(args.random_scale)
            scratchpad_mask = inputs.input_ids.eq(symbiote_id)
            if delta is not None and bool(use_scratchpad):
                ctx = m19_injection_hook(backbone, int(args.tap_layer), scratchpad_mask, delta)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx:
                with torch.no_grad():
                    out = backbone.generate(
                        **inputs,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            gen_ids = out[0][inputs.input_ids.shape[1] :].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            total_tokens += len(gen_ids)
            hit = scoring_fn(gen_text, str(item["answer"]))
            phrase_hit = phrase_scoring_fn(gen_text, str(item["answer"]))
            if hit:
                correct += 1
            if phrase_hit:
                phrase_correct += 1
            rows.append(
                {
                    "prompt": item["prompt"],
                    "answer": item["answer"],
                    "prediction": gen_text,
                    "token_count": len(gen_ids),
                    "correct": hit,
                    "phrase_correct": phrase_hit,
                }
            )

        results[cid] = {
            "accuracy": correct / max(1, len(samples)),
            "phrase_accuracy": phrase_correct / max(1, len(samples)),
            "avg_tokens": total_tokens / max(1, len(samples)),
            "typed_family_accuracy": (sum(typed_family_values) / max(1, len(typed_family_values))) if typed_family_values else None,
            "masked_pointer_zero_rate": (sum(masked_pointer_values) / max(1, len(masked_pointer_values))) if masked_pointer_values else None,
            "family_slot_entropy": (sum(family_entropy_values) / max(1, len(family_entropy_values))) if family_entropy_values else None,
            "predicate_pointer_radial_gap": (sum(radial_gap_values) / max(1, len(radial_gap_values))) if radial_gap_values else None,
            "family_radius_violation_rate": (sum(radial_violation_values) / max(1, len(radial_violation_values))) if radial_violation_values else None,
            "hyperbolic_geodesic_margin": (sum(geodesic_margin_values) / max(1, len(geodesic_margin_values))) if geodesic_margin_values else None,
        }
        previews[cid] = rows[: min(10, len(rows))]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.audit", "scripts/m19/run_m19_audit.py"),
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=str(args.bridge_path).replace("\\", "/"),
            checkpoint_out=None,
            dataset_profile="sanity_check_v1",
            difficulty_tier="sanity",
        ),
        "track": "M19.audit",
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "dataset_path": str(args.dataset_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "scratchpad_length": int(args.scratchpad_length),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "hidden_size": int(args.hidden_size),
            "tap_layer": int(args.tap_layer),
            "max_latent_steps": int(max_latent_steps),
            "seed": int(args.seed),
            "random_scale": float(args.random_scale),
            "checkpoint_missing_keys": list(bridge_load.missing_keys),
            "checkpoint_unexpected_keys": list(bridge_load.unexpected_keys),
            "typed_slot_layout": parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else [],
            "arity_router_mode": str(args.arity_router_mode),
            "gumbel_hard": bool(args.gumbel_hard),
            "gumbel_temp_end": float(args.gumbel_temp_end),
            "geometry_mode": str(args.geometry_mode),
            "poincare_curvature": float(args.poincare_curvature),
        },
        "results": results,
        "headline": {
            "base_accuracy": results.get("BASE-NO-SCRATCHPAD", {}).get("accuracy"),
            "base_phrase_accuracy": results.get("BASE-NO-SCRATCHPAD", {}).get("phrase_accuracy"),
            "scratchpad_only_accuracy": results.get("SCRATCHPAD-ONLY", {}).get("accuracy"),
            "scratchpad_only_phrase_accuracy": results.get("SCRATCHPAD-ONLY", {}).get("phrase_accuracy"),
            "random_accuracy": results.get("RANDOM", {}).get("accuracy"),
            "random_phrase_accuracy": results.get("RANDOM", {}).get("phrase_accuracy"),
            "qformer_accuracy": results.get("Q-FORMER", {}).get("accuracy"),
            "qformer_phrase_accuracy": results.get("Q-FORMER", {}).get("phrase_accuracy"),
            "lift_vs_base": (
                (results.get("Q-FORMER", {}).get("accuracy") or 0.0)
                - (results.get("BASE-NO-SCRATCHPAD", {}).get("accuracy") or 0.0)
            ),
            "phrase_lift_vs_base": (
                (results.get("Q-FORMER", {}).get("phrase_accuracy") or 0.0)
                - (results.get("BASE-NO-SCRATCHPAD", {}).get("phrase_accuracy") or 0.0)
            ),
            "lift_vs_scratchpad_only": (
                (results.get("Q-FORMER", {}).get("accuracy") or 0.0)
                - (results.get("SCRATCHPAD-ONLY", {}).get("accuracy") or 0.0)
            ),
            "phrase_lift_vs_scratchpad_only": (
                (results.get("Q-FORMER", {}).get("phrase_accuracy") or 0.0)
                - (results.get("SCRATCHPAD-ONLY", {}).get("phrase_accuracy") or 0.0)
            ),
            "lift_vs_random": (
                (results.get("Q-FORMER", {}).get("accuracy") or 0.0)
                - (results.get("RANDOM", {}).get("accuracy") or 0.0)
            ),
            "phrase_lift_vs_random": (
                (results.get("Q-FORMER", {}).get("phrase_accuracy") or 0.0)
                - (results.get("RANDOM", {}).get("phrase_accuracy") or 0.0)
            ),
            "typed_family_accuracy": results.get("Q-FORMER", {}).get("typed_family_accuracy"),
            "masked_pointer_zero_rate": results.get("Q-FORMER", {}).get("masked_pointer_zero_rate"),
            "family_slot_entropy": results.get("Q-FORMER", {}).get("family_slot_entropy"),
        },
        "sample_predictions": previews,
        "report_path": str(report_path).replace("\\", "/"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 48)
    print(f"{'Cell':<15} | {'Accuracy':<10} | {'Avg Tokens':<10}")
    print("-" * 48)
    for key, value in results.items():
        print(f"{key:<15} | {value['accuracy']:<10.4f} | {value['avg_tokens']:<10.2f}")
    print("=" * 48)
    print(f"Audit report written to {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Run the structured M19 sanity audit.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", required=True, type=Path)
    parser.add_argument("--dataset-path", type=Path, default=Path(registry["dataset_defaults"]["audit"]))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--scratchpad-token", type=str, default=M19_SCRATCHPAD_TOKEN)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--cell-id", type=str, default="")
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["audit"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.data_path is not None:
        args.dataset_path = args.data_path
    track_key = str(args.track).strip() if str(args.track).strip() in M19_REGISTRY else "M19"
    defaults = M19_REGISTRY.get(track_key, {}).get("defaults", {})
    if defaults:
        if not str(args.typed_slot_layout).strip() and defaults.get("typed_slot_layout"):
            args.typed_slot_layout = str(defaults["typed_slot_layout"])
        if str(args.arity_router_mode).strip() == "soft" and defaults.get("arity_router_mode"):
            args.arity_router_mode = str(defaults["arity_router_mode"])
        if str(args.geometry_mode).strip() == "euclidean" and defaults.get("geometry_mode"):
            args.geometry_mode = str(defaults["geometry_mode"])
        if not args.gumbel_hard and str(defaults.get("arity_router_mode", "")).strip() == "gumbel_hard":
            args.gumbel_hard = True
    return args


if __name__ == "__main__":
    run_m19_audit(parse_args())
