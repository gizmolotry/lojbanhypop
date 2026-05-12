from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from lojban_evolution.m19.engine import M19SymbioteBridge, ensure_special_tokens, m19_injection_hook
from lojban_evolution.m19.family import (
    M19_HIDDEN_SIZE,
    M19_REGISTRY,
    M19_SCRATCHPAD_TOKEN,
    M19_SYMBIOTE_END_TOKEN,
)
from lojban_evolution.m19.typed_physics import parse_typed_slot_layout
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


def _dtype_for_runtime(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _track_key(track: str) -> str:
    candidate = str(track or "").strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _dynamic_mode_enabled(args: argparse.Namespace) -> bool:
    return bool(args.dynamic_pacing or _track_key(args.track) == "M19.4")


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


def _prediction_record(item: dict[str, Any], reg_id: str, gen_text: str, token_count: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    runway_token_count = int((extra or {}).get("runway_token_count", token_count))
    row = {
        "regime": reg_id,
        "prompt": item["prompt"],
        "answer": item["answer"],
        "prediction": gen_text,
        "token_count": token_count,
        "runway_token_count": runway_token_count,
        "correct": scoring_fn(gen_text, str(item["answer"])),
        "phrase_correct": phrase_scoring_fn(gen_text, str(item["answer"])),
    }
    if extra:
        row.update(extra)
    return row


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [_normalize_answer(str(row.get("prediction", ""))) for row in rows]
    golds = [_normalize_answer(str(row.get("answer", ""))) for row in rows]
    counts = Counter(normalized)
    gold_counts = Counter(golds)
    total = max(1, len(rows))
    empty_count = sum(1 for item in normalized if not item)
    one_token_count = sum(1 for item in normalized if len(item.split()) == 1)
    copied_gold_count = sum(1 for pred, gold in zip(normalized, golds) if pred == gold)
    return {
        "sample_count": len(rows),
        "unique_prediction_count": len(counts),
        "unique_gold_count": len(gold_counts),
        "empty_prediction_rate": empty_count / total,
        "one_token_prediction_rate": one_token_count / total,
        "gold_copy_rate": copied_gold_count / total,
        "top_predictions": [
            {"prediction": prediction, "count": count, "rate": count / total}
            for prediction, count in counts.most_common(10)
        ],
        "top_gold_answers": [
            {"answer": answer, "count": count, "rate": count / total}
            for answer, count in gold_counts.most_common(10)
        ],
    }


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _max_judri_slots(typed_slot_layout: str | list[str]) -> int:
    layout = parse_typed_slot_layout(typed_slot_layout) if str(typed_slot_layout).strip() else []
    return max(1, sum(1 for item in layout if str(item) == "judri"))


def _oracle_arity_from_item(item: dict[str, Any], max_judri_slots: int) -> int | None:
    logic = str(item.get("target_logic") or "")
    if not logic:
        return None
    active = 0
    for raw in re.findall(r"<loj_(\d+)>", logic):
        value = int(raw)
        if value > 2000:
            active += 1
    if active <= 0:
        return None
    return max(1, min(int(max_judri_slots), active))


def _arity_override_for_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    item_index: int,
    max_judri_slots: int,
) -> int | None:
    mode = str(args.arity_override_mode).strip().lower()
    if mode in {"", "predicted", "no_mask"}:
        return None
    if mode == "oracle":
        return _oracle_arity_from_item(item, max_judri_slots)
    if mode == "random":
        rng = random.Random((int(args.seed) * 1000003) + int(item_index))
        return rng.randint(1, max(1, int(max_judri_slots)))
    if mode == "force":
        return max(1, min(int(max_judri_slots), int(args.force_arity)))
    raise ValueError(f"unsupported arity override mode: {args.arity_override_mode}")


def _result_metric(results: dict[str, dict[str, float]], regime_id: str | None, metric: str) -> float | None:
    if not regime_id:
        return None
    row = results.get(str(regime_id))
    if not isinstance(row, dict):
        return None
    value = row.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _efficiency_row(
    regime_id: str,
    results: dict[str, dict[str, float]],
    en_cot_accuracy: float | None,
    en_cot_tokens: float | None,
) -> dict[str, float | str | None]:
    regime = results[regime_id]
    accuracy = regime.get("accuracy")
    avg_tokens = regime.get("avg_tokens")
    avg_runway_tokens = regime.get("avg_runway_tokens")
    retention_vs_en_cot = _safe_div(accuracy, en_cot_accuracy)
    token_ratio_vs_en_cot = _safe_div(avg_tokens, en_cot_tokens)
    runway_token_ratio_vs_en_cot = _safe_div(avg_runway_tokens, en_cot_tokens)
    compression_adjusted_retention = None
    if retention_vs_en_cot is not None and token_ratio_vs_en_cot not in (None, 0):
        compression_adjusted_retention = retention_vs_en_cot / token_ratio_vs_en_cot
    runway_compression_adjusted_retention = None
    if retention_vs_en_cot is not None and runway_token_ratio_vs_en_cot not in (None, 0):
        runway_compression_adjusted_retention = retention_vs_en_cot / runway_token_ratio_vs_en_cot
    return {
        "regime": regime_id,
        "accuracy": accuracy,
        "phrase_accuracy": regime.get("phrase_accuracy"),
        "avg_tokens": avg_tokens,
        "accuracy_per_token": _safe_div(accuracy, avg_tokens),
        "avg_runway_tokens": avg_runway_tokens,
        "accuracy_per_runway_token": _safe_div(accuracy, avg_runway_tokens),
        "retention_vs_en_cot": retention_vs_en_cot,
        "token_ratio_vs_en_cot": token_ratio_vs_en_cot,
        "runway_token_ratio_vs_en_cot": runway_token_ratio_vs_en_cot,
        "compression_adjusted_retention": compression_adjusted_retention,
        "runway_compression_adjusted_retention": runway_compression_adjusted_retention,
    }


def _resolve_report_path(args: argparse.Namespace) -> Path:
    registry = M19_REGISTRY[_track_key(args.track)]
    output_root = Path(args.output_root)
    default_root = Path(M19_REGISTRY["M19"]["output_roots"]["benchmark"])
    if _track_key(args.track) != "M19" and output_root == default_root:
        output_root = Path(registry["output_roots"]["benchmark"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / args.run_id
    report_path = Path(args.output_path) if args.output_path else (run_dir / registry["report_names"]["benchmark"])
    validate_series_outputs("M", [output_root], [run_dir])
    assert_output_path_allowed("M", report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _load_samples(dataset_path: Path, eval_size: int) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[: int(eval_size)]


def _filter_regimes(regimes: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
    wanted = {part.strip() for part in str(requested or "").split(",") if part.strip()}
    if not wanted:
        return regimes
    return [row for row in regimes if str(row.get("id")) in wanted]


def _build_instruction_prompt(question: str, instruction: str) -> str:
    return str(question) + str(instruction)


def _build_static_prompt(question: str, scratchpad_token: str, scratchpad_length: int) -> str:
    return (
        "Solve the logic question.\n\n"
        f"Question: {question}\n"
        + " ".join([scratchpad_token] * int(scratchpad_length))
        + "\nFinal answer:"
    )


def _build_prompt_core(question: str) -> str:
    return f"Solve the logic question.\n\nQuestion: {question}\n"


def _decode_generated(tokenizer: AutoTokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _run_instruction_regime(
    backbone,
    tokenizer,
    device: str,
    question: str,
    instruction: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    prompt = _build_instruction_prompt(question, instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = backbone.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][prompt_len:].tolist()
    return _decode_generated(tokenizer, gen_ids), len(gen_ids)


def _run_static_bridge_regime(
    backbone,
    tokenizer,
    bridge: M19SymbioteBridge | None,
    device: str,
    model_dtype: torch.dtype,
    question: str,
    scratchpad_token: str,
    scratchpad_token_id: int,
    scratchpad_length: int,
    tap_layer: int,
    hidden_size: int,
    random_scale: float,
    max_new_tokens: int,
    mode: str,
    gumbel_temperature: float,
    arity_override: int | None,
    disable_arity_mask: bool,
    bridge_channel_mode: str,
) -> tuple[str, int, dict[str, Any]]:
    prompt = _build_static_prompt(question, scratchpad_token, scratchpad_length)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    scratchpad_mask = inputs.input_ids.eq(scratchpad_token_id)

    with torch.no_grad():
        out_prompt = backbone(**inputs, output_hidden_states=True)
        h_tap = out_prompt.hidden_states[int(tap_layer)]
        if mode == "random":
            delta = torch.randn(
                1,
                int(scratchpad_length),
                int(hidden_size),
                device=device,
                dtype=model_dtype,
            ) * float(random_scale)
        elif mode == "none":
            delta = None
        else:
            assert bridge is not None
            delta, _, _, telemetry = bridge(
                h_tap,
                active_steps=int(scratchpad_length),
                gumbel_temperature=gumbel_temperature,
                arity_override=arity_override,
                disable_arity_mask=disable_arity_mask,
                bridge_channel_mode=bridge_channel_mode,
            )

    extra = {
        "latent_steps": int(scratchpad_length),
        "halt_similarity_last": 0.0,
        "premature_stop": False,
        "max_cap_hit": False,
        "scratchpad_bleed": False,
        "typed_family_accuracy": None,
        "arity_violation_rate": None,
        "masked_pointer_zero_rate": None,
        "family_slot_entropy": None,
        "symbolic_trace_alignment": None,
        "predicate_pointer_radial_gap": None,
        "family_radius_violation_rate": None,
        "hyperbolic_geodesic_margin": None,
        "hyperbolic_projection_clip_rate": None,
    }
    if mode == "bridge":
        slot_family_ids = telemetry.get("slot_family_ids") or []
        slot_family_logits = telemetry.get("slot_family_logits")
        if slot_family_logits is not None and slot_family_ids:
            family_targets = torch.tensor(slot_family_ids, device=slot_family_logits.device, dtype=torch.long)
            extra["typed_family_accuracy"] = float(
                (slot_family_logits[0].argmax(dim=-1) == family_targets).float().mean().item()
            )
        masked_pointer_zero_rate = telemetry.get("masked_pointer_zero_rate")
        extra["masked_pointer_zero_rate"] = (
            float(masked_pointer_zero_rate) if masked_pointer_zero_rate is not None else None
        )
        extra["family_slot_entropy"] = float(telemetry.get("slot_family_entropy", 0.0))
        active_budget = telemetry.get("active_arity_budget")
        if active_budget is not None:
            extra["active_arity_budget"] = int(active_budget.detach().cpu().flatten()[0].item())
        extra["arity_mask_disabled"] = bool(telemetry.get("arity_mask_disabled", False))
        hyper_metrics = telemetry.get("hyperbolic_metrics", {})
        extra["bridge_channel_mode"] = str(telemetry.get("bridge_channel_mode", bridge_channel_mode))
        extra["bridge_channel_retained_slot_fraction"] = float(telemetry.get("bridge_channel_retained_slot_fraction", 1.0))
        extra["bridge_channel_family_energy_before"] = telemetry.get("bridge_channel_family_energy_before", {})
        extra["bridge_channel_family_energy_after"] = telemetry.get("bridge_channel_family_energy_after", {})
        for key in (
            "predicate_pointer_radial_gap",
            "family_radius_violation_rate",
            "hyperbolic_geodesic_margin",
            "hyperbolic_projection_clip_rate",
        ):
            if key in hyper_metrics:
                extra[key] = float(hyper_metrics.get(key, 0.0))
    ctx = m19_injection_hook(backbone, int(tap_layer), scratchpad_mask, delta) if delta is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            out = backbone.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    gen_ids = out[0][prompt_len:].tolist()
    return _decode_generated(tokenizer, gen_ids), len(gen_ids), extra


def _run_dynamic_bridge_regime(
    backbone,
    tokenizer,
    bridge: M19SymbioteBridge | None,
    device: str,
    model_dtype: torch.dtype,
    question: str,
    scratchpad_token_id: int,
    symbiote_end_token_id: int,
    tap_layer: int,
    hidden_size: int,
    random_scale: float,
    min_latent_steps: int,
    max_latent_steps: int,
    max_new_tokens: int,
    mode: str,
    gumbel_temperature: float,
    arity_override: int | None,
    disable_arity_mask: bool,
    bridge_channel_mode: str,
) -> tuple[str, int, dict[str, Any]]:
    prompt_core = _build_prompt_core(question)
    current_ids = tokenizer(prompt_core, return_tensors="pt").input_ids[0].tolist() + [int(scratchpad_token_id)]
    halt_emitted = False
    max_cap_hit = False
    scratchpad_bleed = False
    halt_similarity_trace: list[float] = []
    family_accuracy_values: list[float] = []
    masked_zero_values: list[float] = []
    family_entropy_values: list[float] = []
    active_budget_values: list[float] = []
    channel_retained_values: list[float] = []
    channel_mode_value = str(bridge_channel_mode)
    channel_energy_before: dict[str, float] = {}
    channel_energy_after: dict[str, float] = {}
    hyper_metrics_rollup: dict[str, list[float]] = {
        "predicate_pointer_radial_gap": [],
        "family_radius_violation_rate": [],
        "hyperbolic_geodesic_margin": [],
        "hyperbolic_projection_clip_rate": [],
    }

    while True:
        inputs = {"input_ids": torch.tensor([current_ids], device=device, dtype=torch.long)}
        scratchpad_mask = inputs["input_ids"].eq(int(scratchpad_token_id))
        active_steps = int(scratchpad_mask.sum().item())
        if active_steps >= int(max_latent_steps):
            max_cap_hit = True
            break

        with torch.no_grad():
            out_prompt = backbone(**inputs, output_hidden_states=True)
            h_tap = out_prompt.hidden_states[int(tap_layer)]
            delta = None
            if mode == "random":
                delta = torch.randn(
                    1,
                    active_steps,
                    int(hidden_size),
                    device=device,
                    dtype=model_dtype,
                ) * float(random_scale)
            elif mode == "bridge":
                assert bridge is not None
                lengths = torch.tensor([active_steps], device=device, dtype=torch.long)
                delta, _, _, telemetry = bridge(
                    h_tap,
                    active_steps=active_steps,
                    lengths=lengths,
                    gumbel_temperature=gumbel_temperature,
                    arity_override=arity_override,
                    disable_arity_mask=disable_arity_mask,
                    bridge_channel_mode=bridge_channel_mode,
                )
                sim_row = telemetry["halt_cosine_per_step"][0, :active_steps].detach().float().cpu().tolist()
                if sim_row:
                    halt_similarity_trace.append(float(sim_row[-1]))
                slot_family_ids = telemetry.get("slot_family_ids") or []
                slot_family_logits = telemetry.get("slot_family_logits")
                if slot_family_logits is not None and slot_family_ids:
                    family_targets = torch.tensor(slot_family_ids, device=slot_family_logits.device, dtype=torch.long)
                    family_accuracy_values.append(
                        float((slot_family_logits[0].argmax(dim=-1) == family_targets).float().mean().item())
                    )
                masked_pointer_zero_rate = telemetry.get("masked_pointer_zero_rate")
                if masked_pointer_zero_rate is not None:
                    masked_zero_values.append(float(masked_pointer_zero_rate))
                family_entropy_values.append(float(telemetry.get("slot_family_entropy", 0.0)))
                active_budget = telemetry.get("active_arity_budget")
                if active_budget is not None:
                    active_budget_values.append(float(active_budget.detach().cpu().flatten()[0].item()))
                channel_mode_value = str(telemetry.get("bridge_channel_mode", bridge_channel_mode))
                channel_retained_values.append(float(telemetry.get("bridge_channel_retained_slot_fraction", 1.0)))
                if isinstance(telemetry.get("bridge_channel_family_energy_before"), dict):
                    channel_energy_before = telemetry.get("bridge_channel_family_energy_before", {})
                if isinstance(telemetry.get("bridge_channel_family_energy_after"), dict):
                    channel_energy_after = telemetry.get("bridge_channel_family_energy_after", {})
                hyper_metrics = telemetry.get("hyperbolic_metrics", {})
                for key in hyper_metrics_rollup:
                    if key in hyper_metrics:
                        hyper_metrics_rollup[key].append(float(hyper_metrics.get(key, 0.0)))

        ctx = m19_injection_hook(backbone, int(tap_layer), scratchpad_mask, delta) if delta is not None else nullcontext()
        with ctx:
            with torch.no_grad():
                out = backbone.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        next_token_id = int(out[0][-1].item())
        if next_token_id == int(symbiote_end_token_id):
            halt_emitted = True
            current_ids = out[0].tolist()
            break
        if next_token_id == int(scratchpad_token_id):
            current_ids = out[0].tolist()
            continue
        scratchpad_bleed = True
        break

    latent_steps = max(1, sum(1 for token_id in current_ids if int(token_id) == int(scratchpad_token_id)))
    premature_stop = latent_steps < int(min_latent_steps)
    answer_context_ids = list(current_ids)
    if not halt_emitted:
        answer_context_ids.append(int(symbiote_end_token_id))
    answer_context_ids.extend(tokenizer("\nFinal answer:", add_special_tokens=False).input_ids)
    answer_inputs = {"input_ids": torch.tensor([answer_context_ids], device=device, dtype=torch.long)}
    prompt_len = answer_inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = backbone.generate(
            **answer_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_ids = out[0][prompt_len:].tolist()
    total_generated_tokens = len(answer_ids) + int(latent_steps) + (1 if halt_emitted else 0)
    extra = {
        "latent_steps": int(latent_steps),
        "halt_emitted": bool(halt_emitted),
        "premature_stop": bool(premature_stop),
        "max_cap_hit": bool(max_cap_hit),
        "scratchpad_bleed": bool(scratchpad_bleed),
        "halt_similarity_trace": halt_similarity_trace,
        "halt_similarity_last": float(halt_similarity_trace[-1]) if halt_similarity_trace else 0.0,
        "typed_family_accuracy": (sum(family_accuracy_values) / max(1, len(family_accuracy_values))) if family_accuracy_values else None,
        "arity_violation_rate": None,
        "active_arity_budget": int(round(sum(active_budget_values) / max(1, len(active_budget_values)))) if active_budget_values else None,
        "arity_mask_disabled": bool(disable_arity_mask),
        "masked_pointer_zero_rate": (sum(masked_zero_values) / max(1, len(masked_zero_values))) if masked_zero_values else None,
        "family_slot_entropy": (sum(family_entropy_values) / max(1, len(family_entropy_values))) if family_entropy_values else None,
        "symbolic_trace_alignment": None,
        "bridge_channel_mode": channel_mode_value,
        "bridge_channel_retained_slot_fraction": (sum(channel_retained_values) / max(1, len(channel_retained_values))) if channel_retained_values else None,
        "bridge_channel_family_energy_before": channel_energy_before,
        "bridge_channel_family_energy_after": channel_energy_after,
    }
    for key, values in hyper_metrics_rollup.items():
        extra[key] = (sum(values) / max(1, len(values))) if values else None
    return _decode_generated(tokenizer, answer_ids), total_generated_tokens, extra


def run_godtier_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = _dtype_for_runtime(device)
    _set_seed(int(args.seed))

    track_key = _track_key(args.track)
    registry = M19_REGISTRY[track_key]
    dynamic_mode = _dynamic_mode_enabled(args)
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

    token_ids = ensure_special_tokens(
        backbone,
        tokenizer,
        [str(args.scratchpad_token), str(args.symbiote_end_token)],
    )
    scratchpad_token_id = int(token_ids[str(args.scratchpad_token)])
    symbiote_end_token_id = int(token_ids[str(args.symbiote_end_token)])

    bridge = None
    if args.bridge_path:
        bridge = M19SymbioteBridge(
            hidden_size=int(args.hidden_size),
            bottleneck_dim=int(args.bottleneck_dim),
            scratchpad_len=int(args.scratchpad_length),
            num_queries=int(args.num_queries),
            max_latent_steps=int(args.max_latent_steps),
            typed_slot_layout=parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else None,
            geometry_mode=str(args.geometry_mode),
            arity_router_mode=str(args.arity_router_mode),
            gumbel_hard=bool(args.gumbel_hard),
            poincare_curvature=float(args.poincare_curvature),
        ).to(device=device, dtype=model_dtype)
        bridge.load_state_dict(torch.load(args.bridge_path, map_location=device), strict=False)
        bridge.eval()

    static_bridge = None
    if args.static_bridge_path:
        static_bridge = M19SymbioteBridge(
            hidden_size=int(args.hidden_size),
            bottleneck_dim=int(args.static_bottleneck_dim),
            scratchpad_len=int(args.static_scratchpad_length),
            num_queries=int(args.static_num_queries),
            max_latent_steps=max(int(args.static_scratchpad_length), int(args.max_latent_steps)),
            typed_slot_layout=parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else None,
            geometry_mode=str(args.geometry_mode),
            arity_router_mode=str(args.arity_router_mode),
            gumbel_hard=bool(args.gumbel_hard),
            poincare_curvature=float(args.poincare_curvature),
        ).to(device=device, dtype=model_dtype)
        static_bridge.load_state_dict(torch.load(args.static_bridge_path, map_location=device), strict=False)
        static_bridge.eval()

    samples = _load_samples(Path(args.dataset_path), int(args.eval_size))
    max_judri_slots = _max_judri_slots(args.typed_slot_layout)
    disable_arity_mask = str(args.arity_override_mode).strip().lower() == "no_mask"
    regimes: list[dict[str, Any]] = [
        {"id": "BASE", "kind": "instruction", "instruction": "\nAnswer with one word or short phrase.", "max_new_tokens": 32},
        {"id": "EN-COT", "kind": "instruction", "instruction": "\nThink step-by-step. Final answer after 'Answer: '.", "max_new_tokens": 128},
        {"id": "ZH-COT", "kind": "instruction", "instruction": "\n请一步一步思考，并在最后用“答案：”给出最终答案。", "max_new_tokens": 128},
    ]
    if dynamic_mode:
        regimes.append({"id": "SCRATCHPAD-ONLY", "kind": "dynamic", "mode": "none", "max_new_tokens": 32})
        if bool(args.include_random_control):
            regimes.append({"id": "RANDOM-DYNAMIC", "kind": "dynamic", "mode": "random", "max_new_tokens": 32})
        if static_bridge is not None:
            regimes.append({"id": str(args.static_cell_id), "kind": "static", "mode": "bridge", "max_new_tokens": 32})
        if bridge is not None:
            regimes.append({"id": str(args.cell_id), "kind": "dynamic", "mode": "bridge", "max_new_tokens": 32})
    else:
        if bool(args.include_random_control):
            regimes.append({"id": "RANDOM-SHAPE", "kind": "static", "mode": "random", "max_new_tokens": 32})
        if bridge is not None:
            regimes.append({"id": "SCRATCHPAD-ONLY", "kind": "static", "mode": "none", "max_new_tokens": 32})
            regimes.append({"id": str(args.cell_id), "kind": "static", "mode": "bridge", "max_new_tokens": 32})
    regimes = _filter_regimes(regimes, args.regimes)

    results: dict[str, dict[str, float]] = {}
    sample_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_summaries: dict[str, dict[str, Any]] = {}
    dynamic_rollup: dict[str, dict[str, float | list[float]]] = {}

    for reg in regimes:
        correct = 0
        phrase_correct = 0
        total_tokens = 0
        total_runway_tokens = 0
        typed_family_values: list[float] = []
        masked_pointer_values: list[float] = []
        family_entropy_values: list[float] = []
        arity_violation_values: list[float] = []
        channel_retained_values: list[float] = []
        radial_gap_values: list[float] = []
        radial_violation_values: list[float] = []
        geodesic_margin_values: list[float] = []
        clip_rate_values: list[float] = []
        total_premature = 0
        total_cap = 0
        total_bleed = 0
        halt_similarity_last_values: list[float] = []
        entanglement_values: list[float] = []
        rows: list[dict[str, Any]] = []
        print(f"\n--- RUNNING REGIME: {reg['id']} ---")
        for item_index, item in enumerate(tqdm(samples, desc=reg["id"])):
            arity_override = _arity_override_for_item(args, item, item_index, max_judri_slots)
            if reg["kind"] == "instruction":
                gen_text, token_count = _run_instruction_regime(
                    backbone,
                    tokenizer,
                    device,
                    str(item["prompt"]),
                    str(reg["instruction"]),
                    int(reg["max_new_tokens"]),
                )
                extra = {}
            elif reg["kind"] == "static":
                active_bridge = static_bridge if str(reg["id"]) == str(args.static_cell_id) and static_bridge is not None else bridge
                active_scratchpad_len = int(args.static_scratchpad_length) if str(reg["id"]) == str(args.static_cell_id) and static_bridge is not None else int(args.scratchpad_length)
                gen_text, token_count, extra = _run_static_bridge_regime(
                    backbone,
                    tokenizer,
                    active_bridge,
                    device,
                    model_dtype,
                    str(item["prompt"]),
                    str(args.scratchpad_token),
                    scratchpad_token_id,
                    active_scratchpad_len,
                    int(args.tap_layer),
                    int(args.hidden_size),
                    float(args.random_scale),
                    int(reg["max_new_tokens"]),
                    str(reg["mode"]),
                    float(args.gumbel_temp_end),
                    arity_override,
                    disable_arity_mask,
                    str(args.bridge_channel_mode),
                )
                extra["runway_token_count"] = int(token_count) + int(extra.get("latent_steps") or active_scratchpad_len)
            else:
                gen_text, token_count, extra = _run_dynamic_bridge_regime(
                    backbone,
                    tokenizer,
                    bridge,
                    device,
                    model_dtype,
                    str(item["prompt"]),
                    scratchpad_token_id,
                    symbiote_end_token_id,
                    int(args.tap_layer),
                    int(args.hidden_size),
                    float(args.random_scale),
                    int(args.min_latent_steps),
                    int(args.max_latent_steps),
                    int(reg["max_new_tokens"]),
                    str(reg["mode"]),
                    float(args.gumbel_temp_end),
                    arity_override,
                    disable_arity_mask,
                    str(args.bridge_channel_mode),
                )
                total_premature += 1 if bool(extra.get("premature_stop")) else 0
                total_cap += 1 if bool(extra.get("max_cap_hit")) else 0
                total_bleed += 1 if bool(extra.get("scratchpad_bleed")) else 0
                halt_similarity_last_values.append(float(extra.get("halt_similarity_last") or 0.0))
                halt_trace = [float(v) for v in extra.get("halt_similarity_trace", [])]
                if halt_trace:
                    entanglement_values.append(sum(halt_trace[:-1]) / max(1, len(halt_trace[:-1])))
                extra["runway_token_count"] = int(token_count)

            total_tokens += token_count
            if "runway_token_count" not in extra:
                extra["runway_token_count"] = int(token_count)
            total_runway_tokens += int(extra["runway_token_count"])
            if isinstance(extra.get("typed_family_accuracy"), (int, float)):
                typed_family_values.append(float(extra["typed_family_accuracy"]))
            if isinstance(extra.get("masked_pointer_zero_rate"), (int, float)):
                masked_pointer_values.append(float(extra["masked_pointer_zero_rate"]))
            if isinstance(extra.get("family_slot_entropy"), (int, float)):
                family_entropy_values.append(float(extra["family_slot_entropy"]))
            if isinstance(extra.get("bridge_channel_retained_slot_fraction"), (int, float)):
                channel_retained_values.append(float(extra["bridge_channel_retained_slot_fraction"]))
            oracle_arity = _oracle_arity_from_item(item, max_judri_slots)
            active_budget = extra.get("active_arity_budget")
            if oracle_arity is not None and isinstance(active_budget, (int, float)):
                arity_violation_values.append(0.0 if int(round(float(active_budget))) == int(oracle_arity) else 1.0)
            if isinstance(extra.get("predicate_pointer_radial_gap"), (int, float)):
                radial_gap_values.append(float(extra["predicate_pointer_radial_gap"]))
            if isinstance(extra.get("family_radius_violation_rate"), (int, float)):
                radial_violation_values.append(float(extra["family_radius_violation_rate"]))
            if isinstance(extra.get("hyperbolic_geodesic_margin"), (int, float)):
                geodesic_margin_values.append(float(extra["hyperbolic_geodesic_margin"]))
            if isinstance(extra.get("hyperbolic_projection_clip_rate"), (int, float)):
                clip_rate_values.append(float(extra["hyperbolic_projection_clip_rate"]))
            is_correct = scoring_fn(gen_text, str(item["answer"]))
            is_phrase_correct = phrase_scoring_fn(gen_text, str(item["answer"]))
            if is_correct:
                correct += 1
            if is_phrase_correct:
                phrase_correct += 1
            rows.append(_prediction_record(item, str(reg["id"]), gen_text, token_count, extra))

        acc = correct / max(1, len(samples))
        phrase_acc = phrase_correct / max(1, len(samples))
        tok = total_tokens / max(1, len(samples))
        runway_tok = total_runway_tokens / max(1, len(samples))
        results[str(reg["id"])] = {
            "accuracy": acc,
            "phrase_accuracy": phrase_acc,
            "avg_tokens": tok,
            "avg_runway_tokens": runway_tok,
            "accuracy_per_runway_token": _safe_div(acc, runway_tok),
            "typed_family_accuracy": (sum(typed_family_values) / max(1, len(typed_family_values))) if typed_family_values else None,
            "arity_violation_rate": (sum(arity_violation_values) / max(1, len(arity_violation_values))) if arity_violation_values else None,
            "masked_pointer_zero_rate": (sum(masked_pointer_values) / max(1, len(masked_pointer_values))) if masked_pointer_values else None,
            "family_slot_entropy": (sum(family_entropy_values) / max(1, len(family_entropy_values))) if family_entropy_values else None,
            "bridge_channel_retained_slot_fraction": (sum(channel_retained_values) / max(1, len(channel_retained_values))) if channel_retained_values else None,
            "predicate_pointer_radial_gap": (sum(radial_gap_values) / max(1, len(radial_gap_values))) if radial_gap_values else None,
            "family_radius_violation_rate": (sum(radial_violation_values) / max(1, len(radial_violation_values))) if radial_violation_values else None,
            "hyperbolic_geodesic_margin": (sum(geodesic_margin_values) / max(1, len(geodesic_margin_values))) if geodesic_margin_values else None,
            "hyperbolic_projection_clip_rate": (sum(clip_rate_values) / max(1, len(clip_rate_values))) if clip_rate_values else None,
        }
        if reg["kind"] == "dynamic":
            results[str(reg["id"])].update(
                {
                    "premature_stop_rate": total_premature / max(1, len(samples)),
                    "max_cap_hit_rate": total_cap / max(1, len(samples)),
                    "scratchpad_bleed_rate": total_bleed / max(1, len(samples)),
                    "caa_manifold_entanglement_score": sum(entanglement_values) / max(1, len(entanglement_values)),
                }
            )
            dynamic_rollup[str(reg["id"])] = {
                "halt_similarity_last_mean": sum(halt_similarity_last_values) / max(1, len(halt_similarity_last_values)),
                "halt_similarity_last_values": halt_similarity_last_values,
            }
        sample_predictions[str(reg["id"])] = rows[: min(10, len(rows))]
        prediction_summaries[str(reg["id"])] = _prediction_summary(rows)

    mainline_key = str(args.cell_id) if str(args.cell_id) in results else None
    static_key = str(args.static_cell_id) if str(args.static_cell_id) in results else None
    en_cot_accuracy = _result_metric(results, "EN-COT", "accuracy")
    en_cot_tokens = _result_metric(results, "EN-COT", "avg_tokens")
    efficiency_table = [_efficiency_row(regime_id, results, en_cot_accuracy, en_cot_tokens) for regime_id in results]
    mainline_accuracy = _result_metric(results, mainline_key, "accuracy")
    mainline_phrase_accuracy = _result_metric(results, mainline_key, "phrase_accuracy")
    mainline_tokens = _result_metric(results, mainline_key, "avg_tokens")
    mainline_runway_tokens = _result_metric(results, mainline_key, "avg_runway_tokens")
    base_accuracy = _result_metric(results, "BASE", "accuracy")
    zh_cot_accuracy = _result_metric(results, "ZH-COT", "accuracy")
    zh_cot_tokens = _result_metric(results, "ZH-COT", "avg_tokens")
    static_accuracy = _result_metric(results, static_key, "accuracy")
    static_tokens = _result_metric(results, static_key, "avg_tokens")
    static_runway_tokens = _result_metric(results, static_key, "avg_runway_tokens")
    random_dynamic_accuracy = _result_metric(results, "RANDOM-DYNAMIC", "accuracy")
    random_shape_accuracy = _result_metric(results, "RANDOM-SHAPE", "accuracy")
    random_accuracy = random_dynamic_accuracy if random_dynamic_accuracy is not None else random_shape_accuracy

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.benchmark", "scripts/m19/run_m19_godtier_benchmark.py"),
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=str(args.bridge_path).replace("\\", "/") if args.bridge_path else None,
            checkpoint_out=None,
            dataset_profile="m14_5_unified_test",
            difficulty_tier="mixed",
        ),
        "track": track_key,
        "family_contract": {
            "family_name": registry["family"],
            "implementation_label": registry["implementation_label"],
            "runner_script": registry["runner_scripts"]["benchmark"],
            "dag": registry["dags"]["benchmark"],
        },
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/") if args.bridge_path else None,
            "static_bridge_path": str(args.static_bridge_path).replace("\\", "/") if args.static_bridge_path else None,
            "dataset_path": str(args.dataset_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "cell_id": str(args.cell_id),
            "static_cell_id": str(args.static_cell_id),
            "track": track_key,
            "dynamic_pacing": dynamic_mode,
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "static_num_queries": int(args.static_num_queries),
            "static_bottleneck_dim": int(args.static_bottleneck_dim),
            "static_scratchpad_length": int(args.static_scratchpad_length),
            "min_latent_steps": int(args.min_latent_steps),
            "max_latent_steps": int(args.max_latent_steps),
            "scratchpad_token": str(args.scratchpad_token),
            "symbiote_end_token": str(args.symbiote_end_token),
            "tap_layer": int(args.tap_layer),
            "hidden_size": int(args.hidden_size),
            "seed": int(args.seed),
            "include_random_control": bool(args.include_random_control),
            "random_scale": float(args.random_scale),
            "typed_slot_layout": parse_typed_slot_layout(args.typed_slot_layout) if str(args.typed_slot_layout).strip() else [],
            "arity_router_mode": str(args.arity_router_mode),
            "arity_override_mode": str(args.arity_override_mode),
            "force_arity": int(args.force_arity),
            "gumbel_hard": bool(args.gumbel_hard),
            "gumbel_temp_start": float(args.gumbel_temp_start),
            "gumbel_temp_end": float(args.gumbel_temp_end),
            "geometry_mode": str(args.geometry_mode),
            "poincare_curvature": float(args.poincare_curvature),
            "bridge_channel_mode": str(args.bridge_channel_mode),
        },
        "results": results,
        "dynamic_rollup": dynamic_rollup,
        "comparisons": {
            "mainline_vs_base_accuracy_delta": (mainline_accuracy - base_accuracy if mainline_accuracy is not None and base_accuracy is not None else None),
            "mainline_vs_en_cot_accuracy_delta": (mainline_accuracy - en_cot_accuracy if mainline_accuracy is not None and en_cot_accuracy is not None else None),
            "mainline_vs_zh_cot_accuracy_delta": (mainline_accuracy - zh_cot_accuracy if mainline_accuracy is not None and zh_cot_accuracy is not None else None),
            "mainline_vs_static_m19_3_accuracy_delta": (mainline_accuracy - static_accuracy if mainline_accuracy is not None and static_accuracy is not None else None),
            "mainline_vs_static_m19_3_token_ratio": _safe_div(mainline_tokens, static_tokens),
            "mainline_vs_en_cot_token_ratio": _safe_div(mainline_tokens, en_cot_tokens),
            "mainline_vs_en_cot_runway_token_ratio": _safe_div(mainline_runway_tokens, en_cot_tokens),
            "mainline_vs_zh_cot_token_ratio": _safe_div(mainline_tokens, zh_cot_tokens),
        },
        "efficiency_table": efficiency_table,
        "prediction_summaries": prediction_summaries,
        "metrics": {
            "strict_accuracy": mainline_accuracy,
            "overall_accuracy": mainline_accuracy,
            "overall_phrase_accuracy": mainline_phrase_accuracy,
            "avg_tokens": mainline_tokens,
            "accuracy_per_token": _safe_div(mainline_accuracy, mainline_tokens),
            "avg_runway_tokens": mainline_runway_tokens,
            "accuracy_per_runway_token": _safe_div(mainline_accuracy, mainline_runway_tokens),
            "base_accuracy": base_accuracy,
            "en_cot_accuracy": en_cot_accuracy,
            "en_cot_avg_tokens": en_cot_tokens,
            "zh_cot_accuracy": zh_cot_accuracy,
            "zh_cot_avg_tokens": zh_cot_tokens,
            "static_m19_3_accuracy": static_accuracy,
            "static_m19_3_avg_tokens": static_tokens,
            "static_m19_3_avg_runway_tokens": static_runway_tokens,
            "random_accuracy": random_accuracy,
            "lift_vs_base": (mainline_accuracy - base_accuracy if mainline_accuracy is not None and base_accuracy is not None else None),
            "lift_vs_en_cot": (mainline_accuracy - en_cot_accuracy if mainline_accuracy is not None and en_cot_accuracy is not None else None),
            "lift_vs_zh_cot": (mainline_accuracy - zh_cot_accuracy if mainline_accuracy is not None and zh_cot_accuracy is not None else None),
            "lift_vs_static_m19_3": (mainline_accuracy - static_accuracy if mainline_accuracy is not None and static_accuracy is not None else None),
            "lift_vs_random": (mainline_accuracy - random_accuracy if mainline_accuracy is not None and random_accuracy is not None else None),
            "retention_vs_en_cot": _safe_div(mainline_accuracy, en_cot_accuracy),
            "token_ratio_vs_en_cot": _safe_div(mainline_tokens, en_cot_tokens),
            "runway_token_ratio_vs_en_cot": _safe_div(mainline_runway_tokens, en_cot_tokens),
            "compression_adjusted_retention": (
                _safe_div(
                    _safe_div(mainline_accuracy, en_cot_accuracy),
                    _safe_div(mainline_tokens, en_cot_tokens),
                )
            ),
            "runway_compression_adjusted_retention": (
                _safe_div(
                    _safe_div(mainline_accuracy, en_cot_accuracy),
                    _safe_div(mainline_runway_tokens, en_cot_tokens),
                )
            ),
            "premature_stop_rate": _result_metric(results, mainline_key, "premature_stop_rate"),
            "max_cap_hit_rate": _result_metric(results, mainline_key, "max_cap_hit_rate"),
            "scratchpad_bleed_rate": _result_metric(results, mainline_key, "scratchpad_bleed_rate"),
            "caa_manifold_entanglement_score": _result_metric(results, mainline_key, "caa_manifold_entanglement_score"),
            "typed_family_accuracy": _result_metric(results, mainline_key, "typed_family_accuracy"),
            "arity_violation_rate": _result_metric(results, mainline_key, "arity_violation_rate"),
            "masked_pointer_zero_rate": _result_metric(results, mainline_key, "masked_pointer_zero_rate"),
            "family_slot_entropy": _result_metric(results, mainline_key, "family_slot_entropy"),
            "bridge_channel_retained_slot_fraction": _result_metric(results, mainline_key, "bridge_channel_retained_slot_fraction"),
            "symbolic_trace_alignment": _result_metric(results, mainline_key, "symbolic_trace_alignment"),
            "predicate_pointer_radial_gap": _result_metric(results, mainline_key, "predicate_pointer_radial_gap"),
            "family_radius_violation_rate": _result_metric(results, mainline_key, "family_radius_violation_rate"),
            "hyperbolic_geodesic_margin": _result_metric(results, mainline_key, "hyperbolic_geodesic_margin"),
            "hyperbolic_projection_clip_rate": _result_metric(results, mainline_key, "hyperbolic_projection_clip_rate"),
        },
        "headline": {
            "overall_accuracy": mainline_accuracy,
            "avg_tokens": mainline_tokens,
            "accuracy_per_token": _safe_div(mainline_accuracy, mainline_tokens),
            "avg_runway_tokens": mainline_runway_tokens,
            "accuracy_per_runway_token": _safe_div(mainline_accuracy, mainline_runway_tokens),
            "en_cot_accuracy": en_cot_accuracy,
            "zh_cot_accuracy": zh_cot_accuracy,
            "static_m19_3_accuracy": static_accuracy,
            "typed_family_accuracy": _result_metric(results, mainline_key, "typed_family_accuracy"),
            "masked_pointer_zero_rate": _result_metric(results, mainline_key, "masked_pointer_zero_rate"),
            "family_slot_entropy": _result_metric(results, mainline_key, "family_slot_entropy"),
            "bridge_channel_retained_slot_fraction": _result_metric(results, mainline_key, "bridge_channel_retained_slot_fraction"),
        },
        "sample_predictions": sample_predictions,
        "report_path": str(report_path).replace("\\", "/"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Run the M19 benchmark against concise, English-CoT, Chinese-CoT, and runway controls.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", type=Path, default=None)
    parser.add_argument("--static-bridge-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-data-path", type=Path, default=None)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--min-latent-steps", type=int, default=4)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--static-num-queries", type=int, default=8)
    parser.add_argument("--static-bottleneck-dim", type=int, default=128)
    parser.add_argument("--static-scratchpad-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--scratchpad-token", type=str, default=M19_SCRATCHPAD_TOKEN)
    parser.add_argument("--symbiote-end-token", type=str, default=M19_SYMBIOTE_END_TOKEN)
    parser.add_argument("--cell-id", type=str, default="M19.3-PROBE")
    parser.add_argument("--static-cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--include-random-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pacing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--arity-override-mode", type=str, default="predicted", choices=["predicted", "oracle", "random", "force", "no_mask"])
    parser.add_argument("--force-arity", type=int, default=1)
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--bridge-channel-mode", type=str, default="full")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--track", type=str, default="")
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["benchmark"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--regimes", type=str, default="")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.eval_data_path is not None:
        args.dataset_path = args.eval_data_path
    track_key = _track_key(args.track)
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
    run_godtier_benchmark(parse_args())
