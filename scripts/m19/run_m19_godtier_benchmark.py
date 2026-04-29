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
    row = {
        "regime": reg_id,
        "prompt": item["prompt"],
        "answer": item["answer"],
        "prediction": gen_text,
        "token_count": token_count,
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
    retention_vs_en_cot = _safe_div(accuracy, en_cot_accuracy)
    token_ratio_vs_en_cot = _safe_div(avg_tokens, en_cot_tokens)
    compression_adjusted_retention = None
    if retention_vs_en_cot is not None and token_ratio_vs_en_cot not in (None, 0):
        compression_adjusted_retention = retention_vs_en_cot / token_ratio_vs_en_cot
    return {
        "regime": regime_id,
        "accuracy": accuracy,
        "phrase_accuracy": regime.get("phrase_accuracy"),
        "avg_tokens": avg_tokens,
        "accuracy_per_token": _safe_div(accuracy, avg_tokens),
        "retention_vs_en_cot": retention_vs_en_cot,
        "token_ratio_vs_en_cot": token_ratio_vs_en_cot,
        "compression_adjusted_retention": compression_adjusted_retention,
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
            delta, _, _, _ = bridge(h_tap, active_steps=int(scratchpad_length))

    extra = {
        "latent_steps": int(scratchpad_length),
        "halt_similarity_last": 0.0,
        "premature_stop": False,
        "max_cap_hit": False,
        "scratchpad_bleed": False,
    }
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
) -> tuple[str, int, dict[str, Any]]:
    prompt_core = _build_prompt_core(question)
    current_ids = tokenizer(prompt_core, return_tensors="pt").input_ids[0].tolist() + [int(scratchpad_token_id)]
    halt_emitted = False
    max_cap_hit = False
    scratchpad_bleed = False
    halt_similarity_trace: list[float] = []

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
                delta, _, _, telemetry = bridge(h_tap, active_steps=active_steps, lengths=lengths)
                sim_row = telemetry["halt_cosine_per_step"][0, :active_steps].detach().float().cpu().tolist()
                if sim_row:
                    halt_similarity_trace.append(float(sim_row[-1]))

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
    }
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
        ).to(device=device, dtype=model_dtype)
        static_bridge.load_state_dict(torch.load(args.static_bridge_path, map_location=device), strict=False)
        static_bridge.eval()

    samples = _load_samples(Path(args.dataset_path), int(args.eval_size))
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
        total_premature = 0
        total_cap = 0
        total_bleed = 0
        halt_similarity_last_values: list[float] = []
        entanglement_values: list[float] = []
        rows: list[dict[str, Any]] = []
        print(f"\n--- RUNNING REGIME: {reg['id']} ---")
        for item in tqdm(samples, desc=reg["id"]):
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
                )
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
                )
                total_premature += 1 if bool(extra.get("premature_stop")) else 0
                total_cap += 1 if bool(extra.get("max_cap_hit")) else 0
                total_bleed += 1 if bool(extra.get("scratchpad_bleed")) else 0
                halt_similarity_last_values.append(float(extra.get("halt_similarity_last") or 0.0))
                halt_trace = [float(v) for v in extra.get("halt_similarity_trace", [])]
                if halt_trace:
                    entanglement_values.append(sum(halt_trace[:-1]) / max(1, len(halt_trace[:-1])))

            total_tokens += token_count
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
        results[str(reg["id"])] = {"accuracy": acc, "phrase_accuracy": phrase_acc, "avg_tokens": tok}
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
    base_accuracy = _result_metric(results, "BASE", "accuracy")
    zh_cot_accuracy = _result_metric(results, "ZH-COT", "accuracy")
    zh_cot_tokens = _result_metric(results, "ZH-COT", "avg_tokens")
    static_accuracy = _result_metric(results, static_key, "accuracy")
    static_tokens = _result_metric(results, static_key, "avg_tokens")
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
            "base_accuracy": base_accuracy,
            "en_cot_accuracy": en_cot_accuracy,
            "en_cot_avg_tokens": en_cot_tokens,
            "zh_cot_accuracy": zh_cot_accuracy,
            "zh_cot_avg_tokens": zh_cot_tokens,
            "static_m19_3_accuracy": static_accuracy,
            "static_m19_3_avg_tokens": static_tokens,
            "random_accuracy": random_accuracy,
            "lift_vs_base": (mainline_accuracy - base_accuracy if mainline_accuracy is not None and base_accuracy is not None else None),
            "lift_vs_en_cot": (mainline_accuracy - en_cot_accuracy if mainline_accuracy is not None and en_cot_accuracy is not None else None),
            "lift_vs_zh_cot": (mainline_accuracy - zh_cot_accuracy if mainline_accuracy is not None and zh_cot_accuracy is not None else None),
            "lift_vs_static_m19_3": (mainline_accuracy - static_accuracy if mainline_accuracy is not None and static_accuracy is not None else None),
            "lift_vs_random": (mainline_accuracy - random_accuracy if mainline_accuracy is not None and random_accuracy is not None else None),
            "retention_vs_en_cot": _safe_div(mainline_accuracy, en_cot_accuracy),
            "token_ratio_vs_en_cot": _safe_div(mainline_tokens, en_cot_tokens),
            "compression_adjusted_retention": (
                _safe_div(
                    _safe_div(mainline_accuracy, en_cot_accuracy),
                    _safe_div(mainline_tokens, en_cot_tokens),
                )
            ),
            "premature_stop_rate": _result_metric(results, mainline_key, "premature_stop_rate"),
            "max_cap_hit_rate": _result_metric(results, mainline_key, "max_cap_hit_rate"),
            "scratchpad_bleed_rate": _result_metric(results, mainline_key, "scratchpad_bleed_rate"),
            "caa_manifold_entanglement_score": _result_metric(results, mainline_key, "caa_manifold_entanglement_score"),
        },
        "headline": {
            "overall_accuracy": mainline_accuracy,
            "avg_tokens": mainline_tokens,
            "accuracy_per_token": _safe_div(mainline_accuracy, mainline_tokens),
            "en_cot_accuracy": en_cot_accuracy,
            "zh_cot_accuracy": zh_cot_accuracy,
            "static_m19_3_accuracy": static_accuracy,
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
    return args


if __name__ == "__main__":
    run_godtier_benchmark(parse_args())
