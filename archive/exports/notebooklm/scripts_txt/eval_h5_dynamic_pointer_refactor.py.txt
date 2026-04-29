from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import torch

# Ensure repo root is importable when run as `python scripts/...py`.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.train_h5_persistent_vq_advisor import (
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)
from scripts.train_h5_slice2_bridge import AdvisorArityHead as PointerAdvisorArityHead


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    answer: str


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and (p == e or p.startswith(e))


def _cases(limit: int) -> List[EvalCase]:
    rows = [
        EvalCase("A is left of B. B is left of C. Which is leftmost?", "A"),
        EvalCase("D is north of E. E is north of F. Which is farthest north?", "D"),
        EvalCase("Breakfast happens before lunch. Lunch happens before dinner. What happens first?", "breakfast"),
        EvalCase("Task A is after Task B. Task B is after Task C. Which is earliest?", "Task C"),
        EvalCase("Tom is left of Uma. Uma is left of Vic. Who is in the middle?", "Uma"),
        EvalCase("Monday is before Tuesday. Tuesday is before Wednesday. Which day is second?", "Tuesday"),
        EvalCase("M is east of N. N is east of O. Who is westmost?", "O"),
        EvalCase("Draft before Review. Review before Publish. What comes right before Publish?", "Review"),
    ]
    return rows[: max(1, limit)]


def _decode_standard(arity_head: AdvisorArityHead, z_st: torch.Tensor, use_iron_collar: bool) -> List[torch.Tensor]:
    try:
        toks, _, _ = arity_head.decode_with_arity(z_st, use_iron_collar=use_iron_collar)
    except TypeError:
        toks, _, _ = arity_head.decode_with_arity(z_st)
    return toks


def _build_arity_head_from_checkpoint(ckpt: Dict, hidden_size: int, device, dtype):
    state = ckpt["arity_head_state"]
    rel_out = int(state["head_rel.weight"].shape[0])
    var1_out = int(state["head_var1.weight"].shape[0])
    if var1_out == rel_out:
        head = AdvisorArityHead(hidden_size, rel_out).to(device, dtype=dtype)
        head.load_state_dict(state, strict=False)
        return head
    head = PointerAdvisorArityHead(hidden_size=hidden_size, codebook_size=rel_out, pointer_vocab=var1_out).to(
        device, dtype=dtype
    )
    head.load_state_dict(state, strict=False)
    return head


def _decode_dynamic_pointer(
    arity_head: AdvisorArityHead,
    z_st: torch.Tensor,
    pointer_window: int,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    b, l, _ = z_st.shape
    assert b == 1, "Evaluator currently supports batch_size=1."
    tokens: List[torch.Tensor] = []
    rel_history: List[int] = []
    pointer_slots: List[int] = []
    oob_count = 0
    self_ref = 0

    for i in range(l):
        z = z_st[:, i, :]
        rel_id = torch.argmax(arity_head.head_rel(z), dim=-1)
        raw_v1 = torch.argmax(arity_head.head_var1(z), dim=-1)
        raw_v2 = torch.argmax(arity_head.head_var2(z), dim=-1)
        rel_i = int(rel_id.item())

        if rel_history:
            slot_cap = max(1, min(pointer_window, len(rel_history)))
            v1_slot = int(raw_v1.item()) % slot_cap
            v2_slot = int(raw_v2.item()) % slot_cap
            ref1 = rel_history[-1 - v1_slot]
            ref2 = rel_history[-1 - v2_slot]
            pointer_slots.extend([v1_slot, v2_slot])
            if v1_slot == 0:
                self_ref += 1
            if v2_slot == 0:
                self_ref += 1
        else:
            # Cold start: point variable slots at current relation id.
            ref1 = rel_i
            ref2 = rel_i
            pointer_slots.extend([0, 0])
            self_ref += 2
            oob_count += 2

        rel_history.append(rel_i)
        tokens.extend(
            [
                torch.tensor([rel_i], device=z_st.device),
                torch.tensor([ref1], device=z_st.device),
                torch.tensor([ref2], device=z_st.device),
            ]
        )

    slots = torch.tensor(pointer_slots, dtype=torch.float32) if pointer_slots else torch.zeros(1)
    ptr_stats = {
        "avg_pointer_slot": float(slots.mean().item()),
        "max_pointer_slot": float(slots.max().item()),
        "self_ref_rate": float(self_ref) / float(max(1, len(pointer_slots))),
        "cold_start_oob_ratio": float(oob_count) / float(max(1, len(pointer_slots))),
        "unique_slots_used": int(len(set(pointer_slots))),
    }
    return tokens, ptr_stats


def _generate_answer(
    model,
    tokenizer,
    adapter_mod,
    codebook,
    arity_head: AdvisorArityHead,
    tokens: Sequence[torch.Tensor],
    layer_index: int,
    inject_scale: float,
    prompt: str,
    max_final_new_tokens: int,
) -> str:
    advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
    advisor_ids = torch.stack(list(tokens), dim=1)

    prefix = build_final_prefix(prompt)
    cur_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    cur_emb = model.get_input_embeddings()(cur_ids)

    advance_id = tokenizer.convert_tokens_to_ids("[ADVANCE]")
    ptr = 0
    generated: List[int] = []
    for _ in range(max_final_new_tokens):
        p_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
        with persistent_advisor_hook(model, layer_index, adapter_mod, advisor_states, advisor_ids, p_ids, inject_scale):
            out = model(inputs_embeds=cur_emb, return_dict=True)
        nid = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        generated.append(nid)
        if tokenizer.eos_token_id is not None and nid == int(tokenizer.eos_token_id):
            break
        if nid == advance_id:
            ptr = min(ptr + 3, max(0, len(tokens) - 3))
        next_emb = model.get_input_embeddings()(torch.tensor([[nid]], device=model.device))
        cur_emb = torch.cat([cur_emb, next_emb], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dynamic pointer refactor mode (variable slots as pointer indices).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=8)
    p.add_argument("--pointer-window", type=int, default=16)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=24)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--relation-bias", type=float, default=0.0)
    p.add_argument("--use-iron-collar", action="store_true")
    p.add_argument("--output", type=Path, default=Path("runs/h5_dynamic_pointer_eval.json"))
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    if args.pointer_window < 1:
        raise ValueError("--pointer-window must be >= 1")
    return args


def main() -> None:
    args = parse_args()

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

    if tokenizer.convert_tokens_to_ids("[ADVANCE]") == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ADVANCE]"]})
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=model.device)
    hidden_size = int(model.config.hidden_size)

    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    adapter_mod = CouncilCrossAttentionAdapter(hidden_size, use_boolean_surgery=True).to(model.device, dtype=model.dtype)
    adapter_mod.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    arity_head = _build_arity_head_from_checkpoint(ckpt, hidden_size, model.device, model.dtype)
    rows = []
    ptr_stats_accum = {
        "avg_pointer_slot": 0.0,
        "max_pointer_slot": 0.0,
        "self_ref_rate": 0.0,
        "cold_start_oob_ratio": 0.0,
        "unique_slots_used": 0.0,
    }

    for case in _cases(args.sample_size):
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, case.prompt, max_logic_new_tokens=args.max_logic_new_tokens)
            z_st, _, _, _ = codebook.quantize(h_t, relation_bias=args.relation_bias)
            std_tokens = _decode_standard(arity_head, z_st, use_iron_collar=args.use_iron_collar)
            dyn_tokens, dyn_stats = _decode_dynamic_pointer(arity_head, z_st, pointer_window=args.pointer_window)

        pred_std = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            adapter_mod=adapter_mod,
            codebook=codebook,
            arity_head=arity_head,
            tokens=std_tokens,
            layer_index=args.layer_index,
            inject_scale=args.inject_scale,
            prompt=case.prompt,
            max_final_new_tokens=args.max_final_new_tokens,
        )
        pred_dyn = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            adapter_mod=adapter_mod,
            codebook=codebook,
            arity_head=arity_head,
            tokens=dyn_tokens,
            layer_index=args.layer_index,
            inject_scale=args.inject_scale,
            prompt=case.prompt,
            max_final_new_tokens=args.max_final_new_tokens,
        )

        std_ok = _answer_match(case.answer, pred_std)
        dyn_ok = _answer_match(case.answer, pred_dyn)
        for k in ptr_stats_accum:
            ptr_stats_accum[k] += float(dyn_stats[k])

        rows.append(
            {
                "prompt": case.prompt,
                "expected_answer": case.answer,
                "standard_prediction": pred_std,
                "dynamic_prediction": pred_dyn,
                "standard_correct": bool(std_ok),
                "dynamic_correct": bool(dyn_ok),
                "dynamic_pointer_stats": dyn_stats,
            }
        )

    total = len(rows)
    std_correct = sum(1 for r in rows if r["standard_correct"])
    dyn_correct = sum(1 for r in rows if r["dynamic_correct"])
    denom = float(max(1, total))
    ptr_means = {k: float(v) / denom for k, v in ptr_stats_accum.items()}

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "checkpoint": str(args.checkpoint),
        "config": {
            "sample_size": args.sample_size,
            "pointer_window": args.pointer_window,
            "max_logic_new_tokens": args.max_logic_new_tokens,
            "max_final_new_tokens": args.max_final_new_tokens,
            "layer_index": args.layer_index,
            "inject_scale": args.inject_scale,
            "relation_bias": args.relation_bias,
            "use_iron_collar": args.use_iron_collar,
        },
        "summary": {
            "total": total,
            "standard_correct": std_correct,
            "dynamic_correct": dyn_correct,
            "standard_accuracy": float(std_correct) / denom,
            "dynamic_accuracy": float(dyn_correct) / denom,
            "dynamic_minus_standard_accuracy": float(dyn_correct - std_correct) / denom,
        },
        "dynamic_pointer_metrics": ptr_means,
        "samples": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"dynamic acc: {payload['summary']['dynamic_accuracy']:.3f}")


if __name__ == "__main__":
    main()
