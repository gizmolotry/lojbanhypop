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
class OODCase:
    domain: str
    prompt: str
    answer: str


def _normalize(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.strip().lower())


def _answer_match(expected: str, predicted: str) -> bool:
    e = _normalize(expected)
    p = _normalize(predicted)
    return bool(e) and (p == e or p.startswith(e))


def _spatial_cases() -> List[OODCase]:
    return [
        OODCase("spatial", "A is left of B. B is left of C. Which is leftmost?", "A"),
        OODCase("spatial", "D is north of E. E is north of F. Which is farthest north?", "D"),
        OODCase("spatial", "K is right of L. L is right of M. Which is rightmost?", "K"),
        OODCase("spatial", "P is south of Q. Q is south of R. Which is farthest south?", "P"),
        OODCase("spatial", "X is west of Y. Y is west of Z. Which is farthest east?", "Z"),
        OODCase("spatial", "Tom is left of Uma. Uma is left of Vic. Who is in the middle?", "Uma"),
        OODCase("spatial", "A is above B. B is above C. Who is lowest?", "C"),
        OODCase("spatial", "M is east of N. N is east of O. Who is westmost?", "O"),
        OODCase("spatial", "I is north of J. J is east of K. Which one is north of K?", "I"),
        OODCase("spatial", "R is south of S. S is west of T. Which one is west of T?", "S"),
    ]


def _temporal_cases() -> List[OODCase]:
    return [
        OODCase("temporal", "Breakfast happens before lunch. Lunch happens before dinner. What happens first?", "breakfast"),
        OODCase("temporal", "Alpha starts before Beta. Beta starts before Gamma. Which starts last?", "Gamma"),
        OODCase("temporal", "Task A is after Task B. Task B is after Task C. Which is earliest?", "Task C"),
        OODCase("temporal", "Spring comes before Summer, and Summer before Autumn. What comes after Summer?", "Autumn"),
        OODCase("temporal", "Phase 1 ends before Phase 2 starts. Phase 2 ends before Phase 3 starts. Which phase is middle?", "Phase 2"),
        OODCase("temporal", "Event X occurs after Event Y. Event Y occurs after Event Z. Which occurs last?", "Event X"),
        OODCase("temporal", "Monday is before Tuesday. Tuesday is before Wednesday. Which day is second?", "Tuesday"),
        OODCase("temporal", "Checkpoint A before B; B before C; C before D. Which checkpoint is first?", "A"),
        OODCase("temporal", "Input stage before Transform stage. Transform before Output stage. Which stage is last?", "Output"),
        OODCase("temporal", "Draft before Review. Review before Publish. What comes right before Publish?", "Review"),
    ]


def _build_suite(per_domain_limit: int) -> List[OODCase]:
    spatial = _spatial_cases()[: max(1, per_domain_limit)]
    temporal = _temporal_cases()[: max(1, per_domain_limit)]
    return spatial + temporal


def _decode_tokens(arity_head: AdvisorArityHead, z_st: torch.Tensor, use_iron_collar: bool) -> List[torch.Tensor]:
    try:
        tokens, _, _ = arity_head.decode_with_arity(z_st, use_iron_collar=use_iron_collar)
    except TypeError:
        tokens, _, _ = arity_head.decode_with_arity(z_st)
    return tokens


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


def _generate_with_hook(
    model,
    tokenizer,
    adapter_mod,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    prompt: str,
    layer_index: int,
    inject_scale: float,
    max_logic_new_tokens: int,
    max_final_new_tokens: int,
    relation_bias: float,
    use_iron_collar: bool,
) -> Dict:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, max_logic_new_tokens=max_logic_new_tokens)
        z_st, idx, _, _ = codebook.quantize(h_t, relation_bias=relation_bias)
        tokens = _decode_tokens(arity_head, z_st, use_iron_collar=use_iron_collar)
        advisor_states = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
        advisor_ids = torch.stack(tokens, dim=1)

        prefix = build_final_prefix(prompt)
        cur_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        cur_emb = model.get_input_embeddings()(cur_ids)
        advance_token_id = tokenizer.convert_tokens_to_ids("[ADVANCE]")
        ptr = 0
        ptr_advances = 0

        generated: List[int] = []
        for _ in range(max_final_new_tokens):
            p_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
            with persistent_advisor_hook(model, layer_index, adapter_mod, advisor_states, advisor_ids, p_ids, inject_scale):
                out = model(inputs_embeds=cur_emb, return_dict=True)
            next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
            if next_id == advance_token_id:
                ptr = min(ptr + 3, max(0, len(tokens) - 3))
                ptr_advances += 1
            next_emb = model.get_input_embeddings()(torch.tensor([[next_id]], device=model.device))
            cur_emb = torch.cat([cur_emb, next_emb], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    surgery_hits = int((idx < 5).sum().item())
    return {
        "prediction": text,
        "trace_token_count": int(idx.shape[1]),
        "surgery_hits": surgery_hits,
        "pointer_advances": ptr_advances,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD stress evaluator for H5 spatial/temporal prompts using persistent advisor hooks.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="H5 checkpoint with codebook/adapter/arity states.")
    p.add_argument("--per-domain-limit", type=int, default=10)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=24)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--relation-bias", type=float, default=0.0)
    p.add_argument("--use-iron-collar", action="store_true")
    p.add_argument("--output", type=Path, default=Path("runs/h5_ood_stress.json"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


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

    suite = _build_suite(args.per_domain_limit)
    rows = []
    by_domain: Dict[str, Dict[str, float]] = {
        "spatial": {"total": 0, "correct": 0, "surgery_hits": 0, "trace_token_total": 0, "pointer_advances": 0},
        "temporal": {"total": 0, "correct": 0, "surgery_hits": 0, "trace_token_total": 0, "pointer_advances": 0},
    }

    for case in suite:
        result = _generate_with_hook(
            model=model,
            tokenizer=tokenizer,
            adapter_mod=adapter_mod,
            codebook=codebook,
            arity_head=arity_head,
            prompt=case.prompt,
            layer_index=args.layer_index,
            inject_scale=args.inject_scale,
            max_logic_new_tokens=args.max_logic_new_tokens,
            max_final_new_tokens=args.max_final_new_tokens,
            relation_bias=args.relation_bias,
            use_iron_collar=args.use_iron_collar,
        )
        ok = _answer_match(case.answer, result["prediction"])
        d = by_domain[case.domain]
        d["total"] += 1
        d["correct"] += int(ok)
        d["surgery_hits"] += result["surgery_hits"]
        d["trace_token_total"] += result["trace_token_count"]
        d["pointer_advances"] += result["pointer_advances"]
        rows.append(
            {
                "domain": case.domain,
                "prompt": case.prompt,
                "expected_answer": case.answer,
                "prediction": result["prediction"],
                "correct": bool(ok),
                "trace_token_count": result["trace_token_count"],
                "surgery_hits": result["surgery_hits"],
                "pointer_advances": result["pointer_advances"],
            }
        )

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    for dom in ("spatial", "temporal"):
        t = max(1, int(by_domain[dom]["total"]))
        by_domain[dom]["accuracy"] = float(by_domain[dom]["correct"]) / float(t)
        by_domain[dom]["avg_surgery_hits"] = float(by_domain[dom]["surgery_hits"]) / float(t)
        by_domain[dom]["avg_trace_tokens"] = float(by_domain[dom]["trace_token_total"]) / float(t)
        by_domain[dom]["avg_pointer_advances"] = float(by_domain[dom]["pointer_advances"]) / float(t)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "checkpoint": str(args.checkpoint),
        "config": {
            "per_domain_limit": args.per_domain_limit,
            "max_logic_new_tokens": args.max_logic_new_tokens,
            "max_final_new_tokens": args.max_final_new_tokens,
            "layer_index": args.layer_index,
            "inject_scale": args.inject_scale,
            "relation_bias": args.relation_bias,
            "use_iron_collar": args.use_iron_collar,
        },
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": float(correct) / float(max(1, total)),
        },
        "domains": by_domain,
        "samples": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"accuracy: {payload['summary']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
