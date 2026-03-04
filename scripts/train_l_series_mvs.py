from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import generate_dataset, split_dataset
from lojban_evolution.l_series import (
    AugmentedLagrangianController,
    build_scope_tokens_from_triples,
    compute_arity_violation,
    compute_identity_violation_from_ce,
    compute_scope_violation_rate,
    crispness_loss,
    entropy_floor_penalty,
    lukasiewicz_and,
    lukasiewicz_or,
    make_swap_variant,
    overflow_penalty,
    rolling_all_below,
    to_float_dict,
)

# Reuse stable H5 primitives.
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    adapter_disabled,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)


@dataclass
class StepTelemetry:
    step: int
    task_loss: float
    constraint_arity: float
    constraint_scope: float
    constraint_identity: float
    lambda_arity: float
    lambda_scope: float
    lambda_identity: float
    total_loss: float
    tier_b_enabled: bool
    tier_c_enabled: bool


def _safe_item(x: torch.Tensor) -> float:
    return float(x.detach().item())


def _decode_triples(tokens: Sequence[torch.Tensor]) -> List[Tuple[int, int, int]]:
    flat = [int(t[0].detach().item()) for t in tokens]
    triples: List[Tuple[int, int, int]] = []
    for i in range(0, len(flat) - 2, 3):
        triples.append((flat[i], flat[i + 1], flat[i + 2]))
    return triples


def _advisor_state_from_tokens(tokens: Sequence[torch.Tensor], arity_head: AdvisorArityHead, codebook: BooleanAnchorTable) -> Tuple[torch.Tensor, torch.Tensor]:
    adv_state = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
    adv_ids = torch.stack(tokens, dim=1)
    return adv_state, adv_ids


def _teacher_forced_ce(
    model,
    tokenizer,
    advisor_adapter,
    advisor_state: torch.Tensor,
    advisor_ids: torch.Tensor,
    prompt: str,
    answer: str,
    layer_index: int,
    max_answer_tokens: int,
) -> torch.Tensor:
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(model.device)
    t_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :max_answer_tokens]
    with adapter_disabled(model):
        ce = torch.zeros((), device=model.device)
        cur_emb = model.get_input_embeddings()(p_ids)
        ptr = 0
        for t in range(t_ids.shape[1]):
            ptr_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
            with persistent_advisor_hook(model, layer_index, advisor_adapter, advisor_state, advisor_ids, ptr_ids, 1.0):
                out = model(inputs_embeds=cur_emb, use_cache=False)
            ce = ce + F.cross_entropy(out.logits[:, -1, :], t_ids[:, t])
            cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(t_ids[:, t : t + 1])], dim=1)
            ptr = min(ptr + 1, max(0, advisor_ids.shape[1] - 1))
    return ce


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L-Series MVS trainer (Lexicographic Augmented Lagrangian).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-answer-tokens", type=int, default=12)
    p.add_argument("--layer-index", type=int, default=12)

    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--init-lambda", type=float, default=0.0)
    p.add_argument("--max-lambda", type=float, default=100.0)
    p.add_argument("--tier-a-lock-eps", type=float, default=0.02)
    p.add_argument("--tier-a-lock-window", type=int, default=16)

    p.add_argument("--weight-tier-b", type=float, default=0.2)
    p.add_argument("--weight-tier-c", type=float, default=0.2)

    p.add_argument("--crispness-weight", type=float, default=1.0)
    p.add_argument("--entropy-h-min", type=float, default=0.8)
    p.add_argument("--register-capacity", type=int, default=64)
    p.add_argument("--ovf-token-id", type=int, default=1999)
    p.add_argument("--identity-margin", type=float, default=0.05)
    p.add_argument("--output-root", type=Path, default=Path("runs/l_series"))

    p.add_argument("--use-iron-collar", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")

    tokenizer.add_special_tokens({"additional_special_tokens": ["[ADVANCE]", "[OVF]"]})
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    advisor_adapter = CouncilCrossAttentionAdapter(hidden_size).to(model.device, dtype=model.dtype)
    arity_head = AdvisorArityHead(hidden_size, 2000).to(model.device, dtype=model.dtype)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=model.device)
        codebook.load_state_dict(ckpt["codebook_state"])
        advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
        cs, ps = arity_head.state_dict(), ckpt["arity_head_state"]
        for n, p in ps.items():
            if n in cs and cs[n].shape == p.shape:
                cs[n].copy_(p)
        arity_head.load_state_dict(cs)

    params = list(codebook.parameters()) + list(advisor_adapter.parameters()) + list(arity_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    ds = generate_dataset(size=int(args.dataset_size), seed=int(args.seed))
    _, _, test = split_dataset(ds)

    controller = AugmentedLagrangianController(
        rho=float(args.rho),
        constraints=("arity", "scope", "identity"),
        init_lambda=float(args.init_lambda),
        max_lambda=float(args.max_lambda),
    )

    recent_tier_a: List[Dict[str, float]] = []
    tier_b_enabled = False
    tier_c_enabled = False

    telemetry: List[StepTelemetry] = []

    for step in range(int(args.train_steps)):
        item = test[step % len(test)]

        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, int(args.max_logic_new_tokens)).to(model.dtype)

        z_st, _idx, cb_loss, commit_loss = codebook.quantize(h_t)
        tokens, logits, _ = arity_head.decode_with_arity(z_st, use_iron_collar=args.use_iron_collar)
        adv_state, adv_ids = _advisor_state_from_tokens(tokens, arity_head, codebook)

        triples = _decode_triples(tokens)
        c_arity_val = compute_arity_violation(triples, relation_vocab=5, var_min_id=5)
        scope_tokens = build_scope_tokens_from_triples(triples, var_prefix="VAR")
        c_scope_val = compute_scope_violation_rate(scope_tokens)

        ce_base = _teacher_forced_ce(
            model=model,
            tokenizer=tokenizer,
            advisor_adapter=advisor_adapter,
            advisor_state=adv_state,
            advisor_ids=adv_ids,
            prompt=item.prompt,
            answer=item.answer,
            layer_index=int(args.layer_index),
            max_answer_tokens=int(args.max_answer_tokens),
        )

        swapped = make_swap_variant(item.prompt, item.answer)
        ce_swap: Optional[torch.Tensor] = None
        if swapped is not None:
            sw_prompt, sw_answer = swapped
            ce_swap = _teacher_forced_ce(
                model=model,
                tokenizer=tokenizer,
                advisor_adapter=advisor_adapter,
                advisor_state=adv_state,
                advisor_ids=adv_ids,
                prompt=sw_prompt,
                answer=sw_answer,
                layer_index=int(args.layer_index),
                max_answer_tokens=int(args.max_answer_tokens),
            )
        c_identity = compute_identity_violation_from_ce(ce_base, ce_swap, margin=float(args.identity_margin))

        c_terms = {
            "arity": torch.tensor(float(c_arity_val), device=model.device, dtype=model.dtype),
            "scope": torch.tensor(float(c_scope_val), device=model.device, dtype=model.dtype),
            "identity": c_identity.to(dtype=model.dtype),
        }

        tier_a_penalty = controller.penalty(c_terms, device=model.device)

        recent_tier_a.append({k: float(v.detach().item()) for k, v in c_terms.items()})
        if len(recent_tier_a) > int(args.tier_a_lock_window):
            recent_tier_a = recent_tier_a[-int(args.tier_a_lock_window) :]
        tier_a_locked = rolling_all_below(recent_tier_a, threshold=float(args.tier_a_lock_eps))
        if tier_a_locked:
            tier_b_enabled = True
            tier_c_enabled = True

        tier_b_loss = torch.zeros((), device=model.device)
        if tier_b_enabled:
            rel_logits = torch.stack(logits[0::3], dim=1)
            rel_probs = torch.softmax(rel_logits, dim=-1)
            p_t = rel_probs[..., 0]
            p_f = rel_probs[..., 1]
            p_u = 1.0 - torch.clamp(p_t + p_f, max=1.0)
            and_soft = lukasiewicz_and(p_t, 1.0 - p_f)
            or_soft = lukasiewicz_or(p_t, p_u)
            tier_b_loss = crispness_loss(torch.clamp(0.5 * (and_soft + or_soft), min=0.0, max=1.0)) * float(args.crispness_weight)

        tier_c_loss = torch.zeros((), device=model.device)
        if tier_c_enabled:
            rel_logits = torch.stack(logits[0::3], dim=1)
            rel_probs = torch.softmax(rel_logits, dim=-1)
            c_div = entropy_floor_penalty(rel_probs, h_min=float(args.entropy_h_min))
            active_vars = len({v1 for _, v1, _ in triples}.union({v2 for _, _, v2 in triples}))
            ovf_present = any(int(tok[0].detach().item()) == int(args.ovf_token_id) for tok in tokens)
            c_ovf = torch.tensor(
                overflow_penalty(active_refs=active_vars, capacity=int(args.register_capacity), ovf_emitted=ovf_present),
                device=model.device,
                dtype=model.dtype,
            )
            tier_c_loss = c_div + c_ovf

        task_loss = ce_base + cb_loss + (0.25 * commit_loss)
        total_loss = (
            task_loss
            + tier_a_penalty
            + (float(args.weight_tier_b) * tier_b_loss)
            + (float(args.weight_tier_c) * tier_c_loss)
        )

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        codebook.enforce_anchor_values()

        controller.update({k: float(v.detach().item()) for k, v in c_terms.items()})

        telemetry.append(
            StepTelemetry(
                step=step + 1,
                task_loss=_safe_item(task_loss),
                constraint_arity=float(c_arity_val),
                constraint_scope=float(c_scope_val),
                constraint_identity=float(c_identity.detach().item()),
                lambda_arity=float(controller.lambdas["arity"]),
                lambda_scope=float(controller.lambdas["scope"]),
                lambda_identity=float(controller.lambdas["identity"]),
                total_loss=_safe_item(total_loss),
                tier_b_enabled=bool(tier_b_enabled),
                tier_c_enabled=bool(tier_c_enabled),
            )
        )

        if (step + 1) % 10 == 0:
            print(
                f"Step {step+1}/{args.train_steps} | total={_safe_item(total_loss):.4f} "
                f"cA={c_arity_val:.3f} cS={c_scope_val:.3f} cI={float(c_identity.detach().item()):.3f} "
                f"lam=[{controller.lambdas['arity']:.3f},{controller.lambdas['scope']:.3f},{controller.lambdas['identity']:.3f}]"
            )

    ckpt_path = run_dir / "l_series_checkpoint.pt"
    torch.save(
        {
            "codebook_state": codebook.state_dict(),
            "advisor_adapter_state": advisor_adapter.state_dict(),
            "arity_head_state": arity_head.state_dict(),
            "lambdas": dict(controller.lambdas),
            "rho": float(controller.rho),
        },
        ckpt_path,
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "final_lambdas": dict(controller.lambdas),
        "final_step": asdict(telemetry[-1]) if telemetry else None,
        "tier_b_enabled": bool(tier_b_enabled),
        "tier_c_enabled": bool(tier_c_enabled),
        "steps": [asdict(t) for t in telemetry],
    }
    summary_path = run_dir / "l_series_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {ckpt_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
