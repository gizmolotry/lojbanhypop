from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.l_series import build_scope_tokens_from_triples, compute_scope_violation_components
from lojban_evolution.m_bridge_ablation_family import BRIDGE_ABLATION_REGISTRY
from lojban_evolution.m_reentry_reboot_family import REENTRY_REBOOT_REGISTRY, build_reentry_protocol_manifest, reentry_cell_labels
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from run_m3_15d_answer_path_forcing import (  # type: ignore
    _answer_match,
    _build_winograd_pack,
    _build_winograd_split_packs,
    _load_pack,
    _model_device,
    _triples,
)
from run_m3_17_advisor_reentry_bridge import (  # type: ignore
    AdvisorReentryBridge,
    _advisor_stats,
    _build_base_advisor_state,
    _extract_reentry_context,
    _score_candidate_from_hidden,
    _score_candidate_from_return_tokens,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    build_final_prefix,
)


LOJBAN_BLEED_RE = re.compile(r"(<\s*loj_[^>\s]+>|loj_[a-z0-9_]+)", re.IGNORECASE)
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class DecoderReentryResumeBridge(AdvisorReentryBridge):
    def hybrid_resume(
        self,
        prefix_hidden: torch.Tensor,
        advisor_states: torch.Tensor,
        n_tokens: int,
        *,
        token_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        return_tokens, token_stats = self.return_tokens(prefix_hidden, advisor_states, n_tokens)
        delta, residual_stats = self.residual_delta(prefix_hidden, advisor_states)
        return_tokens = return_tokens * float(token_scale)
        stats = {
            "gate": float((token_stats["gate"] + residual_stats["gate"]) / 2.0),
            "token_norm": float(torch.norm(return_tokens, dim=-1).mean().detach().item()),
            "residual_norm": float(residual_stats["token_norm"]),
            "combined_norm": float(torch.norm(return_tokens, dim=-1).mean().detach().item() + residual_stats["token_norm"]),
            "attn_entropy": float((token_stats["attn_entropy"] + residual_stats["attn_entropy"]) / 2.0),
        }
        return return_tokens, delta, stats


def _latest_named_file(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(file_name), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def _resume_target_text(item: dict[str, Any]) -> str:
    explicit = str(item.get("target_text", "") or item.get("resumption_target", "")).strip()
    if explicit:
        return explicit
    candidates = list(item.get("candidates", []))
    gold_index = int(item.get("gold_index", 0))
    if 0 <= gold_index < len(candidates):
        return str(candidates[gold_index]).strip()
    return ""


def _resume_target_ids(tokenizer, item: dict[str, Any], max_tokens: int) -> list[int]:
    text = _resume_target_text(item)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return [int(x) for x in ids[: max(1, int(max_tokens))]]


def _variant_id_int(item: dict[str, Any]) -> int:
    raw = item.get("variant_id", 0)
    if isinstance(raw, int):
        return int(raw)
    text = str(raw).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _contains_contamination(text: str) -> bool:
    return bool(LOJBAN_BLEED_RE.search(str(text)))


def _loop_flag(token_ids: list[int]) -> bool:
    if len(token_ids) < 3:
        return False
    if len(set(token_ids)) <= max(1, len(token_ids) // 2):
        return True
    for n in (1, 2):
        if len(token_ids) < n * 2:
            continue
        seen: set[tuple[int, ...]] = set()
        for i in range(0, len(token_ids) - n + 1):
            gram = tuple(int(t) for t in token_ids[i : i + n])
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _continuation_overlap_f1(pred_text: str, target_text: str) -> float:
    pred_words = Counter(w.lower() for w in ENGLISH_WORD_RE.findall(str(pred_text)))
    target_words = Counter(w.lower() for w in ENGLISH_WORD_RE.findall(str(target_text)))
    if not pred_words and not target_words:
        return 1.0
    if not pred_words or not target_words:
        return 0.0
    overlap = sum(min(pred_words[w], target_words[w]) for w in pred_words.keys() & target_words.keys())
    pred_total = sum(pred_words.values())
    target_total = sum(target_words.values())
    precision = overlap / max(1, pred_total)
    recall = overlap / max(1, target_total)
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _english_fluency_score(text: str, contaminated: bool, looped: bool) -> float:
    stripped = str(text).strip()
    if not stripped:
        return 0.0
    words = ENGLISH_WORD_RE.findall(stripped)
    rough_tokens = re.findall(r"[A-Za-z0-9_<>\-]+", stripped)
    alpha_ratio = float(len(words)) / max(1, len(rough_tokens))
    score = 0.6 * min(1.0, alpha_ratio)
    if not contaminated:
        score += 0.25
    if not looped:
        score += 0.15
    return float(max(0.0, min(1.0, score)))


def _residual_continuation_ce(
    model,
    tokenizer,
    ctx: dict[str, torch.Tensor],
    delta: torch.Tensor,
    target_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not target_ids:
        zero = torch.zeros((), device=ctx["h_base"].device, dtype=ctx["h_base"].dtype)
        return zero, zero

    losses: list[torch.Tensor] = []
    first_logit = model.lm_head(ctx["h_base"] + delta)[:, -1, :]
    first_target = torch.tensor([int(target_ids[0])], device=first_logit.device, dtype=torch.long)
    losses.append(torch.nn.functional.cross_entropy(first_logit, first_target))
    last_logits = first_logit

    if len(target_ids) > 1:
        prefix_ids = ctx["prefix_ids"]
        for step in range(1, len(target_ids)):
            prev = torch.tensor(target_ids[:step], device=prefix_ids.device, dtype=prefix_ids.dtype).unsqueeze(0)
            seq_ids = torch.cat([prefix_ids, prev], dim=1)
            out = model(
                input_ids=seq_ids,
                attention_mask=torch.ones_like(seq_ids),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            step_hidden = out.hidden_states[-1][:, -1:, :]
            step_logits = model.lm_head(step_hidden + delta.to(device=step_hidden.device, dtype=step_hidden.dtype))[:, -1, :]
            step_target = torch.tensor([int(target_ids[step])], device=step_logits.device, dtype=torch.long)
            losses.append(torch.nn.functional.cross_entropy(step_logits, step_target))
            last_logits = step_logits

    return torch.stack(losses).mean(), last_logits


def _generate_short_resumption(
    cell: str,
    bridge: DecoderReentryResumeBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = str(item["prompt"])
    with torch.no_grad():
        ctx = _extract_reentry_context(model, tokenizer, prompt, int(args.layer_index))
        advisor_states, _advisor_ids, _token_ids = _build_base_advisor_state(
            model,
            tokenizer,
            codebook,
            arity_head,
            prompt,
            int(args.relation_vocab),
            int(args.var_min_id),
            int(args.max_logic_new_tokens),
        )
        if cell == "A":
            logits = model(input_ids=ctx["prefix_ids"], attention_mask=ctx["attention_mask"], use_cache=False, return_dict=True).logits[:, -1, :]
        elif cell == "B":
            return_tokens, _stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, 1)
            full_embs = torch.cat([ctx["prefix_embs"], return_tokens.to(device=ctx["prefix_embs"].device, dtype=ctx["prefix_embs"].dtype)], dim=1)
            full_am = torch.ones((1, full_embs.shape[1]), dtype=torch.long, device=full_embs.device)
            logits = model(inputs_embeds=full_embs, attention_mask=full_am, use_cache=False, return_dict=True).logits[:, -1, :]
        elif cell == "C":
            return_tokens, _stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, int(args.num_return_tokens))
            full_embs = torch.cat([ctx["prefix_embs"], return_tokens.to(device=ctx["prefix_embs"].device, dtype=ctx["prefix_embs"].dtype)], dim=1)
            full_am = torch.ones((1, full_embs.shape[1]), dtype=torch.long, device=full_embs.device)
            logits = model(inputs_embeds=full_embs, attention_mask=full_am, use_cache=False, return_dict=True).logits[:, -1, :]
        elif cell == "D":
            delta, _stats = bridge.residual_delta(ctx["prefix_hidden"], advisor_states)
            logits = model.lm_head(ctx["h_base"] + delta)[:, -1, :]
        elif cell == "E":
            return_tokens, delta, _stats = bridge.hybrid_resume(
                ctx["prefix_hidden"],
                advisor_states,
                int(args.num_return_tokens),
                token_scale=float(args.hybrid_token_scale),
            )
            full_embs = torch.cat([ctx["prefix_embs"], return_tokens.to(device=ctx["prefix_embs"].device, dtype=ctx["prefix_embs"].dtype)], dim=1)
            full_am = torch.ones((1, full_embs.shape[1]), dtype=torch.long, device=full_embs.device)
            out = model(inputs_embeds=full_embs, attention_mask=full_am, output_hidden_states=True, use_cache=False, return_dict=True)
            logits = model.lm_head(out.hidden_states[-1][:, -1:, :] + delta.to(device=full_embs.device, dtype=out.hidden_states[-1].dtype))[:, -1, :]
        else:
            raise ValueError(f"Unsupported cell {cell!r}")

        first_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids: list[int] = [int(first_token.item())]
        cur_ids = torch.cat([ctx["prefix_ids"], first_token.to(device=ctx["prefix_ids"].device, dtype=ctx["prefix_ids"].dtype)], dim=1)
        max_new_tokens = max(1, int(args.continuation_eval_tokens))
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=cur_ids, attention_mask=torch.ones_like(cur_ids), use_cache=False, return_dict=True)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_tok.item()))
            cur_ids = torch.cat([cur_ids, next_tok.to(device=cur_ids.device, dtype=cur_ids.dtype)], dim=1)
            if tokenizer.eos_token_id is not None and int(next_tok.item()) == int(tokenizer.eos_token_id):
                break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    target_text = _resume_target_text(item)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids[: max(1, int(args.continuation_eval_tokens))]
    target_first = int(target_ids[0]) if target_ids else None
    first_match = bool(target_first is not None and int(generated_ids[0]) == int(target_first))
    contaminated = _contains_contamination(generated_text)
    looped = _loop_flag(generated_ids)
    target_norm = re.sub(r"[^a-z0-9]+", "", target_text.lower())
    pred_norm = re.sub(r"[^a-z0-9]+", "", generated_text.lower())
    gold_mentioned = bool(target_norm) and target_norm in pred_norm
    exact_match = bool(target_norm) and pred_norm == target_norm
    fluency = _english_fluency_score(generated_text, contaminated, looped)
    return {
        "item_id": item.get("item_id", ""),
        "pair_id": item.get("pair_id", ""),
        "prompt": prompt,
        "target_text": target_text,
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "resume_first_token_accuracy": float(1.0 if first_match else 0.0),
        "continuation_overlap_f1": float(_continuation_overlap_f1(generated_text, target_text)),
        "gold_mention_rate": float(1.0 if gold_mentioned else 0.0),
        "exact_match_rate": float(1.0 if exact_match else 0.0),
        "contamination_rate": float(1.0 if contaminated else 0.0),
        "loop_rate": float(1.0 if looped else 0.0),
        "english_stays_english_rate": float(1.0 if (not contaminated and fluency >= 0.65) else 0.0),
        "english_fluency_score": float(fluency),
    }


def _evaluate_continuation_cell(
    cell: str,
    bridge: DecoderReentryResumeBridge,
    model,
    tokenizer,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    eval_pack: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = [_generate_short_resumption(cell, bridge, model, tokenizer, codebook, arity_head, item, args) for item in eval_pack]
    n = max(1, len(rows))
    metrics = {
        "english_fluency_score": float(sum(float(r["english_fluency_score"]) for r in rows) / n),
        "contamination_rate": float(sum(float(r["contamination_rate"]) for r in rows) / n),
        "loop_rate": float(sum(float(r["loop_rate"]) for r in rows) / n),
        "continuation_overlap_f1": float(sum(float(r["continuation_overlap_f1"]) for r in rows) / n),
        "gold_mention_rate": float(sum(float(r["gold_mention_rate"]) for r in rows) / n),
        "exact_match_rate": float(sum(float(r["exact_match_rate"]) for r in rows) / n),
        "english_stays_english_rate": float(sum(float(r["english_stays_english_rate"]) for r in rows) / n),
        "resume_first_token_accuracy": float(sum(float(r["resume_first_token_accuracy"]) for r in rows) / n),
    }
    return metrics, rows


def _score_candidate_from_hybrid(model, tokenizer, candidate: str, prefix_embs: torch.Tensor, return_tokens: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    from run_m3_15d_answer_path_forcing import _candidate_first_token_id  # type: ignore

    tok_id = _candidate_first_token_id(tokenizer, candidate)
    full_embs = torch.cat([prefix_embs, return_tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)], dim=1)
    full_am = torch.ones((1, full_embs.shape[1]), dtype=torch.long, device=full_embs.device)
    out = model(inputs_embeds=full_embs, attention_mask=full_am, output_hidden_states=True, use_cache=False, return_dict=True)
    h = out.hidden_states[-1][:, -1:, :] + delta.to(device=full_embs.device, dtype=out.hidden_states[-1].dtype)
    logits = model.lm_head(h)
    return torch.log_softmax(logits[:, -1, :], dim=-1)[0, int(tok_id)]


def _train_cell(cell: str, bridge: DecoderReentryResumeBridge, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, train_pack: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float]:
    if cell == "A":
        return {
            "answer_path_loss": 0.0,
            "continuation_ce_loss": 0.0,
            "answer_delta": 0.0,
            "return_norm": 0.0,
            "residual_norm": 0.0,
            "residual_guard_overflow_rate": 0.0,
            "gate": 0.0,
            "attn_entropy": 0.0,
            "train_objective": "none",
        }

    opt = torch.optim.AdamW(bridge.parameters(), lr=float(args.lr), weight_decay=0.01)
    loss_hist: list[float] = []
    ce_hist: list[float] = []
    delta_hist: list[float] = []
    return_norm_hist: list[float] = []
    residual_norm_hist: list[float] = []
    overflow_hist: list[float] = []
    gate_hist: list[float] = []
    entropy_hist: list[float] = []

    for step in range(int(args.train_steps)):
        item = train_pack[step % len(train_pack)]
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])
        target_ids = _resume_target_ids(tokenizer, item, int(args.continuation_target_max_tokens))

        with torch.no_grad():
            ctx = _extract_reentry_context(model, tokenizer, prompt, int(args.layer_index))
            advisor_states, _advisor_ids, _token_ids = _build_base_advisor_state(
                model,
                tokenizer,
                codebook,
                arity_head,
                prompt,
                int(args.relation_vocab),
                int(args.var_min_id),
                int(args.max_logic_new_tokens),
            )

        opt.zero_grad()
        residual_norm = torch.tensor(0.0, device=ctx["h_base"].device, dtype=ctx["h_base"].dtype)
        if cell == "B":
            return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, 1)
            logp_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
            logp_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
            reg = torch.norm(return_tokens, dim=-1).mean()
            residual_norm = torch.tensor(0.0, device=reg.device, dtype=reg.dtype)
        elif cell == "C":
            return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, int(args.num_return_tokens))
            logp_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
            logp_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
            reg = torch.norm(return_tokens, dim=-1).mean()
            residual_norm = torch.tensor(0.0, device=reg.device, dtype=reg.dtype)
        elif cell == "D":
            delta, stats = bridge.residual_delta(ctx["prefix_hidden"], advisor_states)
            logp_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"] + delta)
            logp_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"] + delta)
            reg = torch.norm(delta, dim=-1).mean()
            residual_norm = reg.detach()
        elif cell == "E":
            return_tokens, delta, stats = bridge.hybrid_resume(
                ctx["prefix_hidden"],
                advisor_states,
                int(args.num_return_tokens),
                token_scale=float(args.hybrid_token_scale),
            )
            logp_gold = _score_candidate_from_hybrid(model, tokenizer, gold, ctx["prefix_embs"], return_tokens, delta)
            logp_foil = _score_candidate_from_hybrid(model, tokenizer, foil, ctx["prefix_embs"], return_tokens, delta)
            token_reg = torch.norm(return_tokens, dim=-1).mean()
            residual_reg = torch.norm(delta, dim=-1).mean()
            reg = (float(args.hybrid_token_weight) * token_reg) + (float(args.hybrid_residual_weight) * residual_reg)
            residual_norm = residual_reg.detach()
        else:
            raise ValueError(f"Unsupported cell {cell!r}")

        answer_delta = logp_gold - logp_foil
        margin = torch.tensor(float(args.margin), device=answer_delta.device, dtype=answer_delta.dtype)
        answer_loss = torch.relu(margin - answer_delta)
        continuation_ce = torch.zeros((), device=answer_delta.device, dtype=answer_delta.dtype)
        if cell == "D" and str(args.d_train_objective) == "continuation_ce" and target_ids:
            continuation_ce, _ = _residual_continuation_ce(model, tokenizer, ctx, delta, target_ids)
        residual_guard = torch.relu(reg - float(args.residual_guard_threshold))
        if cell != "D":
            residual_guard = torch.zeros_like(answer_loss)
        base_objective = continuation_ce if (cell == "D" and str(args.d_train_objective) == "continuation_ce" and target_ids) else answer_loss
        loss = (
            float(args.answer_weight) * base_objective
            + float(args.return_norm_weight) * reg
            + float(args.residual_guard_weight) * residual_guard
        )
        loss.backward()
        opt.step()

        loss_hist.append(float(answer_loss.detach().item()))
        ce_hist.append(float(continuation_ce.detach().item()))
        delta_hist.append(float(answer_delta.detach().item()))
        return_norm_hist.append(float(stats.get("combined_norm", stats.get("token_norm", 0.0))))
        residual_norm_hist.append(float(stats.get("residual_norm", float(residual_norm.item()) if torch.is_tensor(residual_norm) else 0.0)))
        overflow_hist.append(float((reg.detach() > float(args.residual_guard_threshold)).item()) if cell == "D" else 0.0)
        gate_hist.append(float(stats["gate"]))
        entropy_hist.append(float(stats["attn_entropy"]))

    return {
        "answer_path_loss": float(sum(loss_hist) / max(1, len(loss_hist))),
        "continuation_ce_loss": float(sum(ce_hist) / max(1, len(ce_hist))),
        "answer_delta": float(sum(delta_hist) / max(1, len(delta_hist))),
        "return_norm": float(sum(return_norm_hist) / max(1, len(return_norm_hist))),
        "residual_norm": float(sum(residual_norm_hist) / max(1, len(residual_norm_hist))),
        "residual_guard_overflow_rate": float(sum(overflow_hist) / max(1, len(overflow_hist))),
        "gate": float(sum(gate_hist) / max(1, len(gate_hist))),
        "attn_entropy": float(sum(entropy_hist) / max(1, len(entropy_hist))),
        "train_objective": "continuation_ce" if (cell == "D" and str(args.d_train_objective) == "continuation_ce") else "margin",
    }


def _evaluate_cell(cell: str, bridge: DecoderReentryResumeBridge, model, tokenizer, codebook: BooleanAnchorTable, arity_head: AdvisorArityHead, eval_pack: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    answer_delta_hist: list[float] = []
    return_norm_hist: list[float] = []
    residual_norm_hist: list[float] = []
    overflow_hist: list[float] = []
    gate_hist: list[float] = []
    entropy_hist: list[float] = []

    for item in eval_pack:
        prompt = str(item["prompt"])
        candidates = list(item["candidates"])
        gi = int(item["gold_index"])
        fi = 1 - gi
        gold = str(candidates[gi])
        foil = str(candidates[fi])

        with torch.no_grad():
            ctx = _extract_reentry_context(model, tokenizer, prompt, int(args.layer_index))
            advisor_states, _advisor_ids, token_ids = _build_base_advisor_state(
                model,
                tokenizer,
                codebook,
                arity_head,
                prompt,
                int(args.relation_vocab),
                int(args.var_min_id),
                int(args.max_logic_new_tokens),
            )
            base_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"])
            base_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"])
            residual_norm = 0.0

            if cell == "A":
                on_gold = base_gold
                on_foil = base_foil
                stats = {"gate": 0.0, "token_norm": 0.0, "residual_norm": 0.0, "combined_norm": 0.0, "attn_entropy": 0.0}
            elif cell == "B":
                return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, 1)
                on_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
                on_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
                stats = {"gate": stats["gate"], "token_norm": stats["token_norm"], "residual_norm": 0.0, "combined_norm": stats["token_norm"], "attn_entropy": stats["attn_entropy"]}
            elif cell == "C":
                return_tokens, stats = bridge.return_tokens(ctx["prefix_hidden"], advisor_states, int(args.num_return_tokens))
                on_gold = _score_candidate_from_return_tokens(model, tokenizer, gold, ctx["prefix_embs"], return_tokens)
                on_foil = _score_candidate_from_return_tokens(model, tokenizer, foil, ctx["prefix_embs"], return_tokens)
                stats = {"gate": stats["gate"], "token_norm": stats["token_norm"], "residual_norm": 0.0, "combined_norm": stats["token_norm"], "attn_entropy": stats["attn_entropy"]}
            elif cell == "D":
                delta, stats = bridge.residual_delta(ctx["prefix_hidden"], advisor_states)
                on_gold = _score_candidate_from_hidden(model, tokenizer, gold, ctx["h_base"] + delta)
                on_foil = _score_candidate_from_hidden(model, tokenizer, foil, ctx["h_base"] + delta)
                residual_norm = float(torch.norm(delta, dim=-1).mean().item())
                stats = {"gate": stats["gate"], "token_norm": stats["token_norm"], "residual_norm": residual_norm, "combined_norm": stats["token_norm"], "attn_entropy": stats["attn_entropy"]}
            elif cell == "E":
                return_tokens, delta, stats = bridge.hybrid_resume(
                    ctx["prefix_hidden"],
                    advisor_states,
                    int(args.num_return_tokens),
                    token_scale=float(args.hybrid_token_scale),
                )
                on_gold = _score_candidate_from_hybrid(model, tokenizer, gold, ctx["prefix_embs"], return_tokens, delta)
                on_foil = _score_candidate_from_hybrid(model, tokenizer, foil, ctx["prefix_embs"], return_tokens, delta)
            else:
                raise ValueError(f"Unsupported cell {cell!r}")

        pred_idx = gi if float(on_gold.item()) >= float(on_foil.item()) else fi
        pred = str(candidates[pred_idx])
        correct = _answer_match(gold, pred)
        scope = compute_scope_violation_components(build_scope_tokens_from_triples(_triples(token_ids), var_prefix="VAR"))
        advisor = _advisor_stats(token_ids)
        answer_delta = float((on_gold - on_foil).item())
        intervention = float((on_gold - base_gold).item())

        answer_delta_hist.append(answer_delta)
        deltas.append(intervention)
        return_norm_hist.append(float(stats.get("combined_norm", stats.get("token_norm", 0.0))))
        residual_norm_hist.append(float(stats.get("residual_norm", 0.0)))
        overflow_hist.append(float(stats.get("residual_norm", 0.0) > float(args.residual_guard_threshold)))
        gate_hist.append(float(stats["gate"]))
        entropy_hist.append(float(stats["attn_entropy"]))

        rows.append({
            "item_id": item.get("item_id", ""),
            "pair_id": item.get("pair_id", ""),
            "variant_id": _variant_id_int(item),
            "family": item.get("family", "other"),
            "prompt": prompt,
            "candidates": candidates,
            "gold_answer": gold,
            "model_answer": pred,
            "correct": bool(correct),
            "active_token_count": int(advisor["active_token_count"]),
            "active_op_count": int(advisor["active_op_count"]),
            "operator_entropy": float(advisor["operator_entropy"]),
            "operator_top1_share": float(advisor["operator_top1_share"]),
            "scope": float(scope.get("scope_total", 1.0)),
            "score_on_gold": float(on_gold.item()),
            "score_on_foil": float(on_foil.item()),
            "score_off_gold": float(base_gold.item()),
            "answer_delta": float(answer_delta),
            "gold_delta_on_off": float(intervention),
            "return_norm": float(stats.get("combined_norm", stats.get("token_norm", 0.0))),
            "residual_norm": float(stats.get("residual_norm", 0.0)),
            "return_gate": float(stats["gate"]),
            "return_attn_entropy": float(stats["attn_entropy"]),
            "cell_mode": {
                "A": "control_no_advisor",
                "B": "frozen_single_return_token",
                "C": "frozen_multi_return_tokens",
                "D": "learned_residual_vector",
                "E": "hybrid_token_plus_residual",
            }[cell],
        })

    n = max(1, len(rows))
    fam_adj = [r for r in rows if str(r["family"]) == "adjective_property"]
    fam_cau = [r for r in rows if str(r["family"]) == "causal_direction"]
    metrics = {
        "overall_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
        "adjective_accuracy": float(sum(1 for r in fam_adj if r["correct"]) / max(1, len(fam_adj))),
        "causal_accuracy": float(sum(1 for r in fam_cau if r["correct"]) / max(1, len(fam_cau))),
        "mean_answer_delta": float(sum(answer_delta_hist) / max(1, len(answer_delta_hist))),
        "mean_active_tokens": float(sum(float(r["active_token_count"]) for r in rows) / n),
        "mean_active_op_count": float(sum(float(r["active_op_count"]) for r in rows) / n),
        "mean_operator_entropy": float(sum(float(r["operator_entropy"]) for r in rows) / n),
        "mean_top1_op_share": float(sum(float(r["operator_top1_share"]) for r in rows) / n),
        "mean_scope": float(sum(float(r["scope"]) for r in rows) / n),
        "mean_intervention_delta_gold": float(sum(deltas) / max(1, len(deltas))),
        "mean_return_norm": float(sum(return_norm_hist) / max(1, len(return_norm_hist))),
        "mean_residual_norm": float(sum(residual_norm_hist) / max(1, len(residual_norm_hist))),
        "residual_guard_overflow_rate": float(sum(overflow_hist) / max(1, len(overflow_hist))),
        "mean_return_gate": float(sum(gate_hist) / max(1, len(gate_hist))),
        "mean_return_attn_entropy": float(sum(entropy_hist) / max(1, len(entropy_hist))),
        "first_token_accuracy": float(sum(1 for r in rows if r["correct"]) / n),
    }
    return metrics, rows


def _promotion_gates(report: dict[str, Any], min_acc_gain: float, scope_tol: float, intervention_min: float) -> dict[str, Any]:
    a = report["cells"]["A"]["metrics"]
    gates: dict[str, Any] = {}
    for cell in ("B", "C", "D", "E"):
        m = report["cells"][cell]["metrics"]
        seed2 = report["cells"].get(f"{cell}_seed2_eval", {}).get("metrics", {})
        gates[cell] = {
            "accuracy_up": float(m["overall_accuracy"]) >= float(a["overall_accuracy"]) + float(min_acc_gain),
            "no_scope_regression": float(m["mean_scope"]) <= float(a["mean_scope"]) + float(scope_tol),
            "positive_intervention_delta": float(m["mean_intervention_delta_gold"]) >= float(intervention_min),
            "fluency_preserved": float(m.get("english_fluency_score", 0.0)) >= float(a.get("english_fluency_score", 0.0)) - 0.05,
            "contamination_below_threshold": float(m.get("contamination_rate", 1.0)) <= float(a.get("contamination_rate", 0.0)) + 0.05,
            "loop_rate_below_threshold": float(m.get("loop_rate", 1.0)) <= float(a.get("loop_rate", 0.0)) + 0.05,
            "first_token_preserved": float(m.get("resume_first_token_accuracy", 0.0)) >= float(a.get("resume_first_token_accuracy", 0.0)) - 0.05,
            "seed_stability": abs(float(seed2.get("overall_accuracy", m["overall_accuracy"])) - float(m["overall_accuracy"])) <= 0.05,
        }
        gates[cell]["promote_to_next"] = all(bool(v) for v in gates[cell].values())
    return gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.18 Decoder Re-entry Resume: advisor reasoning with compressed decoder-native resumption channels.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--num-return-tokens", type=int, default=3)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--return-norm-weight", type=float, default=0.01)
    p.add_argument("--d-train-objective", choices=("margin", "continuation_ce"), default="continuation_ce")
    p.add_argument("--continuation-target-max-tokens", type=int, default=5)
    p.add_argument("--residual-guard-threshold", type=float, default=0.01)
    p.add_argument("--residual-guard-weight", type=float, default=5.0)
    p.add_argument("--hybrid-token-weight", type=float, default=0.2)
    p.add_argument("--hybrid-residual-weight", type=float, default=1.0)
    p.add_argument("--hybrid-token-scale", type=float, default=0.15)
    p.add_argument("--continuation-eval-tokens", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-manifest", type=Path, default=Path(REENTRY_REBOOT_REGISTRY["M3.18"]["baseline_manifest"]))
    p.add_argument("--upstream-bridge-suite", type=Path, default=None)
    p.add_argument("--upstream-m11-manifest", type=Path, default=None)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path(REENTRY_REBOOT_REGISTRY["M3.18"]["output_root"]))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    assert_output_path_allowed("M", args.output_root)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    upstream_bridge_suite = args.upstream_bridge_suite
    if upstream_bridge_suite is None:
        upstream_bridge_suite = _latest_named_file(Path("artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite"), "m_bridge_ablation_suite_manifest.json")

    upstream_m11_manifest = args.upstream_m11_manifest
    if upstream_m11_manifest is None:
        suite_root = Path(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["search_root"])
        upstream_m11_manifest = _latest_named_file(suite_root, str(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["manifest_name"]))
        if upstream_m11_manifest is None:
            fallback = Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json")
            upstream_m11_manifest = fallback if fallback.exists() else None

    adapter_path = Path(args.adapter)
    adapter_has_tokenizer = (adapter_path / "tokenizer.json").exists() or (adapter_path / "tokenizer_config.json").exists()
    tok_src = str(adapter_path) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(adapter_path), local_files_only=args.local_files_only)
    target_device = torch.device("cuda" if torch.cuda.is_available() else _model_device(model))
    model = model.to(target_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ckpt = torch.load(args.checkpoint, map_location=target_device)
    hidden = int(model.config.hidden_size)
    module_dtype = next(model.parameters()).dtype
    codebook = BooleanAnchorTable(2000, hidden).to(target_device, dtype=module_dtype)
    codebook.load_state_dict(ckpt["codebook_state"])
    codebook.eval()
    for param in codebook.parameters():
        param.requires_grad = False
    arity_head = AdvisorArityHead(hidden, 2000).to(target_device, dtype=module_dtype)
    arity_head.load_state_dict(ckpt["arity_head_state"], strict=False)
    arity_head.eval()
    for param in arity_head.parameters():
        param.requires_grad = False

    if args.pack_jsonl is None:
        pack_train, pack_val, pack_eval, split_meta = _build_winograd_split_packs(size=int(args.eval_size), seed=int(args.seed), strict_balance=bool(args.strict_balance))
        _t2, _v2, pack_eval_seed2, split_meta_seed2 = _build_winograd_split_packs(size=int(args.eval_size), seed=int(args.seed) + 1, strict_balance=bool(args.strict_balance))
    else:
        base_pack = _load_pack(args.pack_jsonl, size=int(args.eval_size) * 3, seed=int(args.seed), strict_balance=bool(args.strict_balance))
        pair_ids = sorted({str(r.get("pair_id", "")) for r in base_pack if str(r.get("pair_id", "")).strip()})
        rng = random.Random(int(args.seed))
        rng.shuffle(pair_ids)
        eval_count = max(1, round(len(pair_ids) * 0.30))
        val_count = max(1, round(len(pair_ids) * 0.15))
        train_ids = set(pair_ids[: max(1, len(pair_ids) - eval_count - val_count)])
        val_ids = set(pair_ids[max(1, len(pair_ids) - eval_count - val_count) : max(1, len(pair_ids) - eval_count)])
        eval_ids = set(pair_ids[max(1, len(pair_ids) - eval_count) :])
        if not eval_ids:
            eval_ids = set(pair_ids[-1:])
        if not val_ids:
            val_ids = set(pair_ids[-2:-1] or pair_ids[-1:])
        pack_train = [r for r in base_pack if str(r.get("pair_id", "")) in train_ids][: int(args.eval_size)]
        pack_val = [r for r in base_pack if str(r.get("pair_id", "")) in val_ids][: int(args.eval_size)]
        pack_eval = [r for r in base_pack if str(r.get("pair_id", "")) in eval_ids][: int(args.eval_size)]
        pack_eval_seed2 = _build_winograd_pack(size=int(args.eval_size), seed=int(args.seed) + 2, strict_balance=bool(args.strict_balance))
        split_meta = {"train_pair_ids": sorted(train_ids), "val_pair_ids": sorted(val_ids), "eval_pair_ids": sorted(eval_ids)}
        split_meta_seed2 = {"seed": int(args.seed) + 1}

    (run_dir / "m3_18_eval_pack_preview.json").write_text(json.dumps(pack_eval[:20], indent=2), encoding="utf-8")

    report: dict[str, Any] = build_reentry_protocol_manifest(
        track="M3.18",
        run_id=run_id,
        baseline_manifest_path=args.baseline_manifest,
        baseline_id=str(baseline_manifest.get("baseline_id", "")),
        upstream_bridge_suite=str(upstream_bridge_suite).replace("\\", "/") if upstream_bridge_suite else None,
        upstream_m11_manifest=str(upstream_m11_manifest).replace("\\", "/") if upstream_m11_manifest else None,
        config={k: str(v) for k, v in vars(args).items()},
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["series"] = series_metadata("M", "M3.18", "scripts/run_m3_18_decoder_reentry_resume.py")
    report["lineage"] = lineage_metadata("train", checkpoint_in=str(args.checkpoint).replace("\\", "/"), checkpoint_out=None, dataset_profile="winograd_family_split", difficulty_tier="mixed")
    report["data_split"] = {**split_meta, "seed2_eval_meta": split_meta_seed2, "runtime_policy_source": "validation_metrics", "final_metrics_source": "eval_pack"}
    report["cells"] = {}

    def make_bridge() -> DecoderReentryResumeBridge:
        mod = DecoderReentryResumeBridge(hidden, bottleneck_dim=int(args.bottleneck_dim), max_return_tokens=int(args.num_return_tokens)).to(target_device, dtype=module_dtype)
        mod.eval()
        return mod

    bridge_a = make_bridge()
    train_a = _train_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_train, args)
    met_a_val, _ = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_val, args)
    met_a, rows_a = _evaluate_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_eval, args)
    cont_a, cont_rows_a = _evaluate_continuation_cell("A", bridge_a, model, tokenizer, codebook, arity_head, pack_eval, args)
    met_a.update(cont_a)
    report["cells"]["A"] = {"train": train_a, "metrics": met_a, "validation_metrics": met_a_val, "variant_spec": REENTRY_REBOOT_REGISTRY["M3.18"]["cells"]["A"]}
    (run_dir / "m3_18_A_eval.json").write_text(json.dumps(rows_a, indent=2), encoding="utf-8")
    (run_dir / "m3_18_A_continuation_eval.json").write_text(json.dumps(cont_rows_a, indent=2), encoding="utf-8")

    for cell in ("B", "C", "D", "E"):
        bridge = make_bridge()
        train_cell = _train_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_train, args)
        met_val, _ = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_val, args)
        met_eval, rows_eval = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_eval, args)
        cont_eval, cont_rows_eval = _evaluate_continuation_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_eval, args)
        met_eval.update(cont_eval)
        met_seed2, rows_seed2 = _evaluate_cell(cell, bridge, model, tokenizer, codebook, arity_head, pack_eval_seed2, args)
        report["cells"][cell] = {"train": train_cell, "metrics": met_eval, "validation_metrics": met_val, "variant_spec": REENTRY_REBOOT_REGISTRY["M3.18"]["cells"][cell]}
        report["cells"][f"{cell}_seed2_eval"] = {"metrics": met_seed2}
        (run_dir / f"m3_18_{cell}_eval.json").write_text(json.dumps(rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m3_18_{cell}_continuation_eval.json").write_text(json.dumps(cont_rows_eval, indent=2), encoding="utf-8")
        (run_dir / f"m3_18_{cell}_seed2_eval.json").write_text(json.dumps(rows_seed2, indent=2), encoding="utf-8")

    report["promotion_gates"] = _promotion_gates(report, min_acc_gain=0.02, scope_tol=0.02, intervention_min=0.01)
    (run_dir / "m3_18_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = reentry_cell_labels("M3.18")
    md = [
        "# M3.18 Decoder Re-entry Resume",
        "",
        "| Cell | Regime | Acc | FTok | Fluency | Contam | Loop | Mention | Answer Delta | Gold On-Off | Return Norm | Residual Norm | Scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("A", "B", "C", "D", "E"):
        m = report["cells"][c]["metrics"]
        md.append(f"| {c} | {labels[c]} | {m['overall_accuracy']:.3f} | {m.get('resume_first_token_accuracy', 0.0):.3f} | {m.get('english_fluency_score', 0.0):.3f} | {m.get('contamination_rate', 0.0):.3f} | {m.get('loop_rate', 0.0):.3f} | {m.get('gold_mention_rate', 0.0):.3f} | {m['mean_answer_delta']:.4f} | {m['mean_intervention_delta_gold']:.4f} | {m['mean_return_norm']:.4f} | {m.get('mean_residual_norm', 0.0):.4f} | {m['mean_scope']:.4f} |")
    md.extend([
        "",
        "## Regimes",
        "- B: one-shot single return token before answer continuation.",
        "- C: one-shot short return-token bundle before answer continuation.",
        "- D: one-shot residual continuation vector.",
        "- E: hybrid token bundle plus residual continuation vector.",
        "",
        "## Continuation Metrics",
        "- `FTok`: first-token correctness under one-shot decoder resumption.",
        "- `Fluency`: rough English-likeness after re-entry.",
        "- `Contam`: Lojbanic or sidecar-bleed rate in the resumed text.",
        "- `Loop`: short repetition/degeneracy rate in resumed text.",
        "- `Mention`: whether the resumed continuation explicitly mentions the gold answer.",
        "",
        "## Promotion Gates",
    ])
    for cell in ("B", "C", "D", "E"):
        md.append(f"- {cell}:")
        for key, value in report["promotion_gates"][cell].items():
            md.append(f"  - {key}: `{value}`")
    (run_dir / "m3_18_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"M3.18 complete: {run_dir}")


if __name__ == "__main__":
    main()
