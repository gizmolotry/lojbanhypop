from __future__ import annotations

import argparse
import json
import math
import random
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
    build_scope_tokens_from_events,
    build_scope_tokens_from_triples,
    compute_arity_violation,
    compute_identity_violation_from_ce,
    compute_scope_violation_rate,
    compute_scope_violation_components,
    crispness_loss,
    entropy_floor_penalty,
    lukasiewicz_and,
    lukasiewicz_or,
    make_swap_variant,
    overflow_penalty,
    parse_relation_events_from_sequence,
    RelationEvent,
    rolling_all_below,
    generate_scope_minimal_pair_samples,
    infer_swap_semantics,
)
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata

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
    stage_mode: str
    task_loss: float
    constraint_arity: float
    constraint_arity_strict: float
    constraint_scope: float
    constraint_scope_unbalanced: float
    constraint_scope_lifetime: float
    constraint_scope_unbound: float
    constraint_scope_quantifier_assoc: float
    constraint_scope_shadowing: float
    constraint_identity: float
    arity_crystallization_rate: float
    arity_mean_entropy: float
    arity_mean_mode_share: float
    swap_policy: str
    swap_semantics: str
    swap_active: bool
    lambda_arity: float
    lambda_scope: float
    lambda_identity: float
    total_loss: float
    tier_b_enabled: bool
    tier_c_enabled: bool
    shadow_loss: float
    shadow_align_loss: float
    shadow_separate_loss: float
    shadow_temporal_loss: float
    shadow_pos_similarity: float
    shadow_neg_similarity: float
    shadow_english_pos_similarity: float
    shadow_english_neg_similarity: float
    diversification_mode: str
    diversification_loss: float
    diversification_entropy_loss: float
    diversification_domain_reuse_loss: float
    diversification_family_cluster_loss: float
    operator_entropy: float
    operator_top1_share: float
    source: str


def _safe_item(x: torch.Tensor) -> float:
    return float(x.detach().item())


def _decode_triples(tokens: Sequence[torch.Tensor]) -> List[Tuple[int, int, int]]:
    flat = [int(t[0].detach().item()) for t in tokens]
    triples: List[Tuple[int, int, int]] = []
    for i in range(0, len(flat) - 2, 3):
        triples.append((flat[i], flat[i + 1], flat[i + 2]))
    return triples


def _flatten_token_ids(tokens: Sequence[torch.Tensor]) -> List[int]:
    return [int(t[0].detach().item()) for t in tokens]


def _parse_relation_events_observed(
    token_ids: Sequence[int],
    relation_vocab: int,
    var_min_id: int,
) -> List[RelationEvent]:
    events: List[RelationEvent] = []
    ids = [int(x) for x in token_ids]
    i = 0
    while i < len(ids):
        rel = ids[i]
        if rel < 0 or rel >= int(relation_vocab):
            i += 1
            continue
        j = i + 1
        args: List[int] = []
        while j < len(ids):
            tok = ids[j]
            if 0 <= int(tok) < int(relation_vocab):
                break
            if int(tok) >= int(var_min_id):
                args.append(int(tok))
            j += 1
        events.append(RelationEvent(rel=int(rel), args=tuple(args)))
        i = max(j, i + 1)
    return events


def _compute_crystallization_metrics(
    arity_history: Dict[int, List[int]],
    threshold: float,
    min_events: int,
) -> Tuple[float, float, float]:
    entropies: List[float] = []
    mode_shares: List[float] = []
    crystallized = 0
    eligible = 0
    for _rel, hist in arity_history.items():
        if len(hist) < int(min_events):
            continue
        eligible += 1
        counts: Dict[int, int] = {}
        for n in hist:
            counts[int(n)] = counts.get(int(n), 0) + 1
        total = float(sum(counts.values()))
        probs = [float(c) / total for c in counts.values() if c > 0]
        entropy = 0.0
        for p in probs:
            entropy -= p * float(math.log(max(p, 1e-12)))
        mode_share = max(probs) if probs else 0.0
        entropies.append(float(entropy))
        mode_shares.append(float(mode_share))
        if mode_share >= float(threshold):
            crystallized += 1
    if eligible == 0:
        return 0.0, 0.0, 0.0
    return float(crystallized) / float(eligible), float(sum(entropies) / len(entropies)), float(sum(mode_shares) / len(mode_shares))


def _compute_per_op_arity_metrics(
    arity_history: Dict[int, List[int]],
    threshold: float,
    min_events: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool], float]:
    mode_share_by_op: Dict[str, float] = {}
    entropy_by_op: Dict[str, float] = {}
    crystallized_by_op: Dict[str, bool] = {}
    eligible = 0
    crystallized = 0
    for rel, hist in arity_history.items():
        if len(hist) < int(min_events):
            continue
        counts: Dict[int, int] = {}
        for n in hist:
            counts[int(n)] = counts.get(int(n), 0) + 1
        total = float(sum(counts.values()))
        probs = [float(c) / total for c in counts.values() if c > 0]
        if not probs:
            continue
        entropy = 0.0
        for p in probs:
            entropy -= p * float(math.log(max(p, 1e-12)))
        mode_share = max(probs)
        key = str(int(rel))
        mode_share_by_op[key] = float(mode_share)
        entropy_by_op[key] = float(entropy)
        crystallized_flag = bool(mode_share >= float(threshold))
        crystallized_by_op[key] = crystallized_flag
        eligible += 1
        if crystallized_flag:
            crystallized += 1
    rate = float(crystallized) / float(eligible) if eligible > 0 else 0.0
    return mode_share_by_op, entropy_by_op, crystallized_by_op, rate


def _advisor_state_from_tokens(tokens: Sequence[torch.Tensor], arity_head: AdvisorArityHead, codebook: BooleanAnchorTable) -> Tuple[torch.Tensor, torch.Tensor]:
    adv_state = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
    adv_ids = torch.stack(tokens, dim=1)
    return adv_state, adv_ids


def _masked_logits(logits: torch.Tensor, legal_start: int, legal_end: int, bias: float) -> torch.Tensor:
    out = logits.clone()
    if bias == 0.0:
        return out
    if legal_start > 0:
        out[:, :legal_start] = out[:, :legal_start] + float(bias)
    if legal_end < out.shape[-1]:
        out[:, legal_end:] = out[:, legal_end:] + float(bias)
    return out


def _decode_with_arity_bias(
    arity_head: AdvisorArityHead,
    latent: torch.Tensor,
    relation_vocab: int,
    var_min_id: int,
    mask_bias: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    b, l, _h = latent.shape
    all_tokens: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    deficits: List[torch.Tensor] = []

    for i in range(l):
        z = latent[:, i, :]

        l_rel_raw = arity_head.head_rel(z)
        l_rel = _masked_logits(l_rel_raw, 0, relation_vocab, mask_bias)
        p_rel = torch.softmax(l_rel, dim=-1)
        rel_legal_mass = torch.sum(p_rel[:, :relation_vocab], dim=-1)
        deficits.append(1.0 - torch.mean(rel_legal_mass))
        t_rel = torch.argmax(l_rel, dim=-1)

        l_v1_raw = arity_head.head_var1(z)
        l_v1 = _masked_logits(l_v1_raw, var_min_id, l_v1_raw.shape[-1], mask_bias)
        p_v1 = torch.softmax(l_v1, dim=-1)
        v1_legal_mass = torch.sum(p_v1[:, var_min_id:], dim=-1)
        deficits.append(1.0 - torch.mean(v1_legal_mass))
        t_v1 = torch.argmax(l_v1, dim=-1)

        l_v2_raw = arity_head.head_var2(z)
        l_v2 = _masked_logits(l_v2_raw, var_min_id, l_v2_raw.shape[-1], mask_bias)
        p_v2 = torch.softmax(l_v2, dim=-1)
        v2_legal_mass = torch.sum(p_v2[:, var_min_id:], dim=-1)
        deficits.append(1.0 - torch.mean(v2_legal_mass))
        t_v2 = torch.argmax(l_v2, dim=-1)

        all_tokens.extend([t_rel, t_v1, t_v2])
        all_logits.extend([l_rel, l_v1, l_v2])

    arity_proxy = torch.mean(torch.stack(deficits)) if deficits else torch.zeros((), device=latent.device)
    return all_tokens, all_logits, arity_proxy


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


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return (1.0 - t) * float(a) + t * float(b)


def _infer_relation_family(prompt: str) -> str:
    p = prompt.lower()
    if "knight" in p or "knave" in p:
        return "knights_knaves"
    if "where does" in p and "think" in p:
        return "multi_agent"
    if "too big" in p or "too small" in p or "demonstrators" in p or "councilmen" in p:
        return "winograd"
    if "for all" in p or "there exists" in p or "exists" in p:
        return "scope_logic"
    return "other"


def _simple_paraphrase(prompt: str) -> str:
    swaps = [
        ("because", "since"),
        ("Who", "Which person"),
        ("Where does", "In which place does"),
        (" does not ", " doesn't "),
        (" while ", " as "),
    ]
    out = str(prompt)
    for a, b in swaps:
        if a in out:
            out = out.replace(a, b)
    return out


def _english_semantic_embedding(model, tokenizer, prompt: str, max_tokens: int = 128) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_tokens)).input_ids.to(model.device)
    with adapter_disabled(model):
        with torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
    h = out.hidden_states[-1]  # [1,T,H]
    emb = torch.mean(h, dim=1)  # [1,H]
    return F.normalize(emb, p=2, dim=-1)


def _advisor_signature_from_logits(logits: Sequence[torch.Tensor], relation_vocab: int) -> torch.Tensor:
    rel_logits = torch.stack(logits[0::3], dim=1)  # [B,L,V]
    rel_probs = torch.softmax(rel_logits, dim=-1)[..., : int(relation_vocab)]
    sig = torch.mean(rel_probs, dim=1)  # [B,relation_vocab]
    return F.normalize(sig, p=2, dim=-1)


def _operator_distribution_from_logits(logits: Sequence[torch.Tensor], relation_vocab: int) -> torch.Tensor:
    rel_logits = torch.stack(logits[0::3], dim=1)
    rel_probs = torch.softmax(rel_logits, dim=-1)[..., : int(relation_vocab)]
    dist = torch.mean(rel_probs, dim=(0, 1))
    dist = dist / torch.clamp(torch.sum(dist), min=1e-8)
    return dist


def _encode_shadow_signature_for_prompt(
    model,
    tokenizer,
    prompt: str,
    max_logic_new_tokens: int,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    relation_vocab: int,
    var_min_id: int,
    mask_bias: float,
) -> torch.Tensor:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(max_logic_new_tokens)).to(model.dtype)
    z_st, _idx, _cb_loss, _commit_loss = codebook.quantize(h_t)
    _tokens, logits, _proxy = _decode_with_arity_bias(
        arity_head=arity_head,
        latent=z_st,
        relation_vocab=int(relation_vocab),
        var_min_id=int(var_min_id),
        mask_bias=float(mask_bias),
    )
    return _advisor_signature_from_logits(logits, relation_vocab=int(relation_vocab))


def _invert_binary_answer(answer: str) -> Optional[str]:
    a = str(answer).strip()
    u = a.upper()
    pairs = {
        "TRUE": "FALSE",
        "FALSE": "TRUE",
        "YES": "NO",
        "NO": "YES",
        "A_EQ_B": "A_NEQ_B",
        "A_NEQ_B": "A_EQ_B",
    }
    if u in pairs:
        out = pairs[u]
        return out if a.isupper() else out.lower()
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L-Series MVS trainer (Lexicographic Augmented Lagrangian).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--dataset-profile", type=str, default="diverse_v2", help="Dataset profile for logic corpus generation.")
    p.add_argument(
        "--difficulty-tier",
        choices=("all", "easy", "medium", "hard"),
        default="all",
        help="Difficulty split for corpus generation.",
    )
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
    p.add_argument(
        "--swap-policy",
        choices=("forced_asymmetric", "symmetry_aware", "disabled"),
        default="forced_asymmetric",
        help="Swap-test semantics policy for identity/truth pressure.",
    )
    p.add_argument("--swap-loss-weight", type=float, default=1.0)
    p.add_argument("--output-root", type=Path, default=Path("runs/l_series"))
    p.add_argument("--scope-minimal-pairs", type=int, default=0, help="Inject N scope minimal-pair curriculum samples.")
    p.add_argument("--scope-curriculum-ratio", type=float, default=0.5, help="Probability of sampling scope curriculum each step.")
    p.add_argument("--force-tier-b-after", type=int, default=-1, help="Force-enable Tier B at this step index (0-based).")
    p.add_argument("--force-tier-c-after", type=int, default=-1, help="Force-enable Tier C at this step index (0-based).")

    # Three-stage arity shaping controls.
    p.add_argument("--stage0-steps", type=int, default=40, help="Teacher-forced syntax bootstrap window.")
    p.add_argument("--stage1-steps", type=int, default=120, help="Soft-mask annealing duration after stage0.")
    p.add_argument("--soft-mask-start", type=float, default=-20.0)
    p.add_argument("--soft-mask-mid", type=float, default=-5.0)
    p.add_argument("--soft-mask-end", type=float, default=0.0)
    p.add_argument("--arity-feasible-threshold", type=float, default=0.05)
    p.add_argument("--arity-feasible-window", type=int, default=16)
    p.add_argument("--arity-proxy-weight", type=float, default=2.0)
    p.add_argument(
        "--operator-arity-json",
        type=Path,
        default=None,
        help="Optional JSON map {relation_token_id: arity_n} for dynamic signature enforcement.",
    )
    p.add_argument(
        "--require-observed-registry-provenance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When true, operator_arity_json must be envelope format with provenance='observed_usage'.",
    )
    p.add_argument("--default-relation-arity", type=int, default=2)
    p.add_argument(
        "--dynamic-arity-signatures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic n-ary arity parser/signature enforcement. Off by default for backward compatibility.",
    )
    p.add_argument(
        "--arity-enforcement-mode",
        choices=("legacy_strict", "registry_strict", "crystallization"),
        default="legacy_strict",
        help="Arity handling mode: strict legacy triples, strict registry signatures, or crystallization metrics only.",
    )
    p.add_argument("--arity-crystallization-window", type=int, default=128)
    p.add_argument("--arity-crystallization-threshold", type=float, default=0.9)
    p.add_argument("--arity-crystallization-min-events", type=int, default=8)
    p.add_argument(
        "--shadow-mode",
        choices=("off", "paraphrase", "family", "rolling"),
        default="off",
        help="M3.7 advisor shadowing regime: off(A), paraphrase(B), family/rolling(C).",
    )
    p.add_argument("--shadow-align-weight", type=float, default=0.0)
    p.add_argument("--shadow-separate-weight", type=float, default=0.0)
    p.add_argument("--shadow-temporal-weight", type=float, default=0.0)
    p.add_argument("--shadow-margin", type=float, default=0.10)
    p.add_argument(
        "--diversification-mode",
        choices=("off", "entropy", "domain_reuse", "family_cluster"),
        default="off",
        help="M3.8 diversification pressure mode.",
    )
    p.add_argument("--diversification-weight", type=float, default=0.0)
    p.add_argument("--diversification-domain-overlap-target", type=float, default=0.45)
    p.add_argument("--diversification-top1-penalty", type=float, default=0.25)
    p.add_argument("--diversification-cluster-centroids", type=int, default=3)
    p.add_argument("--diversification-cluster-margin", type=float, default=0.80)

    p.add_argument("--use-iron-collar", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("L", args.output_root)
    operator_arity_registry: Dict[int, int] = {}
    if args.operator_arity_json is not None:
        raw = json.loads(args.operator_arity_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("operator_arity_json must be a JSON object.")
        provenance = ""
        reg_raw = raw
        if "registry" in raw:
            reg_raw = raw.get("registry", {})
            provenance = str(raw.get("provenance", "")).strip()
        if bool(args.require_observed_registry_provenance) and provenance != "observed_usage":
            raise ValueError(
                "operator_arity_json must declare provenance='observed_usage' when strict registry enforcement is used."
            )
        if not isinstance(reg_raw, dict):
            raise ValueError("operator_arity_json.registry must be a JSON object mapping relation token id -> arity.")
        for k, v in reg_raw.items():
            operator_arity_registry[int(k)] = max(1, int(v))

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
    # Match adapter tokenizer vocab before loading PEFT weights.
    backbone.resize_token_embeddings(len(tokenizer))
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

    ds = generate_dataset(
        size=int(args.dataset_size),
        seed=int(args.seed),
        profile=str(args.dataset_profile),
        difficulty_tier=str(args.difficulty_tier),
    )
    train, val, test = split_dataset(ds)
    train_pool = list(train) if train else list(val) if val else list(test)
    if not train_pool:
        raise ValueError("No training examples available after dataset split.")
    scope_curriculum = generate_scope_minimal_pair_samples(int(args.scope_minimal_pairs), seed=int(args.seed)) if int(args.scope_minimal_pairs) > 0 else []

    controller = AugmentedLagrangianController(
        rho=float(args.rho),
        constraints=("arity", "scope", "identity"),
        init_lambda=float(args.init_lambda),
        max_lambda=float(args.max_lambda),
    )

    recent_tier_a: List[Dict[str, float]] = []
    recent_arity_strict: List[float] = []
    tier_b_enabled = False
    tier_c_enabled = False

    telemetry: List[StepTelemetry] = []
    arity_history: Dict[int, List[int]] = {}
    swap_counts: Dict[str, int] = {"invariant": 0, "foil": 0, "disabled": 0, "none": 0}
    swap_active_count = 0
    shadow_rng = random.Random(int(args.seed) + 1337)
    shadow_prev_english: Optional[torch.Tensor] = None
    shadow_prev_advisor: Optional[torch.Tensor] = None
    family_to_prompts: Dict[str, List[str]] = {}
    for pb in train_pool:
        fam = _infer_relation_family(str(pb.prompt))
        family_to_prompts.setdefault(fam, []).append(str(pb.prompt))
    family_op_prototypes: Dict[str, torch.Tensor] = {}
    family_op_counts: Dict[str, int] = {}
    cluster_k = max(1, int(args.diversification_cluster_centroids))
    cluster_template: List[torch.Tensor] = []
    basis = torch.eye(5, device=model.device, dtype=model.dtype)
    for i in range(cluster_k):
        if i < 5:
            v = basis[i]
        else:
            v = torch.rand((5,), device=model.device, dtype=model.dtype)
        cluster_template.append(F.normalize(v, p=2, dim=-1))
    family_cluster_banks: Dict[str, List[torch.Tensor]] = {}

    for step in range(int(args.train_steps)):
        use_scope_item = bool(scope_curriculum) and (float(torch.rand((), device="cpu").item()) < float(args.scope_curriculum_ratio))
        if use_scope_item:
            s = scope_curriculum[step % len(scope_curriculum)]
            prompt = str(s["prompt"])
            answer = str(s["answer"])
            source = "scope_curriculum"
        else:
            item = train_pool[step % len(train_pool)]
            prompt = str(item.prompt)
            answer = str(item.answer)
            source = "base_dataset"

        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(args.max_logic_new_tokens)).to(model.dtype)

        z_st, _idx, cb_loss, commit_loss = codebook.quantize(h_t)

        stage0 = step < int(args.stage0_steps)
        arity_feasible = False
        if recent_arity_strict and len(recent_arity_strict) >= int(args.arity_feasible_window):
            recent_tail = recent_arity_strict[-int(args.arity_feasible_window) :]
            arity_feasible = all(x <= float(args.arity_feasible_threshold) for x in recent_tail)

        if stage0:
            stage_mode = "stage0_hard_syntax"
            mask_bias = -1e9
        elif arity_feasible:
            stage_mode = "stage2_hard_feasible"
            mask_bias = -1e9
        else:
            stage_mode = "stage1_soft_anneal"
            prog = (step - int(args.stage0_steps)) / max(1.0, float(args.stage1_steps))
            if prog < 0.5:
                mask_bias = _lerp(float(args.soft_mask_start), float(args.soft_mask_mid), prog / 0.5)
            else:
                mask_bias = _lerp(float(args.soft_mask_mid), float(args.soft_mask_end), (prog - 0.5) / 0.5)

        tokens, logits, c_arity_proxy = _decode_with_arity_bias(
            arity_head=arity_head,
            latent=z_st,
            relation_vocab=5,
            var_min_id=5,
            mask_bias=float(mask_bias),
        )
        adv_state, adv_ids = _advisor_state_from_tokens(tokens, arity_head, codebook)
        advisor_signature = _advisor_signature_from_logits(logits, relation_vocab=5)
        operator_dist = _operator_distribution_from_logits(logits, relation_vocab=5)

        triples = _decode_triples(tokens)
        flat_token_ids = _flatten_token_ids(tokens)
        dynamic_arity_enabled = bool(args.dynamic_arity_signatures) or bool(operator_arity_registry)
        if str(args.arity_enforcement_mode) == "legacy_strict":
            c_arity_strict_val = compute_arity_violation(triples, relation_vocab=5, var_min_id=5)
            scope_tokens = build_scope_tokens_from_triples(triples, var_prefix="VAR")
            events = [RelationEvent(rel=int(r), args=(int(v1), int(v2))) for (r, v1, v2) in triples]
        elif str(args.arity_enforcement_mode) == "registry_strict":
            if not dynamic_arity_enabled:
                raise ValueError("arity_enforcement_mode=registry_strict requires --dynamic-arity-signatures or --operator-arity-json")
            events, c_arity_strict_val = parse_relation_events_from_sequence(
                flat_token_ids,
                relation_vocab=5,
                var_min_id=5,
                operator_arity_registry=operator_arity_registry,
                default_relation_arity=int(args.default_relation_arity),
            )
            scope_tokens = build_scope_tokens_from_events(events, var_prefix="VAR")
        else:
            # Crystallization mode: do not enforce strict registry arity; track emergent consistency instead.
            events = _parse_relation_events_observed(flat_token_ids, relation_vocab=5, var_min_id=5)
            c_arity_strict_val = 0.0
            scope_tokens = build_scope_tokens_from_events(events, var_prefix="VAR")

        for ev in events:
            rel = int(ev.rel)
            hist = arity_history.setdefault(rel, [])
            hist.append(int(len(ev.args)))
            if len(hist) > int(args.arity_crystallization_window):
                arity_history[rel] = hist[-int(args.arity_crystallization_window) :]

        cryst_rate, arity_mean_entropy, arity_mean_mode_share = _compute_crystallization_metrics(
            arity_history=arity_history,
            threshold=float(args.arity_crystallization_threshold),
            min_events=int(args.arity_crystallization_min_events),
        )
        if str(args.arity_enforcement_mode) == "crystallization":
            c_arity_gate_val = float(c_arity_proxy.detach().item())
        else:
            c_arity_gate_val = float(c_arity_strict_val)
        recent_arity_strict.append(float(c_arity_gate_val))
        if len(recent_arity_strict) > int(args.arity_feasible_window):
            recent_arity_strict = recent_arity_strict[-int(args.arity_feasible_window) :]

        scope_components = compute_scope_violation_components(scope_tokens)
        c_scope_val = float(scope_components["scope_total"])

        if stage0:
            ce_base = torch.zeros((), device=model.device, dtype=model.dtype)
            swap_loss = torch.zeros((), device=model.device, dtype=model.dtype)
            c_identity = torch.zeros((), device=model.device, dtype=model.dtype)
            swap_semantics = "none"
            swap_active = False
        else:
            ce_base = _teacher_forced_ce(
                model=model,
                tokenizer=tokenizer,
                advisor_adapter=advisor_adapter,
                advisor_state=adv_state,
                advisor_ids=adv_ids,
                prompt=prompt,
                answer=answer,
                layer_index=int(args.layer_index),
                max_answer_tokens=int(args.max_answer_tokens),
            )

            swap_loss = torch.zeros((), device=model.device, dtype=model.dtype)
            swap_semantics = "none"
            swap_active = False
            if str(args.swap_policy) == "disabled":
                c_identity = torch.zeros((), device=model.device, dtype=model.dtype)
                swap_counts["disabled"] += 1
            else:
                swapped = make_swap_variant(prompt, answer)
                ce_swap: Optional[torch.Tensor] = None
                if swapped is not None:
                    sw_prompt, sw_answer = swapped
                    if str(args.swap_policy) == "symmetry_aware":
                        swap_semantics = infer_swap_semantics(prompt)
                    else:
                        swap_semantics = "foil"
                    if swap_semantics == "invariant":
                        swap_answer = answer
                    else:
                        inv = _invert_binary_answer(answer)
                        swap_answer = inv if inv is not None else sw_answer
                    ce_swap = _teacher_forced_ce(
                        model=model,
                        tokenizer=tokenizer,
                        advisor_adapter=advisor_adapter,
                        advisor_state=adv_state,
                        advisor_ids=adv_ids,
                        prompt=sw_prompt,
                        answer=swap_answer,
                        layer_index=int(args.layer_index),
                        max_answer_tokens=int(args.max_answer_tokens),
                    )
                    swap_loss = ce_swap
                    swap_active = True
                    swap_active_count += 1
                    swap_counts[swap_semantics] = swap_counts.get(swap_semantics, 0) + 1
                else:
                    swap_counts["none"] += 1
                if swap_semantics == "invariant":
                    c_identity = compute_identity_violation_from_ce(ce_base, ce_swap, margin=float(args.identity_margin))
                else:
                    c_identity = torch.zeros((), device=model.device, dtype=model.dtype)

        c_terms = {
            "arity": c_arity_proxy.to(dtype=model.dtype),
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
        if int(args.force_tier_b_after) >= 0 and step >= int(args.force_tier_b_after):
            tier_b_enabled = True
        if int(args.force_tier_c_after) >= 0 and step >= int(args.force_tier_c_after):
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
            active_vars = len({int(v) for ev in events for v in getattr(ev, "args", ())})
            ovf_present = any(int(tok[0].detach().item()) == int(args.ovf_token_id) for tok in tokens)
            c_ovf = torch.tensor(
                overflow_penalty(active_refs=active_vars, capacity=int(args.register_capacity), ovf_emitted=ovf_present),
                device=model.device,
                dtype=model.dtype,
            )
            tier_c_loss = c_div + c_ovf

        task_loss = ce_base + cb_loss + (0.25 * commit_loss)
        task_loss = task_loss + (float(args.swap_loss_weight) * swap_loss)
        shadow_align_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        shadow_separate_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        shadow_temporal_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        shadow_pos_sim = 0.0
        shadow_neg_sim = 0.0
        shadow_eng_pos_sim = 0.0
        shadow_eng_neg_sim = 0.0
        if (str(args.shadow_mode) != "off") and (not stage0):
            para_prompt = _simple_paraphrase(prompt)
            eng_anchor = _english_semantic_embedding(model, tokenizer, prompt)
            eng_para = _english_semantic_embedding(model, tokenizer, para_prompt)
            advisor_para = _encode_shadow_signature_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=para_prompt,
                max_logic_new_tokens=int(args.max_logic_new_tokens),
                codebook=codebook,
                arity_head=arity_head,
                relation_vocab=5,
                var_min_id=5,
                mask_bias=float(mask_bias),
            )
            pos_sim = F.cosine_similarity(advisor_signature, advisor_para, dim=-1).mean()
            eng_pos_sim = F.cosine_similarity(eng_anchor, eng_para, dim=-1).mean()
            shadow_align_loss = F.mse_loss(pos_sim, eng_pos_sim.detach())
            shadow_pos_sim = float(pos_sim.detach().item())
            shadow_eng_pos_sim = float(eng_pos_sim.detach().item())

            if str(args.shadow_mode) in {"family", "rolling"}:
                fam = _infer_relation_family(prompt)
                neg_fams = [k for k in family_to_prompts.keys() if k != fam and family_to_prompts.get(k)]
                if neg_fams:
                    neg_family = neg_fams[shadow_rng.randrange(len(neg_fams))]
                    neg_prompt = family_to_prompts[neg_family][shadow_rng.randrange(len(family_to_prompts[neg_family]))]
                    eng_neg = _english_semantic_embedding(model, tokenizer, neg_prompt)
                    advisor_neg = _encode_shadow_signature_for_prompt(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=neg_prompt,
                        max_logic_new_tokens=int(args.max_logic_new_tokens),
                        codebook=codebook,
                        arity_head=arity_head,
                        relation_vocab=5,
                        var_min_id=5,
                        mask_bias=float(mask_bias),
                    )
                    neg_sim = F.cosine_similarity(advisor_signature, advisor_neg, dim=-1).mean()
                    eng_neg_sim = F.cosine_similarity(eng_anchor, eng_neg, dim=-1).mean()
                    margin = float(args.shadow_margin)
                    shadow_separate_loss = F.relu(neg_sim - pos_sim + margin) + F.mse_loss(neg_sim, eng_neg_sim.detach())
                    shadow_neg_sim = float(neg_sim.detach().item())
                    shadow_eng_neg_sim = float(eng_neg_sim.detach().item())

            if str(args.shadow_mode) == "rolling":
                if shadow_prev_english is not None and shadow_prev_advisor is not None:
                    eng_roll_sim = F.cosine_similarity(eng_anchor, shadow_prev_english, dim=-1).mean()
                    adv_roll_sim = F.cosine_similarity(advisor_signature, shadow_prev_advisor, dim=-1).mean()
                    shadow_temporal_loss = F.mse_loss(adv_roll_sim, eng_roll_sim.detach())
                shadow_prev_english = eng_anchor.detach()
                shadow_prev_advisor = advisor_signature.detach()

        shadow_loss = (
            float(args.shadow_align_weight) * shadow_align_loss
            + float(args.shadow_separate_weight) * shadow_separate_loss
            + float(args.shadow_temporal_weight) * shadow_temporal_loss
        )
        diversification_entropy_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        diversification_domain_reuse_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        diversification_family_cluster_loss = torch.zeros((), device=model.device, dtype=model.dtype)
        p_safe = torch.clamp(operator_dist, min=1e-8)
        operator_entropy = -torch.sum(p_safe * torch.log(p_safe))
        operator_top1_share = torch.max(p_safe)
        if str(args.diversification_mode) == "entropy":
            diversification_entropy_loss = -operator_entropy
        elif str(args.diversification_mode) == "domain_reuse":
            fam = _infer_relation_family(prompt)
            proto = family_op_prototypes.get(fam)
            if proto is not None:
                intra = 1.0 - F.cosine_similarity(operator_dist.unsqueeze(0), proto.unsqueeze(0), dim=-1).mean()
            else:
                intra = torch.zeros((), device=model.device, dtype=model.dtype)
            overlaps: List[torch.Tensor] = []
            for ofam, oproto in family_op_prototypes.items():
                if ofam == fam:
                    continue
                overlaps.append(F.cosine_similarity(operator_dist.unsqueeze(0), oproto.unsqueeze(0), dim=-1).mean())
            if overlaps:
                overlap_mean = torch.mean(torch.stack(overlaps))
                overlap_target = torch.tensor(float(args.diversification_domain_overlap_target), device=model.device, dtype=model.dtype)
                overlap_loss = torch.abs(overlap_mean - overlap_target)
            else:
                overlap_loss = torch.zeros((), device=model.device, dtype=model.dtype)
            top1_pen = torch.tensor(float(args.diversification_top1_penalty), device=model.device, dtype=model.dtype) * operator_top1_share
            diversification_domain_reuse_loss = intra + overlap_loss + top1_pen
            count = int(family_op_counts.get(fam, 0))
            if proto is None:
                family_op_prototypes[fam] = operator_dist.detach()
                family_op_counts[fam] = 1
            else:
                alpha = 1.0 / float(count + 1)
                family_op_prototypes[fam] = F.normalize((1.0 - alpha) * proto + alpha * operator_dist.detach(), p=2, dim=-1)
                family_op_counts[fam] = count + 1
        elif str(args.diversification_mode) == "family_cluster":
            fam = _infer_relation_family(prompt)
            if fam not in family_cluster_banks:
                family_cluster_banks[fam] = [c.clone() for c in cluster_template]
            centroids = family_cluster_banks[fam]
            sims = [F.cosine_similarity(operator_dist.unsqueeze(0), c.unsqueeze(0), dim=-1).mean() for c in centroids]
            sim_stack = torch.stack(sims)
            best_idx = int(torch.argmax(sim_stack).item())
            cohesion = 1.0 - sim_stack[best_idx]
            sep_terms: List[torch.Tensor] = []
            margin = float(args.diversification_cluster_margin)
            for i, s in enumerate(sims):
                if i == best_idx:
                    continue
                sep_terms.append(F.relu(s - margin))
            sep = torch.mean(torch.stack(sep_terms)) if sep_terms else torch.zeros((), device=model.device, dtype=model.dtype)
            top1_pen = torch.tensor(float(args.diversification_top1_penalty), device=model.device, dtype=model.dtype) * operator_top1_share
            diversification_family_cluster_loss = cohesion + sep + top1_pen
            c_old = centroids[best_idx]
            centroids[best_idx] = F.normalize((0.9 * c_old + 0.1 * operator_dist.detach()), p=2, dim=-1)
            family_cluster_banks[fam] = centroids
        diversification_loss = (
            diversification_entropy_loss
            + diversification_domain_reuse_loss
            + diversification_family_cluster_loss
        )
        total_loss = (
            task_loss
            + tier_a_penalty
            + (float(args.arity_proxy_weight) * c_arity_proxy)
            + (float(args.weight_tier_b) * tier_b_loss)
            + (float(args.weight_tier_c) * tier_c_loss)
            + shadow_loss
            + (float(args.diversification_weight) * diversification_loss)
        )

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        codebook.enforce_anchor_values()

        controller.update({k: float(v.detach().item()) for k, v in c_terms.items()})

        telemetry.append(
            StepTelemetry(
                step=step + 1,
                stage_mode=stage_mode,
                task_loss=_safe_item(task_loss),
                constraint_arity=float(c_arity_proxy.detach().item()),
                constraint_arity_strict=float(c_arity_strict_val),
                constraint_scope=float(c_scope_val),
                constraint_scope_unbalanced=float(scope_components["scope_unbalanced"]),
                constraint_scope_lifetime=float(scope_components["scope_lifetime"]),
                constraint_scope_unbound=float(scope_components["scope_unbound"]),
                constraint_scope_quantifier_assoc=float(scope_components["scope_quantifier_assoc"]),
                constraint_scope_shadowing=float(scope_components["scope_shadowing"]),
                constraint_identity=float(c_identity.detach().item()),
                arity_crystallization_rate=float(cryst_rate),
                arity_mean_entropy=float(arity_mean_entropy),
                arity_mean_mode_share=float(arity_mean_mode_share),
                swap_policy=str(args.swap_policy),
                swap_semantics=str(swap_semantics),
                swap_active=bool(swap_active),
                lambda_arity=float(controller.lambdas["arity"]),
                lambda_scope=float(controller.lambdas["scope"]),
                lambda_identity=float(controller.lambdas["identity"]),
                total_loss=_safe_item(total_loss),
                tier_b_enabled=bool(tier_b_enabled),
                tier_c_enabled=bool(tier_c_enabled),
                shadow_loss=float(shadow_loss.detach().item()),
                shadow_align_loss=float(shadow_align_loss.detach().item()),
                shadow_separate_loss=float(shadow_separate_loss.detach().item()),
                shadow_temporal_loss=float(shadow_temporal_loss.detach().item()),
                shadow_pos_similarity=shadow_pos_sim,
                shadow_neg_similarity=shadow_neg_sim,
                shadow_english_pos_similarity=shadow_eng_pos_sim,
                shadow_english_neg_similarity=shadow_eng_neg_sim,
                diversification_mode=str(args.diversification_mode),
                diversification_loss=float(diversification_loss.detach().item()),
                diversification_entropy_loss=float(diversification_entropy_loss.detach().item()),
                diversification_domain_reuse_loss=float(diversification_domain_reuse_loss.detach().item()),
                diversification_family_cluster_loss=float(diversification_family_cluster_loss.detach().item()),
                operator_entropy=float(operator_entropy.detach().item()),
                operator_top1_share=float(operator_top1_share.detach().item()),
                source=source,
            )
        )

        if (step + 1) % 10 == 0:
            print(
                f"Step {step+1}/{args.train_steps} [{stage_mode}] | total={_safe_item(total_loss):.4f} "
                f"cA={float(c_arity_proxy.detach().item()):.3f} cA_strict={c_arity_strict_val:.3f} "
                f"cS={c_scope_val:.3f} cI={float(c_identity.detach().item()):.3f} "
                f"shadow={float(shadow_loss.detach().item()):.3f} "
                f"div={float(diversification_loss.detach().item()):.3f} "
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

    arity_mode_share_by_op, arity_entropy_by_op, arity_crystallized_by_op, crystallization_rate = _compute_per_op_arity_metrics(
        arity_history=arity_history,
        threshold=float(args.arity_crystallization_threshold),
        min_events=int(args.arity_crystallization_min_events),
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("L", "lagrangian_training", "scripts/train_l_series_mvs.py"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset_splits": {
            "train_count": int(len(train)),
            "val_count": int(len(val)),
            "test_count": int(len(test)),
            "optimization_split": "train",
            "fallback_split_used": "train" if train else "val" if val else "test",
        },
        "final_lambdas": dict(controller.lambdas),
        "final_step": asdict(telemetry[-1]) if telemetry else None,
        "arity_crystallization": {
            "window": int(args.arity_crystallization_window),
            "threshold": float(args.arity_crystallization_threshold),
            "min_events": int(args.arity_crystallization_min_events),
            "operator_histograms": {str(k): v for k, v in arity_history.items()},
            "arity_mode_share_by_op": arity_mode_share_by_op,
            "arity_entropy_by_op": arity_entropy_by_op,
            "arity_crystallized_by_op": arity_crystallized_by_op,
            "crystallization_rate": float(crystallization_rate),
        },
        "swap_metrics": {
            "swap_policy": str(args.swap_policy),
            "swap_active_count": int(swap_active_count),
            "swap_counts": swap_counts,
        },
        "shadow_metrics": {
            "shadow_mode": str(args.shadow_mode),
            "shadow_align_weight": float(args.shadow_align_weight),
            "shadow_separate_weight": float(args.shadow_separate_weight),
            "shadow_temporal_weight": float(args.shadow_temporal_weight),
            "shadow_margin": float(args.shadow_margin),
            "final_shadow_loss": float(telemetry[-1].shadow_loss if telemetry else 0.0),
            "final_shadow_align_loss": float(telemetry[-1].shadow_align_loss if telemetry else 0.0),
            "final_shadow_separate_loss": float(telemetry[-1].shadow_separate_loss if telemetry else 0.0),
            "final_shadow_temporal_loss": float(telemetry[-1].shadow_temporal_loss if telemetry else 0.0),
            "final_shadow_pos_similarity": float(telemetry[-1].shadow_pos_similarity if telemetry else 0.0),
            "final_shadow_neg_similarity": float(telemetry[-1].shadow_neg_similarity if telemetry else 0.0),
            "final_shadow_english_pos_similarity": float(telemetry[-1].shadow_english_pos_similarity if telemetry else 0.0),
            "final_shadow_english_neg_similarity": float(telemetry[-1].shadow_english_neg_similarity if telemetry else 0.0),
        },
        "diversification_metrics": {
            "diversification_mode": str(args.diversification_mode),
            "diversification_weight": float(args.diversification_weight),
            "domain_overlap_target": float(args.diversification_domain_overlap_target),
            "top1_penalty": float(args.diversification_top1_penalty),
            "cluster_centroids": int(args.diversification_cluster_centroids),
            "cluster_margin": float(args.diversification_cluster_margin),
            "final_diversification_loss": float(telemetry[-1].diversification_loss if telemetry else 0.0),
            "final_diversification_entropy_loss": float(telemetry[-1].diversification_entropy_loss if telemetry else 0.0),
            "final_diversification_domain_reuse_loss": float(telemetry[-1].diversification_domain_reuse_loss if telemetry else 0.0),
            "final_diversification_family_cluster_loss": float(telemetry[-1].diversification_family_cluster_loss if telemetry else 0.0),
            "final_operator_entropy": float(telemetry[-1].operator_entropy if telemetry else 0.0),
            "final_operator_top1_share": float(telemetry[-1].operator_top1_share if telemetry else 0.0),
        },
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
