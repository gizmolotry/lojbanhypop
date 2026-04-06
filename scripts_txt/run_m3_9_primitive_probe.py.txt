from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from lojban_evolution.experiment import generate_dataset, split_dataset
from lojban_evolution.l_series import (
    build_scope_tokens_from_triples,
    compute_arity_violation,
    compute_scope_violation_components,
)
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)
from train_h5_persistent_vq_advisor import (  # type: ignore
    AdvisorArityHead,
    BooleanAnchorTable,
    CouncilCrossAttentionAdapter,
    adapter_disabled,
    build_final_prefix,
    extract_trace_hidden_states,
    persistent_advisor_hook,
)


def _infer_family(prompt: str) -> str:
    p = prompt.lower()
    if "knight" in p or "knave" in p:
        return "knights_knaves"
    if "where does" in p and "think" in p:
        return "multi_agent"
    if "too big" in p or "too small" in p or "demonstrators" in p or "councilmen" in p:
        return "winograd"
    return "other"


def _entropy(dist: list[float]) -> float:
    e = 0.0
    for p in dist:
        if p > 0:
            e -= float(p) * math.log(float(p))
    return e


def _latest_m3_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("m3 report must be JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.9 Emergent Grammar / Primitive Discovery Probe (evaluation-only).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--dataset-size", type=int, default=120)
    p.add_argument("--dataset-profile", type=str, default="diverse_v2")
    p.add_argument(
        "--difficulty-tier",
        choices=("all", "easy", "medium", "hard"),
        default="all",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-answer-tokens", type=int, default=12)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--candidate-topk", type=int, default=12)
    p.add_argument("--causal-samples", type=int, default=12)
    p.add_argument("--clusters", type=int, default=5)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe"))
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def _decode_tokens(
    model,
    tokenizer,
    prompt: str,
    codebook: BooleanAnchorTable,
    arity_head: AdvisorArityHead,
    max_logic_new_tokens: int,
    relation_vocab: int,
    var_min_id: int,
) -> list[int]:
    with torch.no_grad():
        h_t = extract_trace_hidden_states(model, tokenizer, prompt, int(max_logic_new_tokens)).to(model.dtype)
        z_st, _idx, _cb, _commit = codebook.quantize(h_t)
    out: list[int] = []
    for i in range(z_st.shape[1]):
        z = z_st[:, i, :]
        l_rel = arity_head.head_rel(z)
        mask_rel = torch.full_like(l_rel, -1e9)
        mask_rel[:, : int(relation_vocab)] = 0.0
        t_rel = int(torch.argmax(l_rel + mask_rel, dim=-1)[0].item())

        l_v1 = arity_head.head_var1(z)
        mask_v = torch.full_like(l_v1, -1e9)
        mask_v[:, int(var_min_id) :] = 0.0
        t_v1 = int(torch.argmax(l_v1 + mask_v, dim=-1)[0].item())

        l_v2 = arity_head.head_var2(z)
        t_v2 = int(torch.argmax(l_v2 + mask_v, dim=-1)[0].item())
        out.extend([t_rel, t_v1, t_v2])
    return out


def _token_ids_to_state(token_ids: list[int], codebook: BooleanAnchorTable) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(token_ids, device=codebook.emb.device, dtype=torch.long).view(1, -1)
    state = codebook.emb[ids].to(dtype=codebook.emb.dtype)
    return state, ids


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
) -> float:
    p_ids = tokenizer(build_final_prefix(prompt), return_tensors="pt").input_ids.to(model.device)
    t_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[:, :max_answer_tokens]
    if advisor_ids.shape[1] <= 0:
        return 0.0
    with adapter_disabled(model):
        ce = torch.zeros((), device=model.device, dtype=model.dtype)
        cur_emb = model.get_input_embeddings()(p_ids)
        ptr = 0
        for t in range(t_ids.shape[1]):
            ptr_ids = torch.full((1, cur_emb.shape[1]), ptr, device=model.device, dtype=torch.long)
            with persistent_advisor_hook(model, layer_index, advisor_adapter, advisor_state, advisor_ids, ptr_ids, 1.0):
                out = model(inputs_embeds=cur_emb, use_cache=False)
            ce = ce + F.cross_entropy(out.logits[:, -1, :], t_ids[:, t])
            cur_emb = torch.cat([cur_emb, model.get_input_embeddings()(t_ids[:, t : t + 1])], dim=1)
            ptr = min(ptr + 1, max(0, advisor_ids.shape[1] - 1))
    return float(ce.detach().item())


def _triples(token_ids: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for i in range(0, len(token_ids) - 2, 3):
        out.append((int(token_ids[i]), int(token_ids[i + 1]), int(token_ids[i + 2])))
    return out


def _evaluate_trace(
    model,
    tokenizer,
    advisor_adapter,
    codebook,
    prompt: str,
    answer: str,
    token_ids: list[int],
    layer_index: int,
    max_answer_tokens: int,
    relation_vocab: int,
    var_min_id: int,
) -> dict[str, float]:
    if not token_ids:
        return {"ce_loss": 0.0, "arity_violation": 1.0, "scope_total": 1.0}
    triples = _triples(token_ids)
    scope = compute_scope_violation_components(build_scope_tokens_from_triples(triples, var_prefix="VAR"))
    arity = compute_arity_violation(triples, relation_vocab=int(relation_vocab), var_min_id=int(var_min_id))
    adv_state, adv_ids = _token_ids_to_state(token_ids, codebook)
    ce = _teacher_forced_ce(
        model=model,
        tokenizer=tokenizer,
        advisor_adapter=advisor_adapter,
        advisor_state=adv_state,
        advisor_ids=adv_ids,
        prompt=prompt,
        answer=answer,
        layer_index=int(layer_index),
        max_answer_tokens=int(max_answer_tokens),
    )
    return {"ce_loss": float(ce), "arity_violation": float(arity), "scope_total": float(scope["scope_total"])}


def _kmeans(features: torch.Tensor, k: int, steps: int = 20) -> tuple[list[int], torch.Tensor]:
    n = features.shape[0]
    if n == 0:
        return [], torch.empty((0, features.shape[1]), device=features.device, dtype=features.dtype)
    k = max(1, min(int(k), int(n)))
    centroids = features[:k].clone()
    assigns = torch.zeros((n,), device=features.device, dtype=torch.long)
    for _ in range(int(steps)):
        d = torch.cdist(features, centroids)
        assigns = torch.argmin(d, dim=1)
        for j in range(k):
            mask = assigns == j
            if torch.any(mask):
                centroids[j] = torch.mean(features[mask], dim=0)
    return [int(x) for x in assigns.detach().cpu().tolist()], centroids


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    assert_output_path_allowed("M", args.output_root)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    codebook = BooleanAnchorTable(2000, hidden_size).to(model.device, dtype=model.dtype)
    advisor_adapter = CouncilCrossAttentionAdapter(hidden_size).to(model.device, dtype=model.dtype)
    arity_head = AdvisorArityHead(hidden_size, 2000).to(model.device, dtype=model.dtype)
    ckpt = torch.load(args.checkpoint, map_location=model.device)
    codebook.load_state_dict(ckpt["codebook_state"])
    advisor_adapter.load_state_dict(ckpt["advisor_adapter_state"], strict=False)
    cs, ps = arity_head.state_dict(), ckpt["arity_head_state"]
    for n, p in ps.items():
        if n in cs and cs[n].shape == p.shape:
            cs[n].copy_(p)
    arity_head.load_state_dict(cs)
    codebook.eval()
    advisor_adapter.eval()
    arity_head.eval()

    ds = generate_dataset(
        size=int(args.dataset_size),
        seed=int(args.seed),
        profile=str(args.dataset_profile),
        difficulty_tier=str(args.difficulty_tier),
    )
    _, _, test = split_dataset(ds)
    samples = [{"prompt": str(x.prompt), "answer": str(x.answer), "family": _infer_family(str(x.prompt))} for x in test]

    traces: list[dict[str, Any]] = []
    token_freq: Counter[int] = Counter()
    role_counts: dict[int, Counter[str]] = defaultdict(Counter)
    domain_counts: dict[int, Counter[str]] = defaultdict(Counter)
    neighbors: dict[int, Counter[int]] = defaultdict(Counter)
    baseline_metrics: list[dict[str, float]] = []

    for s in samples:
        token_ids = _decode_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=s["prompt"],
            codebook=codebook,
            arity_head=arity_head,
            max_logic_new_tokens=int(args.max_logic_new_tokens),
            relation_vocab=int(args.relation_vocab),
            var_min_id=int(args.var_min_id),
        )
        m = _evaluate_trace(
            model=model,
            tokenizer=tokenizer,
            advisor_adapter=advisor_adapter,
            codebook=codebook,
            prompt=s["prompt"],
            answer=s["answer"],
            token_ids=token_ids,
            layer_index=int(args.layer_index),
            max_answer_tokens=int(args.max_answer_tokens),
            relation_vocab=int(args.relation_vocab),
            var_min_id=int(args.var_min_id),
        )
        baseline_metrics.append(m)
        traces.append({"prompt": s["prompt"], "answer": s["answer"], "family": s["family"], "token_ids": token_ids, "baseline": m})
        for i, tok in enumerate(token_ids):
            t = int(tok)
            token_freq[t] += 1
            domain_counts[t][s["family"]] += 1
            if i == 0:
                role_counts[t]["start_of_trace"] += 1
            if i == len(token_ids) - 1:
                role_counts[t]["end_of_trace"] += 1
            mod = i % 3
            if mod == 0:
                role_counts[t]["predicate_head"] += 1
            elif mod == 1:
                role_counts[t]["arg_slot_1"] += 1
            else:
                role_counts[t]["arg_slot_2"] += 1
            if i > 0:
                neighbors[t][int(token_ids[i - 1])] += 1
            if i + 1 < len(token_ids):
                neighbors[t][int(token_ids[i + 1])] += 1

    total_tokens = float(sum(token_freq.values())) if token_freq else 1.0
    primitive_token_report: dict[str, Any] = {}
    token_role_matrix: dict[str, Any] = {}
    token_position_entropy: dict[str, float] = {}

    for tok, cnt in token_freq.items():
        rc = role_counts[tok]
        dc = domain_counts[tok]
        role_keys = ["start_of_trace", "end_of_trace", "predicate_head", "arg_slot_1", "arg_slot_2"]
        rv = [float(rc.get(k, 0)) for k in role_keys]
        rsum = sum(rv) if sum(rv) > 0 else 1.0
        rp = [x / rsum for x in rv]
        dsum = float(sum(dc.values())) if dc else 1.0
        dom = {k: float(v) / dsum for k, v in dc.items()}
        primitive_token_report[str(tok)] = {
            "token_id": int(tok),
            "frequency": int(cnt),
            "frequency_share": float(cnt) / total_tokens,
            "role_distribution": {k: rp[i] for i, k in enumerate(role_keys)},
            "domain_distribution": dom,
            "top_neighbors": [{"token_id": int(n), "count": int(c)} for n, c in neighbors[tok].most_common(8)],
        }
        token_role_matrix[str(tok)] = {k: rp[i] for i, k in enumerate(role_keys)}
        token_position_entropy[str(tok)] = float(_entropy(rp))

    # Stage 2 causal perturbation.
    candidates = [int(t) for t, _ in token_freq.most_common(int(args.candidate_topk))]
    freq_map = {int(t): int(c) for t, c in token_freq.items()}
    substitute_for: dict[int, int] = {}
    toks = list(freq_map.keys())
    for t in candidates:
        target = freq_map[t]
        best = None
        best_d = 10**9
        for u in toks:
            if u == t:
                continue
            d = abs(freq_map[u] - target)
            if d < best_d:
                best_d = d
                best = u
        substitute_for[t] = int(best) if best is not None else int(t)

    causal_rows: dict[str, Any] = {}
    causal_subset = traces[: max(1, min(int(args.causal_samples), len(traces)))]
    for tok in candidates:
        deltas = {"ce": [], "arity": [], "scope": []}
        deltas_sub = {"ce": [], "arity": [], "scope": []}
        deltas_swap = {"ce": [], "arity": [], "scope": []}
        hits = 0
        for tr in causal_subset:
            ids = [int(x) for x in tr["token_ids"]]
            if tok not in ids:
                continue
            hits += 1
            base = tr["baseline"]

            # deletion
            ids_del = [x for x in ids if x != tok]
            if len(ids_del) < 3:
                ids_del = ids[:3]
            m_del = _evaluate_trace(
                model=model,
                tokenizer=tokenizer,
                advisor_adapter=advisor_adapter,
                codebook=codebook,
                prompt=tr["prompt"],
                answer=tr["answer"],
                token_ids=ids_del,
                layer_index=int(args.layer_index),
                max_answer_tokens=int(args.max_answer_tokens),
                relation_vocab=int(args.relation_vocab),
                var_min_id=int(args.var_min_id),
            )
            deltas["ce"].append(m_del["ce_loss"] - base["ce_loss"])
            deltas["arity"].append(m_del["arity_violation"] - base["arity_violation"])
            deltas["scope"].append(m_del["scope_total"] - base["scope_total"])

            # substitution
            sub = int(substitute_for[tok])
            ids_sub = [sub if x == tok else x for x in ids]
            m_sub = _evaluate_trace(
                model=model,
                tokenizer=tokenizer,
                advisor_adapter=advisor_adapter,
                codebook=codebook,
                prompt=tr["prompt"],
                answer=tr["answer"],
                token_ids=ids_sub,
                layer_index=int(args.layer_index),
                max_answer_tokens=int(args.max_answer_tokens),
                relation_vocab=int(args.relation_vocab),
                var_min_id=int(args.var_min_id),
            )
            deltas_sub["ce"].append(m_sub["ce_loss"] - base["ce_loss"])
            deltas_sub["arity"].append(m_sub["arity_violation"] - base["arity_violation"])
            deltas_sub["scope"].append(m_sub["scope_total"] - base["scope_total"])

            # neighbor swap
            ids_sw = ids[:]
            for i in range(len(ids_sw) - 1):
                if ids_sw[i] == tok:
                    ids_sw[i], ids_sw[i + 1] = ids_sw[i + 1], ids_sw[i]
                    break
            m_sw = _evaluate_trace(
                model=model,
                tokenizer=tokenizer,
                advisor_adapter=advisor_adapter,
                codebook=codebook,
                prompt=tr["prompt"],
                answer=tr["answer"],
                token_ids=ids_sw,
                layer_index=int(args.layer_index),
                max_answer_tokens=int(args.max_answer_tokens),
                relation_vocab=int(args.relation_vocab),
                var_min_id=int(args.var_min_id),
            )
            deltas_swap["ce"].append(m_sw["ce_loss"] - base["ce_loss"])
            deltas_swap["arity"].append(m_sw["arity_violation"] - base["arity_violation"])
            deltas_swap["scope"].append(m_sw["scope_total"] - base["scope_total"])

        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        causal_rows[str(tok)] = {
            "token_id": int(tok),
            "samples_with_token": int(hits),
            "deletion": {
                "delta_ce_loss": _avg(deltas["ce"]),
                "delta_arity_violation": _avg(deltas["arity"]),
                "delta_scope_total": _avg(deltas["scope"]),
            },
            "substitution": {
                "token_substitute": int(substitute_for[tok]),
                "delta_ce_loss": _avg(deltas_sub["ce"]),
                "delta_arity_violation": _avg(deltas_sub["arity"]),
                "delta_scope_total": _avg(deltas_sub["scope"]),
            },
            "neighbor_swap": {
                "delta_ce_loss": _avg(deltas_swap["ce"]),
                "delta_arity_violation": _avg(deltas_swap["arity"]),
                "delta_scope_total": _avg(deltas_swap["scope"]),
            },
        }

    # Stage 3: clustering.
    feature_tokens = sorted(token_freq.keys())
    families = sorted({tr["family"] for tr in traces})
    role_keys = ["start_of_trace", "end_of_trace", "predicate_head", "arg_slot_1", "arg_slot_2"]
    feats = []
    for t in feature_tokens:
        rd = token_role_matrix[str(t)]
        f = [float(rd.get(k, 0.0)) for k in role_keys]
        dd = primitive_token_report[str(t)]["domain_distribution"]
        f.extend([float(dd.get(k, 0.0)) for k in families])
        f.append(float(token_position_entropy[str(t)]))
        f.append(float(primitive_token_report[str(t)]["frequency_share"]))
        feats.append(f)
    feature_tensor = torch.tensor(feats, device=model.device, dtype=torch.float32) if feats else torch.empty((0, 1), device=model.device)
    assignments, _centroids = _kmeans(feature_tensor, int(args.clusters))
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(assignments):
        cluster_members[int(c)].append(int(feature_tokens[i]))

    grammatical_cluster_report: dict[str, Any] = {}
    for cid, members in cluster_members.items():
        dom_role = "mixed"
        role_totals = Counter()
        for t in members:
            rd = token_role_matrix[str(t)]
            for rk, rv in rd.items():
                role_totals[rk] += float(rv)
        if role_totals:
            dom_role = role_totals.most_common(1)[0][0]
        if dom_role == "predicate_head":
            label = "predicate_heads"
        elif dom_role in {"arg_slot_1", "arg_slot_2"}:
            label = "referential_markers"
        elif dom_role in {"start_of_trace", "end_of_trace"}:
            label = "boundary_markers"
        else:
            label = "relation_modifiers"
        grammatical_cluster_report[str(cid)] = {
            "cluster_id": int(cid),
            "label": label,
            "member_tokens": [int(x) for x in members],
            "member_count": int(len(members)),
            "dominant_role": dom_role,
        }

    # Primitive candidates.
    candidate_rows = []
    for tok in candidates:
        t = str(tok)
        ent = float(token_position_entropy.get(t, 0.0))
        c = causal_rows.get(t, {})
        d = c.get("deletion", {})
        s = c.get("substitution", {})
        w = c.get("neighbor_swap", {})
        causal_score = (
            abs(float(d.get("delta_ce_loss", 0.0)))
            + abs(float(d.get("delta_scope_total", 0.0)))
            + abs(float(d.get("delta_arity_violation", 0.0)))
            + abs(float(s.get("delta_ce_loss", 0.0)))
            + abs(float(w.get("delta_ce_loss", 0.0)))
        ) / 5.0
        specialization = 1.0 - (ent / max(math.log(5.0), 1e-8))
        score = (0.5 * causal_score) + (0.5 * specialization)
        candidate_rows.append(
            {
                "token_id": int(tok),
                "score": float(score),
                "specialization": float(specialization),
                "causal_score": float(causal_score),
                "position_entropy": float(ent),
                "samples_with_token": int(c.get("samples_with_token", 0)),
                "deletion": d,
                "substitution": s,
                "neighbor_swap": w,
            }
        )
    candidate_rows.sort(key=lambda x: x["score"], reverse=True)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)

    primitive_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_9_primitive_probe",
        "series": series_metadata("M", "M3.9", "scripts/run_m3_9_primitive_probe.py"),
        "track": "M3.9",
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=str(args.checkpoint).replace("\\", "/"),
            checkpoint_out=None,
            dataset_profile=str(args.dataset_profile),
            difficulty_tier=str(args.difficulty_tier),
        ),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "checkpoint": str(args.checkpoint).replace("\\", "/"),
            "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
            "baseline_id": str(baseline_manifest.get("baseline_id", "")),
            "dataset_size": int(args.dataset_size),
            "dataset_profile": str(args.dataset_profile),
            "difficulty_tier": str(args.difficulty_tier),
            "candidate_topk": int(args.candidate_topk),
            "causal_samples": int(args.causal_samples),
        },
        "summary": {
            "active_token_count": int(len(token_freq)),
            "primitive_candidate_count": int(len(candidate_rows)),
            "mean_baseline_ce_loss": float(sum(x["ce_loss"] for x in baseline_metrics) / max(1, len(baseline_metrics))),
            "mean_baseline_scope": float(sum(x["scope_total"] for x in baseline_metrics) / max(1, len(baseline_metrics))),
            "mean_baseline_arity": float(sum(x["arity_violation"] for x in baseline_metrics) / max(1, len(baseline_metrics))),
        },
    }

    (out_dir / "primitive_token_report.json").write_text(json.dumps(primitive_token_report, indent=2), encoding="utf-8")
    (out_dir / "token_role_matrix.json").write_text(json.dumps(token_role_matrix, indent=2), encoding="utf-8")
    (out_dir / "token_position_entropy.json").write_text(json.dumps(token_position_entropy, indent=2), encoding="utf-8")
    (out_dir / "grammatical_cluster_report.json").write_text(json.dumps(grammatical_cluster_report, indent=2), encoding="utf-8")
    (out_dir / "primitive_candidate_list.json").write_text(json.dumps(candidate_rows, indent=2), encoding="utf-8")
    (out_dir / "m3_9_primitive_probe_report.json").write_text(json.dumps(primitive_report, indent=2), encoding="utf-8")

    md_lines = [
        "# M3.9 Primitive Probe Report",
        "",
        f"- run_id: `{run_id}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- active_token_count: `{primitive_report['summary']['active_token_count']}`",
        f"- primitive_candidate_count: `{primitive_report['summary']['primitive_candidate_count']}`",
        f"- mean_baseline_ce_loss: `{primitive_report['summary']['mean_baseline_ce_loss']:.6f}`",
        f"- mean_baseline_scope: `{primitive_report['summary']['mean_baseline_scope']:.6f}`",
        "",
        "## Top Primitive Candidates",
        "",
        "| token_id | score | specialization | causal_score | position_entropy | samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows[:10]:
        md_lines.append(
            f"| `{row['token_id']}` | {row['score']:.6f} | {row['specialization']:.6f} | "
            f"{row['causal_score']:.6f} | {row['position_entropy']:.6f} | {row['samples_with_token']} |"
        )
    (out_dir / "m3_9_primitive_probe_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {out_dir / 'primitive_token_report.json'}")
    print(f"Wrote: {out_dir / 'token_role_matrix.json'}")
    print(f"Wrote: {out_dir / 'token_position_entropy.json'}")
    print(f"Wrote: {out_dir / 'grammatical_cluster_report.json'}")
    print(f"Wrote: {out_dir / 'primitive_candidate_list.json'}")
    print(f"Wrote: {out_dir / 'm3_9_primitive_probe_report.json'}")
    print(f"Wrote: {out_dir / 'm3_9_primitive_probe_report.md'}")


if __name__ == "__main__":
    main()
