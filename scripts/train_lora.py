from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Set, Tuple


def patch_nope_qwen2() -> bool:
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    except Exception:
        return False

    def _identity_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return q, k

    qwen2_mod.apply_rotary_pos_emb = _identity_rotary
    return True


def parse_pair_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for spec in specs:
        if ":" not in spec:
            continue
        left, right = spec.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        pairs.append((left, right))
    return pairs


def load_compositional_anchors(path: Path) -> List[Tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    if isinstance(payload, dict):
        raw = payload.get("pairs", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    for row in raw:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            left = str(row[0]).strip()
            right = str(row[1]).strip()
            if left and right:
                pairs.append((left, right))
        elif isinstance(row, dict):
            left = str(row.get("left", "")).strip()
            right = str(row.get("right", "")).strip()
            if left and right:
                pairs.append((left, right))
    out: List[Tuple[str, str]] = []
    seen = set()
    for l, r in pairs:
        key = (l, r)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_anchor_boundary(
    text: str,
    anchor: str,
    ids_len: int,
    prefix_token_len_fn,
) -> int:
    if not anchor:
        return -1
    split_idx = text.find(anchor)
    if split_idx < 0:
        return -1
    prefix = text[: split_idx + len(anchor)]
    return min(int(prefix_token_len_fn(prefix)), int(ids_len))


def compute_segment_weights(
    ids_len: int,
    prompt_w: float,
    trace_w: float,
    answer_w: float,
    trace_start: int,
    answer_start: int,
) -> List[float]:
    weights = [float(prompt_w)] * int(ids_len)
    if trace_start >= 0:
        tail_end = answer_start if answer_start >= 0 else len(weights)
        for j in range(trace_start, tail_end):
            weights[j] = float(trace_w)
    if answer_start >= 0:
        for j in range(answer_start, len(weights)):
            weights[j] = float(answer_w)
    return weights


def _resolve_single_token_ids(tokenizer, token_texts: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for token in token_texts:
        pieces = tokenizer(token, add_special_tokens=False).input_ids
        if len(pieces) == 1:
            ids.append(int(pieces[0]))
    return sorted(set(ids))


def _resolve_pair_token_ids(tokenizer, pairs: Sequence[Tuple[str, str]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for left, right in pairs:
        l_ids = tokenizer(left, add_special_tokens=False).input_ids
        r_ids = tokenizer(right, add_special_tokens=False).input_ids
        if len(l_ids) == 1 and len(r_ids) == 1:
            out.append((int(l_ids[0]), int(r_ids[0])))
    return out


def _discover_placeholder_tokens(rows: Sequence[dict]) -> List[str]:
    found: Set[str] = set()
    pattern = re.compile(r"<\|[^|<>]+\|>")
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        for token in pattern.findall(text):
            found.add(token)
    return sorted(found)


def _needs_token_registration(tokenizer, token: str) -> bool:
    pieces = tokenizer(token, add_special_tokens=False).input_ids
    if len(pieces) != 1:
        return True
    tid = int(pieces[0])
    if tokenizer.unk_token_id is None:
        return False
    return tid == int(tokenizer.unk_token_id) and token != str(tokenizer.unk_token)


def _register_symbolic_tokens(tokenizer, model, token_texts: Sequence[str]) -> List[str]:
    # Make symbolic spans atomic so mask extraction can reliably target them.
    unique = [t for t in sorted(set(token_texts)) if t]
    to_add = [tok for tok in unique if _needs_token_registration(tokenizer, tok)]
    if not to_add:
        return []
    added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return to_add


def _resolve_token_bank_tensor(tokenizer, token_texts: Sequence[str], device):
    import torch

    ids: List[int] = []
    for token in token_texts:
        pieces = tokenizer(token, add_special_tokens=False).input_ids
        if len(pieces) != 1:
            continue
        tid = int(pieces[0])
        if tokenizer.unk_token_id is not None and tid == int(tokenizer.unk_token_id) and token != str(tokenizer.unk_token):
            continue
        ids.append(tid)
    if not ids:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)


def _gather_masked_hidden(hidden, input_ids, attention_mask, token_bank):
    import torch

    if not hasattr(token_bank, "numel"):
        token_bank = torch.tensor(list(token_bank), dtype=torch.long, device=input_ids.device)

    if token_bank.numel() == 0:
        empty_states = hidden.new_empty((0, hidden.shape[-1]))
        empty_ids = input_ids.new_empty((0,))
        empty_mask = attention_mask.new_zeros(attention_mask.shape, dtype=torch.bool)
        return empty_states, empty_ids, empty_mask

    if hasattr(torch, "isin"):
        token_hits = torch.isin(input_ids, token_bank)
    else:
        token_hits = (input_ids.unsqueeze(-1) == token_bank.view(1, 1, -1)).any(dim=-1)
    mask = token_hits & attention_mask.bool()
    if int(mask.sum().item()) == 0:
        empty_states = hidden.new_empty((0, hidden.shape[-1]))
        empty_ids = input_ids.new_empty((0,))
        return empty_states, empty_ids, mask
    states = hidden[mask]
    ids = input_ids[mask]
    return states, ids, mask


def _resolve_pair_ids(pair_texts, tokenizer=None) -> List[Tuple[int, int]]:
    pair_ids: List[Tuple[int, int]] = []
    if not pair_texts:
        return pair_ids
    if tokenizer is None:
        for left, right in pair_texts:
            pair_ids.append((int(left), int(right)))
        return pair_ids
    for left, right in pair_texts:
        l_ids = tokenizer(left, add_special_tokens=False).input_ids
        r_ids = tokenizer(right, add_special_tokens=False).input_ids
        if len(l_ids) == 1 and len(r_ids) == 1:
            pair_ids.append((int(l_ids[0]), int(r_ids[0])))
    return pair_ids


def _dynamic_cooccurring_pairs(input_ids, attention_mask, token_bank=None, max_pairs: int = 64) -> List[Tuple[int, int]]:
    import torch

    valid = attention_mask.bool()
    bsz, _ = input_ids.shape
    out: List[Tuple[int, int]] = []
    seen = set()
    for b in range(bsz):
        seq = input_ids[b][valid[b]]
        if token_bank is not None and hasattr(token_bank, "numel") and int(token_bank.numel()) > 0:
            if hasattr(torch, "isin"):
                keep = torch.isin(seq, token_bank)
            else:
                keep = (seq.unsqueeze(-1) == token_bank.view(1, -1)).any(dim=-1)
            seq = seq[keep]
        if int(seq.numel()) < 2:
            continue
        vals = [int(x.item()) for x in seq]
        for i in range(len(vals) - 1):
            li = vals[i]
            for j in range(i + 1, len(vals)):
                rj = vals[j]
                if li == rj:
                    continue
                pair = (li, rj)
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(pair)
                if len(out) >= max_pairs:
                    return out
    return out


def _trace_mask_from_bounds(attention_mask, trace_start, answer_start):
    import torch

    bsz, seqlen = attention_mask.shape
    idx = torch.arange(seqlen, device=attention_mask.device).unsqueeze(0).expand(bsz, seqlen)
    t = trace_start.unsqueeze(1).to(attention_mask.device)
    a = answer_start.unsqueeze(1).to(attention_mask.device)
    valid_trace = t >= 0
    valid_answer = a >= 0
    no_trace = ~valid_trace
    has_trace_no_answer = valid_trace & (~valid_answer)
    has_both = valid_trace & valid_answer
    mask = torch.zeros((bsz, seqlen), device=attention_mask.device, dtype=torch.bool)
    mask = mask | (has_trace_no_answer & (idx >= t))
    mask = mask | (has_both & (idx >= t) & (idx < a))
    mask = mask & attention_mask.bool() & (~no_trace)
    return mask


class DynamicAnchorMiner:
    """Rolling co-occurrence miner for emergent compositional anchors."""

    def __init__(
        self,
        buffer_size: int = 1000,
        top_k: int = 150,
        z_threshold: float = 2.0,
        min_count: int = 2,
        mode: str = "zscore",
        null_permutations: int = 2,
        min_enrichment: float = 2.0,
        persistence_refreshes: int = 2,
        elbow_min_pairs: int = 2,
        mi_floor: float = 0.0,
        seed: int = 7,
    ):
        self.buffer: Deque[List[int]] = deque(maxlen=max(1, int(buffer_size)))
        self.top_k = max(1, int(top_k))
        self.z_threshold = float(z_threshold)
        self.min_count = max(1, int(min_count))
        self.mode = str(mode).strip().lower()
        self.null_permutations = max(1, int(null_permutations))
        self.min_enrichment = float(min_enrichment)
        self.persistence_refreshes = max(1, int(persistence_refreshes))
        self.elbow_min_pairs = max(1, int(elbow_min_pairs))
        self.mi_floor = float(mi_floor)
        self.rng = random.Random(int(seed))
        self.current_pair_ids: List[Tuple[int, int]] = []
        self.current_pair_tokens: List[Tuple[str, str]] = []
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.last_candidate_count = 0
        self.last_selected_count = 0
        self.last_elbow_index = 0
        self.last_mean_enrichment = 0.0

    @staticmethod
    def _count_adjacent_pairs(seqs: Sequence[Sequence[int]]) -> Counter:
        pair_counts: Counter = Counter()
        for seq in seqs:
            if len(seq) < 2:
                continue
            for a, b in zip(seq[:-1], seq[1:]):
                if a == b:
                    continue
                pair_counts[(int(a), int(b))] += 1
        return pair_counts

    @staticmethod
    def _count_token_occurrences(seqs: Sequence[Sequence[int]]) -> Counter:
        token_counts: Counter = Counter()
        for seq in seqs:
            for tok in seq:
                token_counts[int(tok)] += 1
        return token_counts

    @staticmethod
    def _knee_index(contribs: Sequence[float], min_pairs: int) -> int:
        if not contribs:
            return 0
        n = len(contribs)
        if n <= 2:
            return n
        total = sum(float(x) for x in contribs)
        if total <= 0.0:
            return min(n, max(1, int(min_pairs)))
        cum = []
        run = 0.0
        for c in contribs:
            run += float(c)
            cum.append(run / total)
        best_i = 0
        best_d = -1.0
        for i in range(n):
            x = float(i + 1) / float(n)
            y = float(cum[i])
            # Distance from identity line on normalized cumulative-gain curve.
            d = y - x
            if d > best_d:
                best_d = d
                best_i = i
        return max(min(n, best_i + 1), max(1, int(min_pairs)))

    def add_batch(self, input_ids, attention_mask, trace_start, answer_start):
        import torch

        trace_mask = _trace_mask_from_bounds(attention_mask, trace_start, answer_start)
        if int(trace_mask.sum().item()) == 0:
            return
        bsz = input_ids.shape[0]
        for i in range(bsz):
            m = trace_mask[i]
            if int(m.sum().item()) == 0:
                m = attention_mask[i].bool()
            seq = input_ids[i][m]
            if int(seq.numel()) < 2:
                continue
            # Keep ids as plain Python ints for cheap rolling storage.
            self.buffer.append([int(x.item()) for x in seq])

    def mine(self, tokenizer=None) -> List[Tuple[int, int]]:
        pair_counts = self._count_adjacent_pairs(self.buffer)
        if not pair_counts:
            self.current_pair_ids = []
            self.current_pair_tokens = []
            self.last_candidate_count = 0
            self.last_selected_count = 0
            self.last_elbow_index = 0
            self.last_mean_enrichment = 0.0
            return []

        chosen: List[Tuple[int, int]] = []
        if self.mode == "non_arbitrary":
            token_counts = self._count_token_occurrences(self.buffer)
            total_pairs = max(1.0, float(sum(pair_counts.values())))
            scored: List[Tuple[float, float, int, Tuple[int, int]]] = []
            for pair, c in pair_counts.items():
                if int(c) < self.min_count:
                    continue
                left, right = pair
                cl = max(1, int(token_counts.get(int(left), 0)))
                cr = max(1, int(token_counts.get(int(right), 0)))
                # PMI-like signal for directed adjacency.
                pmi = math.log((float(c) * total_pairs) / (float(cl) * float(cr)))
                info_gain = (float(c) / total_pairs) * max(0.0, pmi)
                if info_gain < self.mi_floor:
                    continue
                scored.append((info_gain, pmi, int(c), (int(left), int(right))))
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            contribs = [x[0] for x in scored]
            elbow = self._knee_index(contribs, min_pairs=self.elbow_min_pairs)
            self.last_elbow_index = int(elbow)
            elbow_candidates = scored[: min(int(elbow), int(self.top_k))]
            self.last_candidate_count = len(elbow_candidates)

            # Null-model baseline via in-sequence permutations.
            null_counts: Counter = Counter()
            seqs = list(self.buffer)
            for _ in range(self.null_permutations):
                shuffled = []
                for seq in seqs:
                    if len(seq) < 2:
                        continue
                    s = list(seq)
                    self.rng.shuffle(s)
                    shuffled.append(s)
                null_counts.update(self._count_adjacent_pairs(shuffled))
            denom = max(1, self.null_permutations)
            enriched: List[Tuple[float, Tuple[int, int]]] = []
            for _, _, c, pair in elbow_candidates:
                exp_c = float(null_counts.get(pair, 0)) / float(denom)
                enr = (float(c) + 1e-6) / (exp_c + 1e-6)
                if enr >= self.min_enrichment:
                    enriched.append((enr, pair))
            if enriched:
                self.last_mean_enrichment = float(sum(e for e, _ in enriched) / len(enriched))
            else:
                self.last_mean_enrichment = 0.0
            enriched_pairs = [pair for _, pair in sorted(enriched, key=lambda x: x[0], reverse=True)]

            # Temporal persistence gate.
            current = set(enriched_pairs)
            for pair in list(self.persistence_counts.keys()):
                if pair in current:
                    self.persistence_counts[pair] = int(self.persistence_counts.get(pair, 0)) + 1
                else:
                    self.persistence_counts[pair] = 0
            for pair in current:
                if pair not in self.persistence_counts:
                    self.persistence_counts[pair] = 1
            hardened = [p for p in enriched_pairs if int(self.persistence_counts.get(p, 0)) >= self.persistence_refreshes]
            chosen = hardened[: self.top_k]
        else:
            values = list(pair_counts.values())
            n = float(len(values))
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = var ** 0.5
            threshold = mean + self.z_threshold * std

            ranked: List[Tuple[float, int, Tuple[int, int]]] = []
            for pair, c in pair_counts.items():
                if c < self.min_count:
                    continue
                z = (float(c) - mean) / std if std > 1e-8 else 0.0
                if float(c) >= threshold:
                    ranked.append((z, int(c), pair))
            ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = [pair for _, _, pair in ranked[: self.top_k]]
            self.last_candidate_count = len(ranked)
            self.last_elbow_index = 0
            self.last_mean_enrichment = 0.0

        self.current_pair_ids = [(int(a), int(b)) for a, b in chosen]
        self.last_selected_count = len(self.current_pair_ids)

        tokens: List[Tuple[str, str]] = []
        if tokenizer is not None:
            for a, b in self.current_pair_ids:
                ta = tokenizer.convert_ids_to_tokens([a])[0]
                tb = tokenizer.convert_ids_to_tokens([b])[0]
                tokens.append((str(ta), str(tb)))
        self.current_pair_tokens = tokens
        return self.current_pair_ids

    def top_pairs_text(self, k: int = 5) -> str:
        if self.current_pair_tokens:
            pairs = self.current_pair_tokens[: max(1, int(k))]
            text = ", ".join([f"{a}->{b}" for a, b in pairs])
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        pairs = self.current_pair_ids[: max(1, int(k))]
        return ", ".join([f"{a}->{b}" for a, b in pairs])

    def diagnostics(self) -> Dict[str, float]:
        persisted = 0
        if self.persistence_counts:
            persisted = sum(1 for _, c in self.persistence_counts.items() if int(c) >= self.persistence_refreshes)
        return {
            "dynamic_anchor_candidate_count": float(self.last_candidate_count),
            "dynamic_anchor_selected_count": float(self.last_selected_count),
            "dynamic_anchor_elbow_index": float(self.last_elbow_index),
            "dynamic_anchor_mean_enrichment": float(self.last_mean_enrichment),
            "dynamic_anchor_persisted_count": float(persisted),
        }


def semantic_unambiguity_loss(hidden, input_ids, attention_mask, token_bank):
    import torch

    states, token_ids, _ = _gather_masked_hidden(hidden, input_ids, attention_mask, token_bank)
    if int(states.shape[0]) < 2:
        return hidden.new_tensor(0.0, requires_grad=True)

    per_token = []
    for tid in token_ids.unique():
        token_states = states[token_ids == tid]
        if int(token_states.shape[0]) < 2:
            continue
        # trace(covariance) = sum of feature variances for that symbolic token's span.
        per_token.append(token_states.var(dim=0, unbiased=False).sum())
    if not per_token:
        return hidden.new_tensor(0.0, requires_grad=True)
    return torch.stack(per_token).mean()


def semantic_hit_count(input_ids, attention_mask, token_bank) -> int:
    _, _, mask = _gather_masked_hidden(
        hidden=attention_mask.unsqueeze(-1).float(),
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_bank=token_bank,
    )
    return int(mask.sum().item())


def compositional_consistency_loss(hidden, input_ids, attention_mask, pair_texts, tokenizer=None, dynamic_token_bank=None):
    import torch

    pair_ids = _resolve_pair_ids(pair_texts, tokenizer=tokenizer)
    if not pair_ids:
        pair_ids = _dynamic_cooccurring_pairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_bank=dynamic_token_bank,
        )
        if not pair_ids:
            return hidden.new_tensor(0.0, requires_grad=True)

    bsz, seqlen = input_ids.shape
    valid = attention_mask.bool()
    diffs = []
    for left_id, right_id in pair_ids:
        left_hit = (input_ids == int(left_id)) & valid
        right_hit = (input_ids == int(right_id)) & valid
        for b in range(bsz):
            left_pos = torch.nonzero(left_hit[b], as_tuple=False).flatten()
            right_pos = torch.nonzero(right_hit[b], as_tuple=False).flatten()
            if left_pos.numel() == 0 or right_pos.numel() == 0:
                continue
            for lp in left_pos:
                rp = right_pos[right_pos > lp]
                if rp.numel() == 0:
                    continue
                first_r = rp[0]
                diffs.append(hidden[b, first_r] - hidden[b, lp])
    if len(diffs) < 2:
        dynamic_pairs = _dynamic_cooccurring_pairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_bank=dynamic_token_bank,
        )
        for left_id, right_id in dynamic_pairs:
            left_hit = (input_ids == int(left_id)) & valid
            right_hit = (input_ids == int(right_id)) & valid
            for b in range(bsz):
                left_pos = torch.nonzero(left_hit[b], as_tuple=False).flatten()
                right_pos = torch.nonzero(right_hit[b], as_tuple=False).flatten()
                if left_pos.numel() == 0 or right_pos.numel() == 0:
                    continue
                for lp in left_pos:
                    rp = right_pos[right_pos > lp]
                    if rp.numel() == 0:
                        continue
                    diffs.append(hidden[b, rp[0]] - hidden[b, lp])
        if len(diffs) < 2:
            return hidden.new_tensor(0.0, requires_grad=True)
    stack = torch.stack(diffs, dim=0)
    return stack.var(dim=0, unbiased=False).mean()


def compositional_match_count(input_ids, attention_mask, pair_texts, tokenizer=None, dynamic_token_bank=None) -> int:
    import torch

    pair_ids = _resolve_pair_ids(pair_texts, tokenizer=tokenizer)
    if not pair_ids:
        pair_ids = _dynamic_cooccurring_pairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_bank=dynamic_token_bank,
        )
        if not pair_ids:
            return 0
    bsz, _ = input_ids.shape
    valid = attention_mask.bool()
    matches = 0
    for left_id, right_id in pair_ids:
        left_hit = (input_ids == int(left_id)) & valid
        right_hit = (input_ids == int(right_id)) & valid
        for b in range(bsz):
            left_pos = torch.nonzero(left_hit[b], as_tuple=False).flatten()
            right_pos = torch.nonzero(right_hit[b], as_tuple=False).flatten()
            if left_pos.numel() == 0 or right_pos.numel() == 0:
                continue
            for lp in left_pos:
                rp = right_pos[right_pos > lp]
                if rp.numel() > 0:
                    matches += 1
    return matches


def roundtrip_consistency_loss(hidden, problem_ids, mode_ids, trace_start, answer_start, attention_mask):
    import torch

    # Surrogate round-trip: crystal/fluid representations for same problem should align.
    trace_mask = _trace_mask_from_bounds(attention_mask, trace_start, answer_start)
    pooled: List[Tuple[int, int, object]] = []
    for i in range(hidden.shape[0]):
        m = trace_mask[i]
        if int(m.sum().item()) == 0:
            m = attention_mask[i].bool()
        if int(m.sum().item()) == 0:
            continue
        vec = hidden[i][m].mean(dim=0)
        pooled.append((int(problem_ids[i].item()), int(mode_ids[i].item()), vec))

    by_pid: Dict[int, Dict[int, object]] = {}
    for pid, mode_id, vec in pooled:
        by_pid.setdefault(pid, {})
        by_pid[pid][mode_id] = vec

    losses = []
    for _, row in by_pid.items():
        if 0 in row and 1 in row:
            losses.append(torch.mean((row[0] - row[1]) ** 2))
    if not losses:
        return hidden.new_tensor(0.0)
    return torch.stack(losses).mean()


def coverage_regularization_loss(logits, input_ids, attention_mask, token_ids, trace_start, answer_start):
    import torch

    if not token_ids:
        return logits.new_tensor(0.0)
    trace_mask = _trace_mask_from_bounds(attention_mask, trace_start, answer_start)
    if int(trace_mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    idx = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    sel = logits[trace_mask][:, idx]
    probs = sel.softmax(dim=-1).mean(dim=0)
    probs = probs / probs.sum().clamp_min(1e-8)
    k = probs.numel()
    uniform = torch.full_like(probs, 1.0 / float(k))
    kl = torch.sum(probs * (torch.log(probs.clamp_min(1e-8)) - torch.log(uniform.clamp_min(1e-8))))
    return kl


def compression_regularization_loss(hidden, attention_mask, trace_start, answer_start):
    trace_mask = _trace_mask_from_bounds(attention_mask, trace_start, answer_start)
    if int(trace_mask.sum().item()) == 0:
        return hidden.new_tensor(0.0)
    trace_hidden = hidden[trace_mask]
    return (trace_hidden.pow(2).mean()).sqrt()


def trajectory_balance_loss(
    logits,
    input_ids,
    attention_mask,
    trace_start,
    answer_start,
    reward_beta: float = 1.0,
    reward_floor: float = 1e-6,
    air_gapped_oracle: bool = True,
    uniform_backward_policy: bool = True,
):
    import math
    import torch

    trace_mask = _trace_mask_from_bounds(attention_mask, trace_start, answer_start)
    if int(trace_mask.sum().item()) == 0:
        return logits.new_tensor(0.0)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    steps = trace_mask.sum(dim=1).clamp_min(1).to(logits.device).float()
    # Length-normalize before squaring to avoid O(L^2) residual growth.
    log_pf = (token_logp * trace_mask.float()).sum(dim=1) / steps

    if uniform_backward_policy:
        log_pb = (-torch.log(steps + 1.0)) / steps
    else:
        log_pb = torch.zeros_like(log_pf)

    oracle_source = token_logp.detach() if air_gapped_oracle else token_logp
    mean_logp = (oracle_source * trace_mask.float()).sum(dim=1) / steps
    reward = torch.exp(mean_logp).clamp_min(float(reward_floor))
    beta = float(reward_beta)
    if beta != 1.0:
        reward = reward.pow(beta)
    log_reward = torch.log(reward.clamp_min(float(reward_floor)))

    log_z = torch.zeros_like(log_pf)
    residual = log_z + log_pf - log_reward - log_pb
    return (residual.pow(2)).mean()


def embedding_anchor_loss(model, token_ids, anchor_vectors):
    import torch

    if token_ids is None or anchor_vectors is None:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.new_tensor(0.0)
        return None
    if int(token_ids.numel()) == 0:
        return anchor_vectors.new_tensor(0.0)
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        return anchor_vectors.new_tensor(0.0)
    current = emb.weight[token_ids.to(emb.weight.device)]
    target = anchor_vectors.to(emb.weight.device)
    return torch.mean((current - target) ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on symbolic identity traces.")
    parser.add_argument("--base-model", required=True, help="HF model id or local model path")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL with a text field")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_adapter"))
    parser.add_argument("--adapter-init", type=Path, default=None, help="Optional LoRA adapter path to continue training from")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="If >0, override epochs and stop after this many optimizer steps.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--answer-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--prompt-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--trace-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--trace-anchor", default="\nTRACE:")
    parser.add_argument("--answer-anchor", default="\nANSWER:")

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    parser.add_argument("--phase5-objectives", action="store_true")
    parser.add_argument("--semantic-unambiguity-weight", type=float, default=0.05)
    parser.add_argument("--compositional-consistency-weight", type=float, default=0.05)
    parser.add_argument("--roundtrip-consistency-weight", type=float, default=0.05)
    parser.add_argument("--coverage-regularization-weight", type=float, default=0.01)
    parser.add_argument("--compression-regularization-weight", type=float, default=0.01)
    parser.add_argument("--trajectory-balance-weight", type=float, default=0.0)
    parser.add_argument("--embedding-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--phase5-air-gapped-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach oracle reward from policy graph during trajectory-balance updates.",
    )
    parser.add_argument("--phase5-reward-temperature-beta", type=float, default=1.0)
    parser.add_argument("--phase5-reward-floor", type=float, default=1e-6)
    parser.add_argument(
        "--phase5-uniform-backward-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a uniform backward prior in trajectory-balance loss.",
    )
    parser.add_argument(
        "--phase5-freeze-symbolic-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze all input embeddings after symbolic token registration.",
    )
    parser.add_argument(
        "--semantic-tokens",
        nargs="+",
        default=["ASSUME", "PROVE", "VERIFY", "ANS", "CONSISTENT", "INCONSISTENT"],
    )
    parser.add_argument(
        "--compositional-pairs",
        nargs="+",
        default=["IF:THEN", "ASSUME:VERIFY", "BIND:STATE", "BELIEF:CLAIM"],
        help="Token pairs in the form LEFT:RIGHT",
    )
    parser.add_argument(
        "--compositional-anchors-file",
        type=Path,
        default=None,
        help="Optional JSON file with pre-mined compositional pairs; overrides --compositional-pairs if provided.",
    )
    parser.add_argument("--dynamic-anchor-miner", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dynamic-anchor-mode",
        type=str,
        choices=["zscore", "non_arbitrary"],
        default="zscore",
        help="Anchor miner mode. non_arbitrary uses MI elbow + permutation null baseline + persistence gate.",
    )
    parser.add_argument("--dynamic-anchor-buffer-size", type=int, default=1000)
    parser.add_argument("--dynamic-anchor-refresh-steps", type=int, default=500)
    parser.add_argument("--dynamic-anchor-top-k", type=int, default=150)
    parser.add_argument("--dynamic-anchor-z-threshold", type=float, default=2.0)
    parser.add_argument("--dynamic-anchor-min-count", type=int, default=2)
    parser.add_argument("--dynamic-anchor-null-permutations", type=int, default=2)
    parser.add_argument("--dynamic-anchor-min-enrichment", type=float, default=2.0)
    parser.add_argument("--dynamic-anchor-persistence-refreshes", type=int, default=2)
    parser.add_argument("--dynamic-anchor-elbow-min-pairs", type=int, default=2)
    parser.add_argument("--dynamic-anchor-mi-floor", type=float, default=0.0)
    parser.add_argument("--disable-rope", action="store_true", help="Apply DroPE-style NoPE patch for Qwen2 attention.")
    parser.add_argument(
        "--coverage-tokens",
        nargs="+",
        default=["ASSUME", "VERIFY", "BIND", "STATE", "BELIEF", "CLAIM", "ANS"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        missing = str(exc)
        raise SystemExit(
            "Missing training dependency. Install with: "
            "pip install transformers datasets peft accelerate torch bitsandbytes\n"
            f"Original error: {missing}"
        )

    rope_patch_active = False
    if args.disable_rope:
        rope_patch_active = patch_nope_qwen2()
        print(f"DroPE patch active: {rope_patch_active}")

    rows = load_jsonl(args.dataset)
    if not rows:
        raise ValueError(f"No rows found in dataset: {args.dataset}")

    dataset = Dataset.from_list(rows)
    pair_specs = parse_pair_specs(args.compositional_pairs)
    if args.compositional_anchors_file is not None and args.compositional_anchors_file.exists():
        pair_specs = load_compositional_anchors(args.compositional_anchors_file)
        print(f"Loaded {len(pair_specs)} compositional anchor pairs from {args.compositional_anchors_file}.")
    discovered_placeholders = _discover_placeholder_tokens(rows)
    semantic_target_tokens = sorted(set(list(args.semantic_tokens) + discovered_placeholders))
    coverage_target_tokens = sorted(set(list(args.coverage_tokens) + discovered_placeholders))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[args.bnb_4bit_compute_dtype]

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    symbolic_tokens = list(semantic_target_tokens) + list(coverage_target_tokens)
    symbolic_tokens.extend([left for left, _ in pair_specs])
    symbolic_tokens.extend([right for _, right in pair_specs])
    added = _register_symbolic_tokens(tokenizer, model, symbolic_tokens)
    if added:
        print(f"Registered {len(added)} symbolic tokens for stable span extraction.")
    anchor_token_texts = sorted(set(symbolic_tokens))
    anchor_token_ids = _resolve_single_token_ids(tokenizer, anchor_token_texts)
    if args.phase5_freeze_symbolic_embeddings:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight.requires_grad:
            emb.weight.requires_grad = False
            print("Froze input embeddings for symbolic grounding hygiene.")

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.adapter_init is not None:
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_init),
            local_files_only=args.local_files_only,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )
        model = get_peft_model(model, lora_config)

    anchor_token_ids_tensor = None
    anchor_embedding_vectors = None
    if anchor_token_ids:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            anchor_token_ids_tensor = torch.tensor(anchor_token_ids, dtype=torch.long, device=emb.weight.device)
            anchor_embedding_vectors = emb.weight.detach()[anchor_token_ids_tensor].clone()

    def tokenize(batch: dict) -> dict:
        input_ids = []
        attention_mask = []
        labels = []
        loss_weights = []
        trace_starts = []
        answer_starts = []
        problem_ids = []
        mode_ids = []
        trace_anchors = batch.get("trace_anchor")
        answer_anchors = batch.get("answer_anchor")
        prompt_ws = batch.get("prompt_loss_multiplier")
        trace_ws = batch.get("trace_loss_multiplier")
        answer_ws = batch.get("answer_loss_multiplier")
        row_problem_ids = batch.get("problem_id")
        row_modes = batch.get("mode")

        for i, text in enumerate(batch["text"]):
            enc = tokenizer(text, truncation=True, max_length=args.max_length)
            ids = enc["input_ids"]
            am = enc["attention_mask"]
            label = list(ids)

            prompt_w = float(prompt_ws[i]) if prompt_ws is not None else float(args.prompt_loss_multiplier)
            trace_w = float(trace_ws[i]) if trace_ws is not None else float(args.trace_loss_multiplier)
            answer_w = float(answer_ws[i]) if answer_ws is not None else float(args.answer_loss_multiplier)

            trace_anchor = str(trace_anchors[i]) if trace_anchors is not None else str(args.trace_anchor)
            answer_anchor = str(answer_anchors[i]) if answer_anchors is not None else str(args.answer_anchor)
            trace_start = find_anchor_boundary(
                text=text,
                anchor=trace_anchor,
                ids_len=len(ids),
                prefix_token_len_fn=lambda prefix: len(
                    tokenizer(prefix, truncation=True, max_length=args.max_length)["input_ids"]
                ),
            )
            answer_start = find_anchor_boundary(
                text=text,
                anchor=answer_anchor,
                ids_len=len(ids),
                prefix_token_len_fn=lambda prefix: len(
                    tokenizer(prefix, truncation=True, max_length=args.max_length)["input_ids"]
                ),
            )
            weights = compute_segment_weights(
                ids_len=len(ids),
                prompt_w=prompt_w,
                trace_w=trace_w,
                answer_w=answer_w,
                trace_start=trace_start,
                answer_start=answer_start,
            )

            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(label)
            loss_weights.append(weights)
            trace_starts.append(int(trace_start))
            answer_starts.append(int(answer_start))
            pid = int(row_problem_ids[i]) if row_problem_ids is not None else i
            mode_value = str(row_modes[i]).strip().lower() if row_modes is not None else "unknown"
            mode_id = 0 if mode_value == "crystal" else 1 if mode_value == "fluid" else 2
            problem_ids.append(pid)
            mode_ids.append(mode_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
            "trace_start": trace_starts,
            "answer_start": answer_starts,
            "problem_id": problem_ids,
            "mode_id": mode_ids,
        }

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    def collator(features: list[dict]) -> dict:
        padded = tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]

        batch_labels = []
        batch_weights = []
        for f in features:
            lbl = list(f["labels"])
            wts = list(f["loss_weights"])
            pad_len = max_len - len(lbl)
            batch_labels.append(lbl + ([-100] * pad_len))
            batch_weights.append(wts + ([0.0] * pad_len))

        padded["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        padded["loss_weights"] = torch.tensor(batch_weights, dtype=torch.float32)
        padded["trace_start"] = torch.tensor([int(f.get("trace_start", -1)) for f in features], dtype=torch.long)
        padded["answer_start"] = torch.tensor([int(f.get("answer_start", -1)) for f in features], dtype=torch.long)
        padded["problem_id"] = torch.tensor([int(f.get("problem_id", -1)) for f in features], dtype=torch.long)
        padded["mode_id"] = torch.tensor([int(f.get("mode_id", 2)) for f in features], dtype=torch.long)
        return padded

    class WeightedLossTrainer(Trainer):
        def __init__(self, *inner_args, **inner_kwargs):
            super().__init__(*inner_args, **inner_kwargs)
            self.last_loss_breakdown = {}
            self.dynamic_miner = (
                DynamicAnchorMiner(
                    buffer_size=args.dynamic_anchor_buffer_size,
                    top_k=args.dynamic_anchor_top_k,
                    z_threshold=args.dynamic_anchor_z_threshold,
                    min_count=args.dynamic_anchor_min_count,
                    mode=args.dynamic_anchor_mode,
                    null_permutations=args.dynamic_anchor_null_permutations,
                    min_enrichment=args.dynamic_anchor_min_enrichment,
                    persistence_refreshes=args.dynamic_anchor_persistence_refreshes,
                    elbow_min_pairs=args.dynamic_anchor_elbow_min_pairs,
                    mi_floor=args.dynamic_anchor_mi_floor,
                    seed=args.seed,
                )
                if args.dynamic_anchor_miner
                else None
            )
            self.dynamic_pair_ids: List[Tuple[int, int]] = []
            self.dynamic_next_refresh_step = max(1, int(args.dynamic_anchor_refresh_steps))
            self.dynamic_top5 = ""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            weights = inputs.pop("loss_weights", None)
            trace_start = inputs.pop("trace_start", None)
            answer_start = inputs.pop("answer_start", None)
            problem_ids = inputs.pop("problem_id", None)
            mode_ids = inputs.pop("mode_id", None)
            outputs = model(output_hidden_states=args.phase5_objectives, **inputs)
            if labels is None or weights is None:
                loss = outputs.loss if hasattr(outputs, "loss") else None
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous().to(shift_logits.device)

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
            active = (shift_labels != -100).float()
            weighted_loss = token_loss * shift_weights * active
            denom = (shift_weights * active).sum().clamp_min(1.0)
            ce_loss = weighted_loss.sum() / denom
            loss = ce_loss
            self.last_loss_breakdown = {"ce_loss": float(ce_loss.detach().item())}

            if args.phase5_objectives:
                hidden = outputs.hidden_states[-1]
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                t_start = trace_start if trace_start is not None else labels.new_full((labels.shape[0],), -1)
                a_start = answer_start if answer_start is not None else labels.new_full((labels.shape[0],), -1)
                pids = problem_ids if problem_ids is not None else labels.new_full((labels.shape[0],), -1)
                mids = mode_ids if mode_ids is not None else labels.new_full((labels.shape[0],), 2)
                if self.dynamic_miner is not None:
                    self.dynamic_miner.add_batch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        trace_start=t_start,
                        answer_start=a_start,
                    )
                    # Re-anchor grammar every refresh interval.
                    if (self.state.global_step + 1) >= self.dynamic_next_refresh_step:
                        self.dynamic_pair_ids = self.dynamic_miner.mine(tokenizer=tokenizer)
                        self.dynamic_top5 = self.dynamic_miner.top_pairs_text(k=5)
                        self.dynamic_next_refresh_step += max(1, int(args.dynamic_anchor_refresh_steps))

                semantic_bank = _resolve_token_bank_tensor(tokenizer, semantic_target_tokens, input_ids.device)
                coverage_bank = _resolve_token_bank_tensor(tokenizer, coverage_target_tokens, input_ids.device)
                dynamic_pair_bank = semantic_bank if int(semantic_bank.numel()) > 0 else coverage_bank
                pair_target = self.dynamic_pair_ids if self.dynamic_pair_ids else pair_specs
                sem_hits = semantic_hit_count(input_ids, attention_mask, semantic_bank)
                comp_hits = compositional_match_count(
                    input_ids,
                    attention_mask,
                    pair_target,
                    tokenizer if not self.dynamic_pair_ids else None,
                    dynamic_token_bank=dynamic_pair_bank,
                )

                sem_loss = semantic_unambiguity_loss(hidden, input_ids, attention_mask, semantic_bank)
                comp_loss = compositional_consistency_loss(
                    hidden,
                    input_ids,
                    attention_mask,
                    pair_target,
                    tokenizer if not self.dynamic_pair_ids else None,
                    dynamic_token_bank=dynamic_pair_bank,
                )
                rt_loss = roundtrip_consistency_loss(hidden, pids, mids, t_start, a_start, attention_mask)
                cov_loss = coverage_regularization_loss(
                    logits=logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_ids=coverage_bank.tolist(),
                    trace_start=t_start,
                    answer_start=a_start,
                )
                cmp_loss = compression_regularization_loss(hidden, attention_mask, t_start, a_start)
                tb_loss = trajectory_balance_loss(
                    logits=logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    trace_start=t_start,
                    answer_start=a_start,
                    reward_beta=args.phase5_reward_temperature_beta,
                    reward_floor=args.phase5_reward_floor,
                    air_gapped_oracle=args.phase5_air_gapped_oracle,
                    uniform_backward_policy=args.phase5_uniform_backward_policy,
                )
                emb_anchor_loss = embedding_anchor_loss(
                    model=model,
                    token_ids=anchor_token_ids_tensor,
                    anchor_vectors=anchor_embedding_vectors,
                )
                if emb_anchor_loss is None:
                    emb_anchor_loss = ce_loss.new_tensor(0.0)

                loss = (
                    ce_loss
                    + args.semantic_unambiguity_weight * sem_loss
                    + args.compositional_consistency_weight * comp_loss
                    + args.roundtrip_consistency_weight * rt_loss
                    + args.coverage_regularization_weight * cov_loss
                    + args.compression_regularization_weight * cmp_loss
                    + args.trajectory_balance_weight * tb_loss
                    + args.embedding_anchor_weight * emb_anchor_loss
                )

                self.last_loss_breakdown.update(
                    {
                        "semantic_unambiguity_loss": float(sem_loss.detach().item()),
                        "semantic_hit_count": float(sem_hits),
                        "compositional_consistency_loss": float(comp_loss.detach().item()),
                        "compositional_match_count": float(comp_hits),
                        "roundtrip_consistency_loss": float(rt_loss.detach().item()),
                        "coverage_regularization_loss": float(cov_loss.detach().item()),
                        "compression_regularization_loss": float(cmp_loss.detach().item()),
                        "trajectory_balance_loss": float(tb_loss.detach().item()),
                        "embedding_anchor_loss": float(emb_anchor_loss.detach().item()),
                        "total_loss": float(loss.detach().item()),
                    }
                )
                if self.dynamic_miner is not None:
                    self.last_loss_breakdown["dynamic_anchor_pair_count"] = float(len(self.dynamic_pair_ids))
                    self.last_loss_breakdown["dynamic_anchor_mode"] = str(args.dynamic_anchor_mode)
                    self.last_loss_breakdown.update(self.dynamic_miner.diagnostics())
                    if self.dynamic_top5:
                        self.last_loss_breakdown["dynamic_anchor_top5"] = self.dynamic_top5
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *inner_args, **inner_kwargs):
            if args.phase5_objectives and self.last_loss_breakdown:
                for k, v in self.last_loss_breakdown.items():
                    if k not in logs:
                        logs[k] = v
            return super().log(logs, *inner_args, **inner_kwargs)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16 and torch.cuda.is_available()),
        fp16=(compute_dtype == torch.float16 and torch.cuda.is_available()),
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
