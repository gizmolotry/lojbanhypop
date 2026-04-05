from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AugmentedLagrangianController:
    """Tier-A controller for lexicographic augmented Lagrangian training."""

    rho: float = 1.0
    constraints: Sequence[str] = ("arity", "scope", "identity")
    init_lambda: float = 0.0
    max_lambda: float = 1_000.0
    lambdas: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lambdas:
            self.lambdas = {name: float(self.init_lambda) for name in self.constraints}
        else:
            for name in self.constraints:
                self.lambdas.setdefault(name, float(self.init_lambda))

    def penalty(self, constraints: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        total = torch.zeros((), device=device)
        for name in self.constraints:
            ci = constraints[name]
            lam = float(self.lambdas.get(name, 0.0))
            total = total + (lam * ci) + (0.5 * float(self.rho) * ci * ci)
        return total

    def update(self, constraint_values: Mapping[str, float]) -> None:
        for name in self.constraints:
            ci = max(0.0, float(constraint_values.get(name, 0.0)))
            new_lam = max(0.0, float(self.lambdas.get(name, 0.0)) + float(self.rho) * ci)
            self.lambdas[name] = min(float(self.max_lambda), new_lam)


def compute_arity_violation(triples: Sequence[Tuple[int, int, int]], relation_vocab: int, var_min_id: int) -> float:
    if not triples:
        return 0.0
    bad = 0
    for rel, v1, v2 in triples:
        if rel < 0 or rel >= int(relation_vocab):
            bad += 1
            continue
        if int(v1) < int(var_min_id) or int(v2) < int(var_min_id):
            bad += 1
    return float(bad) / float(len(triples))


@dataclass(frozen=True)
class RelationEvent:
    rel: int
    args: Tuple[int, ...]


def parse_relation_events_from_sequence(
    token_ids: Sequence[int],
    relation_vocab: int,
    var_min_id: int,
    operator_arity_registry: Mapping[int, int] | None = None,
    default_relation_arity: int = 2,
) -> Tuple[List[RelationEvent], float]:
    """Parse flat token stream into dynamic-signature relation events.

    Each event starts with a relation token (< relation_vocab). The parser looks up the
    expected arity for that relation and consumes exactly N subsequent variable tokens.
    """
    ids = [int(x) for x in token_ids]
    if not ids:
        return [], 0.0
    reg = {int(k): int(v) for k, v in (operator_arity_registry or {}).items()}
    default_n = max(1, int(default_relation_arity))
    i = 0
    bad = 0
    events: List[RelationEvent] = []
    while i < len(ids):
        rel = int(ids[i])
        if rel < 0 or rel >= int(relation_vocab):
            bad += 1
            i += 1
            continue
        n = int(reg.get(rel, default_n))
        n = max(1, n)
        args: List[int] = []
        for j in range(n):
            k = i + 1 + j
            if k >= len(ids):
                bad += 1
                break
            tok = int(ids[k])
            args.append(tok)
            if tok < int(var_min_id):
                bad += 1
        if len(args) == n:
            events.append(RelationEvent(rel=rel, args=tuple(args)))
        i += 1 + n
    denom = float(max(1, len(events) + bad))
    return events, float(bad) / denom


@dataclass
class ScopeViolation:
    unbalanced: int = 0
    unbound: int = 0
    escape: int = 0
    quantifier_assoc: int = 0
    shadowing: int = 0

    @property
    def total(self) -> int:
        return self.unbalanced + self.unbound + self.escape + self.quantifier_assoc + self.shadowing

    @property
    def mismatch(self) -> int:
        # Backward-compatible aggregate for older callers/tests.
        return self.unbalanced + self.quantifier_assoc


def _norm_tok(token: str) -> str:
    return token.strip().upper()


def parse_scope_trace(tokens: Iterable[str]) -> ScopeViolation:
    """Deterministic scope checker for FORALL/EXISTS + SCOPE_OPEN/SCOPE_CLOSE + VAR_* traces."""
    stack: List[set[str]] = []
    pending_bindings: List[str] = []
    seen: set[str] = set()
    out = ScopeViolation()
    toks = [_norm_tok(t) for t in tokens if str(t).strip()]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in {"FORALL", "EXISTS"}:
            if i + 1 >= len(toks) or not toks[i + 1].startswith("VAR_"):
                out.quantifier_assoc += 1
            else:
                active_bound = set().union(*stack) if stack else set()
                if toks[i + 1] in active_bound:
                    out.shadowing += 1
                pending_bindings.append(toks[i + 1])
                seen.add(toks[i + 1])
                i += 1
        elif t == "SCOPE_OPEN":
            stack.append(set(pending_bindings))
            pending_bindings = []
        elif t == "SCOPE_CLOSE":
            if not stack:
                out.unbalanced += 1
            else:
                stack.pop()
        elif t.startswith("VAR_"):
            is_bound = any(t in frame for frame in stack)
            if not is_bound:
                if t in seen:
                    out.escape += 1
                else:
                    out.unbound += 1
                    seen.add(t)
        i += 1

    if stack:
        out.unbalanced += len(stack)
    if pending_bindings:
        out.quantifier_assoc += len(pending_bindings)
    return out


def compute_scope_violation_components(tokens: Iterable[str]) -> Dict[str, float]:
    tok_list = [str(t) for t in tokens if str(t).strip()]
    if not tok_list:
        return {
            "scope_total": 0.0,
            "scope_unbalanced": 0.0,
            "scope_lifetime": 0.0,
            "scope_unbound": 0.0,
            "scope_quantifier_assoc": 0.0,
            "scope_shadowing": 0.0,
        }
    v = parse_scope_trace(tok_list)
    denom = float(len(tok_list))
    return {
        "scope_total": float(v.total) / denom,
        "scope_unbalanced": float(v.unbalanced) / denom,
        "scope_lifetime": float(v.escape) / denom,
        "scope_unbound": float(v.unbound) / denom,
        "scope_quantifier_assoc": float(v.quantifier_assoc) / denom,
        "scope_shadowing": float(v.shadowing) / denom,
    }


def compute_scope_violation_rate(tokens: Iterable[str]) -> float:
    return compute_scope_violation_components(tokens)["scope_total"]


def build_scope_tokens_from_triples(triples: Sequence[Tuple[int, int, int]], var_prefix: str = "VAR") -> List[str]:
    """Builds a deterministic pseudo-scope trace from relation triples.

    This gives Tier-A scope discipline a concrete parser surface even when the
    upstream representation is relation/var ids rather than explicit quantifier text.
    """
    out: List[str] = ["SCOPE_OPEN"]
    introduced: set[int] = set()
    for _rel, v1, v2 in triples:
        if v1 not in introduced:
            out.extend(["FORALL", f"{var_prefix}_{int(v1)}", "SCOPE_OPEN"])
            introduced.add(v1)
        out.append(f"{var_prefix}_{int(v1)}")
        out.append(f"{var_prefix}_{int(v2)}")
    # close in reverse of introductions + outer scope
    out.extend(["SCOPE_CLOSE"] * (len(introduced) + 1))
    return out


def build_scope_tokens_from_events(events: Sequence[RelationEvent], var_prefix: str = "VAR") -> List[str]:
    out: List[str] = ["SCOPE_OPEN"]
    introduced: set[int] = set()
    for ev in events:
        for idx, v in enumerate(ev.args):
            if int(v) not in introduced:
                quant = "FORALL" if idx == 0 else "EXISTS"
                out.extend([quant, f"{var_prefix}_{int(v)}", "SCOPE_OPEN"])
                introduced.add(int(v))
            out.append(f"{var_prefix}_{int(v)}")
    out.extend(["SCOPE_CLOSE"] * (len(introduced) + 1))
    return out


def make_swap_variant(prompt: str, answer: str) -> Optional[Tuple[str, str]]:
    names = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", prompt)
    stop = {
        "SCOPE",
        "MINIMAL",
        "PAIR",
        "QUANTIFIER",
        "ORDER",
        "NEGATION",
        "RETURN",
        "EXACTLY",
        "FORALL",
        "EXISTS",
        "NOT",
        "AND",
        "OR",
        "A",
        "B",
    }
    uniq: List[str] = []
    for n in names:
        if n.upper() in stop:
            continue
        if n not in uniq:
            uniq.append(n)
    if len(uniq) < 2:
        return None
    a, b = uniq[0], uniq[1]

    def _swap(text: str) -> str:
        marker_a = "__SWAP_A__"
        marker_b = "__SWAP_B__"
        x = re.sub(rf"\b{re.escape(a)}\b", marker_a, text)
        x = re.sub(rf"\b{re.escape(b)}\b", marker_b, x)
        x = x.replace(marker_a, b)
        x = x.replace(marker_b, a)
        return x

    return _swap(prompt), _swap(answer)


def compute_identity_violation_from_ce(base_ce: torch.Tensor, swapped_ce: Optional[torch.Tensor], margin: float = 0.05) -> torch.Tensor:
    if swapped_ce is None:
        return torch.zeros_like(base_ce)
    # Violation if swapped variant is much harder than the base variant.
    return F.relu(swapped_ce - base_ce - float(margin))


def lukasiewicz_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b - 1.0, min=0.0, max=1.0)


def lukasiewicz_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b, min=0.0, max=1.0)


def crispness_loss(p: torch.Tensor) -> torch.Tensor:
    return torch.mean(4.0 * p * (1.0 - p))


def entropy_floor_penalty(q: torch.Tensor, h_min: float, eps: float = 1e-8) -> torch.Tensor:
    qn = q / torch.clamp(torch.sum(q, dim=-1, keepdim=True), min=eps)
    h = -torch.sum(qn * torch.log(torch.clamp(qn, min=eps)), dim=-1)
    h_mean = torch.mean(h)
    return torch.clamp(float(h_min) - h_mean, min=0.0)


def overflow_penalty(active_refs: int, capacity: int, ovf_emitted: bool) -> float:
    if int(active_refs) <= int(capacity):
        return 0.0
    return 0.0 if bool(ovf_emitted) else 1.0


def rolling_all_below(values: Sequence[Mapping[str, float]], threshold: float) -> bool:
    if not values:
        return False
    for row in values:
        if any(float(v) > float(threshold) for v in row.values()):
            return False
    return True


def token_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=eps)
    return -torch.sum(p * torch.log(torch.clamp(p, min=eps)), dim=-1)


def to_float_dict(d: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        out[str(k)] = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
    return out


def generate_scope_minimal_pair_samples(n: int = 500, seed: int = 7) -> List[Dict[str, str]]:
    """Build quantifier-negation minimal pairs for scope discipline drills."""
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    names = [
        ("knight", "village"),
        ("agent", "station"),
        ("box", "room"),
        ("student", "class"),
        ("robot", "lab"),
    ]
    rels = ["trusts", "visits", "sees", "contains", "supports"]
    preds = ["truthful", "active", "safe", "connected", "stable"]

    out: List[Dict[str, str]] = []
    for i in range(int(n)):
        pair_type = int(torch.randint(0, 2, (1,), generator=rng).item())
        name = names[int(torch.randint(0, len(names), (1,), generator=rng).item())]
        rel = rels[int(torch.randint(0, len(rels), (1,), generator=rng).item())]
        pred = preds[int(torch.randint(0, len(preds), (1,), generator=rng).item())]
        x, y = name
        if pair_type == 0:
            a = f"FORALL x EXISTS y : {x}(x) AND {y}(y) AND {rel}(x,y)"
            b = f"EXISTS y FORALL x : {x}(x) AND {y}(y) AND {rel}(x,y)"
            truth = "A_NEQ_B"
            prompt = (
                "Scope Minimal Pair (quantifier order):\n"
                f"A: {a}\n"
                f"B: {b}\n"
                "Return exactly A_NEQ_B if meaning differs, else A_EQ_B."
            )
            answer = truth
        else:
            a = f"NOT EXISTS x : {x}(x) AND {pred}(x)"
            b = f"FORALL x : {x}(x) -> NOT {pred}(x)"
            truth = "A_EQ_B"
            prompt = (
                "Scope Minimal Pair (negation/quantifier):\n"
                f"A: {a}\n"
                f"B: {b}\n"
                "Return exactly A_NEQ_B if meaning differs, else A_EQ_B."
            )
            answer = truth
        out.append({"prompt": prompt, "answer": answer, "pair": "scope_minimal_pair", "index": str(i)})
    return out


def infer_swap_semantics(prompt: str) -> str:
    """Infer whether variable swap should preserve or flip semantics.

    Returns:
      - "invariant" for clearly symmetric predicate patterns
      - "foil" otherwise
    """
    txt = str(prompt)
    up = txt.upper()
    symmetric_names = {
        "AND",
        "OR",
        "AND3",
        "SAME_AS",
        "EQUAL",
        "EQUALITY",
        "EQ",
        "COREF",
        "SET_EQ",
        "SETEQ",
        "IDENTICAL",
        "SIBLING",
        "CONNECTED",
        "INTERSECTS",
        "OVERLAPS",
    }
    asymmetric_names = {
        "GT",
        "INSIDE",
        "CONTAINS",
        "CAUSE",
        "CAUSES",
        "NORTH_OF",
        "GIVE",
        "MOVE",
        "PARENT_OF",
    }
    preds = re.findall(r"\b([A-Z_][A-Z0-9_]*)\s*\(", up)
    if preds:
        if any(p in asymmetric_names for p in preds):
            return "foil"
        if all(p in symmetric_names for p in preds):
            return "invariant"
        return "foil"
    # Light natural-language fallback for symmetric forms.
    if (
        (" SAME AS " in f" {up} ")
        or (" EQUAL " in f" {up} ")
        or (" IDENTICAL " in f" {up} ")
        or (" CONNECTED " in f" {up} ")
        or (" SIBLING " in f" {up} ")
    ):
        return "invariant"
    if (
        (" GREATER THAN " in f" {up} ")
        or (" INSIDE " in f" {up} ")
        or (" CONTAINS " in f" {up} ")
        or (" CAUSES " in f" {up} ")
        or (" NORTH OF " in f" {up} ")
        or (" GIVES " in f" {up} ")
    ):
        return "foil"
    return "foil"
