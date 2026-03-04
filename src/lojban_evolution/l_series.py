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


@dataclass
class ScopeViolation:
    unbound: int = 0
    escape: int = 0
    mismatch: int = 0

    @property
    def total(self) -> int:
        return self.unbound + self.escape + self.mismatch


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
                out.mismatch += 1
            else:
                pending_bindings.append(toks[i + 1])
                seen.add(toks[i + 1])
                i += 1
        elif t == "SCOPE_OPEN":
            stack.append(set(pending_bindings))
            pending_bindings = []
        elif t == "SCOPE_CLOSE":
            if not stack:
                out.mismatch += 1
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
        out.mismatch += len(stack)
    if pending_bindings:
        out.mismatch += len(pending_bindings)
    return out


def compute_scope_violation_rate(tokens: Iterable[str]) -> float:
    tok_list = [str(t) for t in tokens if str(t).strip()]
    if not tok_list:
        return 0.0
    v = parse_scope_trace(tok_list)
    return float(v.total) / float(len(tok_list))


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


def make_swap_variant(prompt: str, answer: str) -> Optional[Tuple[str, str]]:
    names = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", prompt)
    uniq: List[str] = []
    for n in names:
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
