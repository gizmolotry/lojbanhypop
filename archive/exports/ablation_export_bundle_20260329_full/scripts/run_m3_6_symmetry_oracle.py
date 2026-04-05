from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from lojban_evolution.l_series import infer_swap_semantics


@dataclass
class OracleExample:
    cell: str
    operator: str
    arity: int
    expected_label: str  # invariant | foil
    original_expr: str
    swapped_expr: str
    original_value: Any
    swapped_value: Any
    canonical_equivalent: bool
    predicted_label: str
    routing_correct: bool


def _and(args: Tuple[Any, ...]) -> bool:
    return all(bool(x) for x in args)


def _or(args: Tuple[Any, ...]) -> bool:
    return any(bool(x) for x in args)


def _equal(args: Tuple[Any, ...]) -> bool:
    return args[0] == args[1]


def _coref(args: Tuple[Any, ...]) -> bool:
    return str(args[0]).lower() == str(args[1]).lower()


def _seteq(args: Tuple[Any, ...]) -> bool:
    return set(args[0]) == set(args[1])


def _gt(args: Tuple[Any, ...]) -> bool:
    return int(args[0]) > int(args[1])


def _inside(args: Tuple[Any, ...]) -> bool:
    return str(args[0]).startswith(str(args[1]))


def _cause(args: Tuple[Any, ...]) -> str:
    return f"{args[0]}->{args[1]}"


def _north_of(args: Tuple[Any, ...]) -> bool:
    return int(args[0]) > int(args[1])


def _give(args: Tuple[Any, ...]) -> str:
    giver, recipient, obj = args
    return f"{giver}|{recipient}|{obj}"


def _move(args: Tuple[Any, ...]) -> str:
    agent, origin, destination = args
    return f"{agent}:{origin}>{destination}"


def _and3(args: Tuple[Any, ...]) -> bool:
    return bool(args[0]) and bool(args[1]) and bool(args[2])


def _permute_binary(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return (args[1], args[0])


def _permute_nary(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    if len(args) < 3:
        return _permute_binary(args)
    return (args[1], args[2], args[0])


def _expr(op: str, symbols: Tuple[str, ...]) -> str:
    return f"{op}(" + ",".join(symbols) + ")"


def _label_from_symmetry(is_symmetric: bool) -> str:
    return "invariant" if is_symmetric else "foil"


def _make_args(rng: random.Random, arity: int, op: str) -> Tuple[Any, ...]:
    if op in {"AND", "OR", "AND3"}:
        pool = [True, False]
        return tuple(rng.choice(pool) for _ in range(arity))
    if op in {"EQUAL", "GT", "NORTH_OF"}:
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        while b == a:
            b = rng.randint(0, 9)
        return (a, b)
    if op == "COREF":
        a = rng.choice(["Alice", "ALICE", "Bob", "BOB"])
        b = rng.choice(["alice", "ALICE", "bob", "BOB"])
        return (a, b)
    if op == "SET_EQ":
        a = tuple(sorted(rng.sample([1, 2, 3, 4], 2)))
        b = tuple(sorted(rng.sample([1, 2, 3, 4], 2)))
        return (a, b)
    if op == "INSIDE":
        outer = rng.choice(["BOX", "ROOM", "ZONE"])
        inner = outer + rng.choice(["_A", "_B", "_C"])
        return (inner, outer)
    if op == "CAUSE":
        return (rng.choice(["RAIN", "WIND", "HEAT"]), rng.choice(["FLOOD", "DUST", "FIRE"]))
    if op == "GIVE":
        return (
            rng.choice(["ALICE", "BOB", "CAROL"]),
            rng.choice(["X", "Y", "Z"]),
            rng.choice(["BOOK", "KEY", "MAP"]),
        )
    if op == "MOVE":
        return (
            rng.choice(["ROBOT", "DRONE", "AGENT"]),
            rng.choice(["A", "B", "C"]),
            rng.choice(["D", "E", "F"]),
        )
    return tuple(rng.randint(0, 9) for _ in range(arity))


def _suite_specs() -> Dict[str, List[Tuple[str, int, bool, Callable[[Tuple[Any, ...]], Any]]]]:
    symmetric = [
        ("AND", 2, True, _and),
        ("OR", 2, True, _or),
        ("EQUAL", 2, True, _equal),
        ("COREF", 2, True, _coref),
        ("SET_EQ", 2, True, _seteq),
        ("AND3", 3, True, _and3),
    ]
    asymmetric = [
        ("GT", 2, False, _gt),
        ("INSIDE", 2, False, _inside),
        ("CAUSE", 2, False, _cause),
        ("NORTH_OF", 2, False, _north_of),
        ("GIVE", 3, False, _give),
        ("MOVE", 3, False, _move),
    ]
    mixed = symmetric + asymmetric
    return {"M3.6.A": symmetric, "M3.6.B": asymmetric, "M3.6.C": mixed}


def _confusion(rows: List[OracleExample]) -> Dict[str, int]:
    out = {
        "symmetric_to_invariant": 0,
        "symmetric_to_foil": 0,
        "asymmetric_to_invariant": 0,
        "asymmetric_to_foil": 0,
    }
    for r in rows:
        if r.expected_label == "invariant":
            if r.predicted_label == "invariant":
                out["symmetric_to_invariant"] += 1
            else:
                out["symmetric_to_foil"] += 1
        else:
            if r.predicted_label == "invariant":
                out["asymmetric_to_invariant"] += 1
            else:
                out["asymmetric_to_foil"] += 1
    return out


def _acc(rows: List[OracleExample], label: str | None = None) -> float:
    target = [r for r in rows if label is None or r.expected_label == label]
    if not target:
        return 0.0
    return sum(1 for r in target if r.routing_correct) / float(len(target))


def _rate(rows: List[OracleExample], label: str, predicate: Callable[[OracleExample], bool]) -> float:
    target = [r for r in rows if r.expected_label == label]
    if not target:
        return 0.0
    return sum(1 for r in target if predicate(r)) / float(len(target))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.6 Symmetry Oracle Validation Suite (evaluation-only).")
    p.add_argument("--samples-per-operator", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--baseline-manifest", type=Path, default=Path("docs/baselines/m_series_baseline_manifest.json"))
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle"))
    p.add_argument("--run-id", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_manifest.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {args.baseline_manifest}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rid = args.run_id.strip() or f"m3_6_{ts}"
    out_dir = args.output_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    specs = _suite_specs()
    rows: List[OracleExample] = []
    symbols_by_arity = {2: ("A", "B"), 3: ("A", "B", "C")}

    for cell, ops in specs.items():
        for op, arity, symmetric, fn in ops:
            for _ in range(int(args.samples_per_operator)):
                args_orig = _make_args(rng, arity, op)
                args_swap = _permute_nary(args_orig) if arity > 2 else _permute_binary(args_orig)
                expr_orig = _expr(op, symbols_by_arity[arity])
                expr_swap = _expr(op, _permute_nary(symbols_by_arity[arity]) if arity > 2 else _permute_binary(symbols_by_arity[arity]))
                expected = _label_from_symmetry(bool(symmetric))
                v1 = fn(args_orig)
                v2 = fn(args_swap)
                canonical_eq = bool(v1 == v2)
                pred = infer_swap_semantics(expr_orig)
                pred = "invariant" if pred == "invariant" else "foil"
                rows.append(
                    OracleExample(
                        cell=cell,
                        operator=op,
                        arity=arity,
                        expected_label=expected,
                        original_expr=expr_orig,
                        swapped_expr=expr_swap,
                        original_value=v1,
                        swapped_value=v2,
                        canonical_equivalent=canonical_eq,
                        predicted_label=pred,
                        routing_correct=(pred == expected),
                    )
                )

    by_cell: Dict[str, List[OracleExample]] = {}
    for r in rows:
        by_cell.setdefault(r.cell, []).append(r)

    report_cells: Dict[str, Any] = {}
    for cell, rs in by_cell.items():
        symmetric_rows = [x for x in rs if x.expected_label == "invariant"]
        sym_false_foil = (
            sum(1 for x in symmetric_rows if x.predicted_label == "foil") / float(max(1, len(symmetric_rows)))
        )
        report_cells[cell] = {
            "count": len(rs),
            "symmetry_oracle_accuracy_total": _acc(rs),
            "symmetry_oracle_accuracy_symmetric": _acc(rs, "invariant"),
            "symmetry_oracle_accuracy_asymmetric": _acc(rs, "foil"),
            "symmetry_false_foil_rate": float(sym_false_foil),
            "confusion_matrix": _confusion(rs),
            "truth_preservation_rate_for_invariants": _rate(rs, "invariant", lambda x: x.canonical_equivalent),
            "truth_flip_rate_for_foils": _rate(rs, "foil", lambda x: not x.canonical_equivalent),
            "canonical_equivalence_rate": sum(1 for x in rs if x.canonical_equivalent == (x.expected_label == "invariant"))
            / float(max(1, len(rs))),
        }

    all_sym = [x for x in rows if x.expected_label == "invariant"]
    overall_sym_false_foil = sum(1 for x in all_sym if x.predicted_label == "foil") / float(max(1, len(all_sym)))

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "series_id": "M",
        "track": "M3.6",
        "report_type": "symmetry_oracle_validation_suite",
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "suite_size": len(rows),
        "samples_per_operator": int(args.samples_per_operator),
        "symmetry_false_foil_rate": float(overall_sym_false_foil),
        "cells": report_cells,
    }

    suite_path = out_dir / "m3_6_symmetry_oracle_suite.json"
    rep_json = out_dir / "m3_6_symmetry_oracle_report.json"
    rep_md = out_dir / "m3_6_symmetry_oracle_report.md"

    suite_path.write_text(json.dumps([asdict(x) for x in rows], indent=2, default=str), encoding="utf-8")
    rep_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M3.6 Symmetry Oracle Validation Suite",
        "",
        f"- generated_utc: `{report['generated_utc']}`",
        f"- suite_size: `{report['suite_size']}`",
        f"- samples_per_operator: `{report['samples_per_operator']}`",
        "",
        f"- symmetry_false_foil_rate: `{float(overall_sym_false_foil):.4f}`",
        "",
        "| cell | total_acc | symmetric_acc | asymmetric_acc | sym_false_foil | truth_preserve_inv | truth_flip_foil | canonical_eq_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ("M3.6.A", "M3.6.B", "M3.6.C"):
        c = report_cells.get(cell, {})
        md.append(
            f"| `{cell}` | {float(c.get('symmetry_oracle_accuracy_total', 0.0)):.4f} | "
            f"{float(c.get('symmetry_oracle_accuracy_symmetric', 0.0)):.4f} | "
            f"{float(c.get('symmetry_oracle_accuracy_asymmetric', 0.0)):.4f} | "
            f"{float(c.get('symmetry_false_foil_rate', 0.0)):.4f} | "
            f"{float(c.get('truth_preservation_rate_for_invariants', 0.0)):.4f} | "
            f"{float(c.get('truth_flip_rate_for_foils', 0.0)):.4f} | "
            f"{float(c.get('canonical_equivalence_rate', 0.0)):.4f} |"
        )
    rep_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {suite_path}")
    print(f"Wrote: {rep_json}")
    print(f"Wrote: {rep_md}")


if __name__ == "__main__":
    main()
