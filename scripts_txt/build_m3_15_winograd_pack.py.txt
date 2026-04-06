from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class FamilySpec:
    family: str
    candidate_a: str
    candidate_b: str
    template: str
    variant0_clause: str
    variant1_clause: str
    causal_direction: str


def _specs() -> List[FamilySpec]:
    return [
        FamilySpec(
            family="trophy_suitcase",
            candidate_a="trophy",
            candidate_b="suitcase",
            template="The {a} does not fit in the {b} because it is {clause}.",
            variant0_clause="too large",
            variant1_clause="too small",
            causal_direction="effect_to_cause",
        ),
        FamilySpec(
            family="councilmen_demonstrators",
            candidate_a="councilmen",
            candidate_b="demonstrators",
            template="The {a} refused the permit to the {b} because they {clause}.",
            variant0_clause="feared violence",
            variant1_clause="advocated revolution",
            causal_direction="effect_to_cause",
        ),
        FamilySpec(
            family="apology_fault",
            candidate_a="Alex",
            candidate_b="Riley",
            template="{a} apologized to {b} because they {clause}.",
            variant0_clause="were at fault",
            variant1_clause="were offended",
            causal_direction="effect_to_cause",
        ),
        FamilySpec(
            family="help_credit",
            candidate_a="Taylor",
            candidate_b="Jordan",
            template="{a} thanked {b} because they {clause}.",
            variant0_clause="offered useful help",
            variant1_clause="needed support",
            causal_direction="effect_to_cause",
        ),
        FamilySpec(
            family="supervision_outcome",
            candidate_a="manager",
            candidate_b="intern",
            template="The {a} monitored the {b}, so they {clause}.",
            variant0_clause="prevented mistakes",
            variant1_clause="learned the process",
            causal_direction="cause_to_effect",
        ),
        FamilySpec(
            family="warning_response",
            candidate_a="parent",
            candidate_b="teenager",
            template="The {a} warned the {b}, so they {clause}.",
            variant0_clause="set clear limits",
            variant1_clause="changed behavior",
            causal_direction="cause_to_effect",
        ),
    ]


def _build_row(spec: FamilySpec, pair_id: int, variant_index: int) -> dict:
    candidates = [spec.candidate_a, spec.candidate_b]
    clause = spec.variant0_clause if variant_index == 0 else spec.variant1_clause
    prompt = spec.template.format(a=spec.candidate_a, b=spec.candidate_b, clause=clause)
    gold_index = 0 if variant_index == 0 else 1
    return {
        "prompt": prompt,
        "answer": candidates[gold_index],
        "family": spec.family,
        "polarity": "candidate_0" if gold_index == 0 else "candidate_1",
        "causal_direction": spec.causal_direction,
        "pair_id": f"pair_{pair_id:05d}",
        "variant_id": f"v{variant_index}",
        "candidates": candidates,
        "gold_index": gold_index,
    }


def _validate_rows(rows: List[dict], strict_balance: bool) -> None:
    if len(rows) < 500:
        raise ValueError("Generated pack must contain at least 500 examples.")

    c0 = sum(1 for r in rows if int(r.get("gold_index", -1)) == 0)
    c1 = sum(1 for r in rows if int(r.get("gold_index", -1)) == 1)
    if strict_balance and c0 != c1:
        raise ValueError(f"strict balance violated: candidate_0={c0}, candidate_1={c1}")

    required = {
        "family",
        "polarity",
        "causal_direction",
        "pair_id",
        "variant_id",
        "candidates",
        "gold_index",
    }
    for i, row in enumerate(rows):
        missing = required.difference(row.keys())
        if missing:
            raise ValueError(f"row {i} missing required metadata fields: {sorted(missing)}")
        candidates = row["candidates"]
        gold_index = int(row["gold_index"])
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError(f"row {i} candidates must be a list of length 2")
        if gold_index not in (0, 1):
            raise ValueError(f"row {i} gold_index must be 0 or 1")
        if row.get("answer") != candidates[gold_index]:
            raise ValueError(f"row {i} answer must equal candidates[gold_index]")
        if row.get("variant_id") not in {"v0", "v1"}:
            raise ValueError(f"row {i} variant_id must be 'v0' or 'v1'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic M3.15 Winograd pack JSONL.")
    p.add_argument("--size", type=int, default=500, help="Total number of examples (must be >=500).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("runs/m3_15_winograd_pack.jsonl"))
    p.add_argument(
        "--strict-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exact global label balance between candidate_0 and candidate_1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    size = int(args.size)
    if size < 500:
        raise ValueError("--size must be >= 500")
    if bool(args.strict_balance) and (size % 2 != 0):
        raise ValueError("--strict-balance requires an even --size")

    rng = random.Random(int(args.seed))
    specs = _specs()
    rows: List[dict] = []
    pair_idx = 0

    target_even = size if (size % 2 == 0) else (size - 1)
    while len(rows) < target_even:
        spec = specs[pair_idx % len(specs)]
        rows.append(_build_row(spec, pair_id=pair_idx, variant_index=0))
        rows.append(_build_row(spec, pair_id=pair_idx, variant_index=1))
        pair_idx += 1

    if len(rows) > target_even:
        rows = rows[:target_even]

    if size % 2 != 0:
        spec = specs[pair_idx % len(specs)]
        extra = _build_row(spec, pair_id=pair_idx, variant_index=rng.randint(0, 1))
        rows.append(extra)

    rng.shuffle(rows)
    _validate_rows(rows, strict_balance=bool(args.strict_balance))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    c0 = sum(1 for r in rows if int(r["gold_index"]) == 0)
    c1 = sum(1 for r in rows if int(r["gold_index"]) == 1)
    print(f"Wrote: {args.output}")
    print(f"Examples: {len(rows)}")
    print(f"Label balance: candidate_0={c0}, candidate_1={c1}")


if __name__ == "__main__":
    main()
