from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


BUCKETS = (
    "noun_phrase",
    "clause_continuation",
    "disambiguation",
    "explanatory",
)


@dataclass(frozen=True)
class FamilySpec:
    family: str
    candidate_a: str
    candidate_b: str
    template: str
    variant0_clause: str
    variant1_clause: str
    explanation0: str
    explanation1: str


def _specs() -> List[FamilySpec]:
    return [
        FamilySpec(
            family="trophy_suitcase",
            candidate_a="trophy",
            candidate_b="suitcase",
            template="The {a} does not fit in the {b} because it is {clause}.",
            variant0_clause="too large",
            variant1_clause="too small",
            explanation0="it was too large",
            explanation1="it was too small",
        ),
        FamilySpec(
            family="councilmen_demonstrators",
            candidate_a="councilmen",
            candidate_b="demonstrators",
            template="The {a} refused the permit to the {b} because they {clause}.",
            variant0_clause="feared violence",
            variant1_clause="advocated revolution",
            explanation0="they feared violence",
            explanation1="they advocated revolution",
        ),
        FamilySpec(
            family="engineer_analyst",
            candidate_a="engineer",
            candidate_b="analyst",
            template="The {a} sent the report to the {b} because they {clause}.",
            variant0_clause="needed feedback",
            variant1_clause="had deeper context",
            explanation0="they needed feedback",
            explanation1="they had deeper context",
        ),
        FamilySpec(
            family="lawyer_witness",
            candidate_a="lawyer",
            candidate_b="witness",
            template="The {a} questioned the {b} after they {clause}.",
            variant0_clause="reviewed the affidavit",
            variant1_clause="contradicted prior testimony",
            explanation0="they reviewed it first",
            explanation1="they contradicted testimony",
        ),
        FamilySpec(
            family="alex_riley",
            candidate_a="Alex",
            candidate_b="Riley",
            template="{a} thanked {b} because they {clause}.",
            variant0_clause="offered useful help",
            variant1_clause="needed support",
            explanation0="they offered help",
            explanation1="they needed support",
        ),
        FamilySpec(
            family="manager_intern",
            candidate_a="manager",
            candidate_b="intern",
            template="The {a} monitored the {b}, so they {clause}.",
            variant0_clause="prevented mistakes",
            variant1_clause="learned the process",
            explanation0="they prevented mistakes",
            explanation1="they learned the process",
        ),
        FamilySpec(
            family="morgan_casey",
            candidate_a="memo",
            candidate_b="Casey",
            template="Morgan did not forward the {a} to {b} because they were {clause}.",
            variant0_clause="confidential",
            variant1_clause="unauthorized",
            explanation0="it was confidential",
            explanation1="they were unauthorized",
        ),
        FamilySpec(
            family="truck_bus",
            candidate_a="truck",
            candidate_b="bus",
            template="The {a} passed the {b} because it was {clause}.",
            variant0_clause="very fast",
            variant1_clause="very slow",
            explanation0="it was very fast",
            explanation1="it was very slow",
        ),
    ]


def _target_text(bucket: str, answer: str, foil: str, explanation: str) -> str:
    if bucket == "noun_phrase":
        return answer
    if bucket == "clause_continuation":
        return f"it was {answer}"
    if bucket == "disambiguation":
        return f"{answer}, not {foil}"
    if bucket == "explanatory":
        return f"{answer}, because {explanation}"
    raise ValueError(f"Unsupported bucket {bucket!r}")


def _build_row(spec: FamilySpec, pair_id: int, variant_index: int, bucket: str) -> dict:
    clause = spec.variant0_clause if variant_index == 0 else spec.variant1_clause
    explanation = spec.explanation0 if variant_index == 0 else spec.explanation1
    candidates = [spec.candidate_a, spec.candidate_b]
    gold_index = 0 if variant_index == 0 else 1
    answer = candidates[gold_index]
    foil = candidates[1 - gold_index]
    prompt = spec.template.format(a=spec.candidate_a, b=spec.candidate_b, clause=clause)
    return {
        "prompt": prompt,
        "answer": answer,
        "target_text": _target_text(bucket, answer, foil, explanation),
        "family": spec.family,
        "bucket": bucket,
        "polarity": "candidate_0" if gold_index == 0 else "candidate_1",
        "pair_id": f"m3_19_pair_{pair_id:05d}",
        "variant_id": f"v{variant_index}",
        "candidates": candidates,
        "gold_index": gold_index,
        "target_kind": "rich_resumption" if bucket != "noun_phrase" else "answer_span",
    }


def _validate_rows(rows: List[dict], strict_balance: bool) -> None:
    required = {
        "prompt",
        "answer",
        "target_text",
        "family",
        "bucket",
        "polarity",
        "pair_id",
        "variant_id",
        "candidates",
        "gold_index",
        "target_kind",
    }
    for idx, row in enumerate(rows):
        missing = required.difference(row.keys())
        if missing:
            raise ValueError(f"row {idx} missing fields: {sorted(missing)}")
        if row["bucket"] not in BUCKETS:
            raise ValueError(f"row {idx} has invalid bucket {row['bucket']!r}")
        if row["gold_index"] not in (0, 1):
            raise ValueError(f"row {idx} has invalid gold_index")
        if row["answer"] != row["candidates"][row["gold_index"]]:
            raise ValueError(f"row {idx} answer does not align with candidates/gold_index")
        if not str(row["target_text"]).strip():
            raise ValueError(f"row {idx} has empty target_text")

    if strict_balance:
        by_bucket = {bucket: [r for r in rows if r["bucket"] == bucket] for bucket in BUCKETS}
        for bucket, bucket_rows in by_bucket.items():
            c0 = sum(1 for r in bucket_rows if int(r["gold_index"]) == 0)
            c1 = sum(1 for r in bucket_rows if int(r["gold_index"]) == 1)
            if c0 != c1:
                raise ValueError(f"strict balance violated in {bucket}: candidate_0={c0}, candidate_1={c1}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic M3.19 4-bucket rich resumption pack.")
    p.add_argument("--size", type=int, default=512, help="Total examples. Use a multiple of 8 for exact bucket/label balance.")
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--output", type=Path, default=Path("runs/m3_19_resumption_pack.jsonl"))
    p.add_argument("--summary-output", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    size = int(args.size)
    if bool(args.strict_balance) and size % (len(BUCKETS) * 2) != 0:
        raise ValueError("--strict-balance requires --size to be divisible by 8")

    rng = random.Random(int(args.seed))
    specs = _specs()
    rows: List[dict] = []
    pair_id = 0

    while len(rows) < size:
        spec = specs[pair_id % len(specs)]
        bucket = BUCKETS[(pair_id // len(specs)) % len(BUCKETS)]
        rows.append(_build_row(spec, pair_id=pair_id, variant_index=0, bucket=bucket))
        if len(rows) < size:
            rows.append(_build_row(spec, pair_id=pair_id, variant_index=1, bucket=bucket))
        pair_id += 1

    rows = rows[:size]
    rng.shuffle(rows)
    _validate_rows(rows, strict_balance=bool(args.strict_balance))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "size": len(rows),
        "strict_balance": bool(args.strict_balance),
        "bucket_counts": {bucket: sum(1 for r in rows if r["bucket"] == bucket) for bucket in BUCKETS},
        "bucket_label_counts": {
            bucket: {
                "candidate_0": sum(1 for r in rows if r["bucket"] == bucket and int(r["gold_index"]) == 0),
                "candidate_1": sum(1 for r in rows if r["bucket"] == bucket and int(r["gold_index"]) == 1),
            }
            for bucket in BUCKETS
        },
        "families": sorted({str(r["family"]) for r in rows}),
        "output": str(args.output).replace("\\", "/"),
    }
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
