from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_trace_tokens(text: str, trace_anchor: str, answer_anchor: str) -> List[str]:
    i = text.find(trace_anchor)
    if i < 0:
        return []
    start = i + len(trace_anchor)
    j = text.find(answer_anchor, start)
    trace = text[start:] if j < 0 else text[start:j]
    toks = [t.strip() for t in trace.replace("\n", " ").split(" ") if t.strip()]
    return toks


def mine_pairs(rows: List[dict], trace_anchor: str, answer_anchor: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        toks = extract_trace_tokens(text, trace_anchor=trace_anchor, answer_anchor=answer_anchor)
        if len(toks) < 2:
            continue
        for a, b in zip(toks[:-1], toks[1:]):
            if a == b:
                continue
            counts[(a, b)] += 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine frequent trace token pairs for compositional anchors.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/compositional_anchors.json"))
    p.add_argument("--trace-anchor", default="\nTRACE:")
    p.add_argument("--answer-anchor", default="\nANSWER:")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--min-count", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    counts = mine_pairs(rows, trace_anchor=args.trace_anchor, answer_anchor=args.answer_anchor)
    ranked = [((a, b), c) for (a, b), c in counts.most_common() if int(c) >= int(args.min_count)]
    ranked = ranked[: int(args.top_k)]
    pairs = [[a, b] for (a, b), _ in ranked]
    payload = {
        "dataset": str(args.dataset),
        "trace_anchor": args.trace_anchor,
        "answer_anchor": args.answer_anchor,
        "top_k": int(args.top_k),
        "min_count": int(args.min_count),
        "pairs": pairs,
        "counts": [{"left": a, "right": b, "count": int(c)} for (a, b), c in ranked],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
