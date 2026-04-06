from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GGUF reasoning pack from successful eval rows.")
    parser.add_argument("--eval-json", nargs="+", type=Path, required=True)
    parser.add_argument("--mode", default="phase3_fewshot")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/gguf_reasoning_pack.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dedup: Dict[str, Dict[str, str]] = {}
    source_models: List[str] = []

    for path in args.eval_json:
        payload = load_eval(path)
        source_models.append(str(payload.get("model", "unknown")))
        for row in payload.get("rows", []):
            if row.get("mode") != args.mode:
                continue
            if not bool(row.get("correct", False)):
                continue
            if row.get("error") is not None:
                continue

            question = str(row.get("prompt", "")).strip()
            expected = str(row.get("expected", "")).strip()
            if not question or not expected:
                continue
            if question in dedup:
                continue

            dedup[question] = {
                "question": question,
                "answer": expected,
            }

    examples = list(dedup.values())[: args.max_examples]

    pack = {
        "name": "lojban_hypothesis_gguf_pack_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode_source": args.mode,
        "source_models": sorted(set(source_models)),
        "system_rules": [
            "Bind entities to fixed identities and never swap referents.",
            "Track observer belief separately from world state.",
            "Return only the final answer text.",
        ],
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    main()
