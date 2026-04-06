from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from lojban_evolution.experiment import generate_dataset, split_dataset


ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b", re.IGNORECASE)


@dataclass
class EvalRow:
    problem_id: int
    mode: str
    prompt: str
    expected: str
    predicted: str
    normalized_expected: str
    normalized_predicted: str
    correct: bool
    latency_s: float
    error: str | None


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("in the ", "").replace("the ", "")
    compact = NON_ALNUM_RE.sub("", lowered)
    return compact


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_model_output(raw: str) -> str:
    cleaned = strip_ansi(raw).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def canonicalize_roles(text: str) -> str:
    found = {}
    for person, role in ROLE_RE.findall(text):
        found[person.lower()] = role.lower()
    if {"a", "b", "c"}.issubset(found.keys()):
        return f"a={found['a']},b={found['b']},c={found['c']}"
    return ""


def answers_match(expected: str, predicted: str) -> tuple[bool, str, str]:
    n_expected = normalize_answer(expected)
    n_pred = normalize_answer(predicted)
    if "a=knight,b=knave,c=knight" in n_expected or "a=knight,b=knight,c=knave" in n_expected:
        c_expected = canonicalize_roles(expected)
        c_pred = canonicalize_roles(predicted)
        if c_expected and c_pred:
            return c_expected == c_pred, c_expected, c_pred
    if n_pred.startswith(n_expected):
        return True, n_expected, n_pred
    return n_expected == n_pred, n_expected, n_pred


def load_pack(path: Path | None) -> Dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _examples_text(examples: List[Dict[str, str]]) -> str:
    chunks = []
    for ex in examples:
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a:
            continue
        chunks.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(chunks)


def build_prompt(base_prompt: str, mode: str, pack: Dict[str, object] | None) -> str:
    if mode == "baseline":
        return (
            "Solve the logic question. Return only the final answer with no explanation.\n\n"
            f"Question: {base_prompt}\n"
            "Final answer:"
        )

    if mode == "rigid_ids":
        return (
            "Solve the logic question using rigid identity tracking.\n"
            "Rules:\n"
            "1) Bind each entity to fixed IDs (E1/E2/E3).\n"
            "2) Never switch referents once bound.\n"
            "3) Track each agent's belief state separately when needed.\n"
            "4) Return only the final answer with no explanation.\n\n"
            f"Question: {base_prompt}\n"
            "Final answer:"
        )

    if mode == "gguf_pack" and pack is not None:
        rules = pack.get("system_rules", [])
        examples = pack.get("examples", [])
        rules_text = "\n".join(f"- {r}" for r in rules if isinstance(r, str))
        examples_text = _examples_text([e for e in examples if isinstance(e, dict)])
        return (
            "Solve the logic question with strict identity tracking.\n"
            "Rules:\n"
            f"{rules_text}\n"
            "Return only the final answer text.\n\n"
            "Solved examples:\n"
            f"{examples_text}\n\n"
            f"Question: {base_prompt}\n"
            "Final answer:"
        )

    return (
        "Solve the logic question using rigid identity tracking.\n"
        "Rules:\n"
        "1) Bind each entity to fixed IDs (E1/E2/E3).\n"
        "2) Never switch referents once bound.\n"
        "3) Track each agent's belief state separately when needed.\n"
        "4) Use the solved examples as pattern guidance.\n"
        "5) Return only the final answer with no explanation.\n\n"
        "Solved examples:\n"
        "Q: The city councilmen refused the demonstrators a permit because they feared violence. Who feared violence?\n"
        "A: councilmen\n\n"
        "Q: Alice puts the red ball in the box. Bob moves it to the drawer while Alice is outside. Charlie takes the blue ball. Where does Alice think the red ball is?\n"
        "A: box\n\n"
        "Q: A says: 'B is a knave.' B says: 'A and C are both knights.' C says: 'A is a knave.' Who is what?\n"
        "A: A=knight,B=knave,C=knight\n\n"
        f"Question: {base_prompt}\n"
        "Final answer:"
    )


def call_lms(lms_exe: Path, model: str, prompt: str, timeout_s: int, ttl: int) -> tuple[str, float, str | None]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                str(lms_exe),
                "chat",
                model,
                "--yes",
                "--ttl",
                str(ttl),
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=False,
        )
        latency = time.perf_counter() - started
        output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        parsed = parse_model_output(output)
        if completed.returncode != 0:
            return parsed, latency, f"exit_code={completed.returncode}"
        return parsed, latency, None
    except subprocess.TimeoutExpired:
        latency = time.perf_counter() - started
        return "", latency, "timeout"


def run_eval(
    lms_exe: Path,
    model: str,
    sample_size: int,
    seed: int,
    timeout_s: int,
    ttl: int,
    modes: List[str],
    pack: Dict[str, object] | None,
) -> dict:
    dataset = generate_dataset(size=1000, seed=seed)
    _, _, test = split_dataset(dataset)
    sample = test[:sample_size]
    rows: List[EvalRow] = []

    for mode in modes:
        for p in sample:
            prompt = build_prompt(p.prompt, mode, pack)
            predicted, latency, error = call_lms(lms_exe, model, prompt, timeout_s=timeout_s, ttl=ttl)
            ok, n_expected, n_pred = answers_match(p.answer, predicted)
            rows.append(
                EvalRow(
                    problem_id=p.problem_id,
                    mode=mode,
                    prompt=p.prompt,
                    expected=p.answer,
                    predicted=predicted,
                    normalized_expected=n_expected,
                    normalized_predicted=n_pred,
                    correct=ok,
                    latency_s=latency,
                    error=error,
                )
            )

    by_mode = {}
    for mode in modes:
        subset = [r for r in rows if r.mode == mode]
        total = len(subset)
        correct = sum(1 for r in subset if r.correct)
        errors = sum(1 for r in subset if r.error is not None)
        mean_latency = sum(r.latency_s for r in subset) / total if total else 0.0
        by_mode[mode] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "errors": errors,
            "avg_latency_s": mean_latency,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "sample_size": sample_size,
        "seed": seed,
        "timeout_s": timeout_s,
        "ttl": ttl,
        "modes": modes,
        "pack_name": (pack.get("name") if pack else None),
        "summary": by_mode,
        "rows": [asdict(r) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local LM Studio models on identity-tracking tasks.")
    parser.add_argument("--model", required=True, help="Model name known to lms, e.g. qwen2.5-coder-14b-instruct")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--ttl", type=int, default=600)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "rigid_ids", "phase3_fewshot"],
        help="Modes to run. Add gguf_pack with --pack-file for GGUF prompt-pack eval.",
    )
    parser.add_argument("--pack-file", type=Path, default=None)
    parser.add_argument(
        "--lms-exe",
        type=Path,
        default=Path(r"C:\Users\Andrew\.lmstudio\bin\lms.exe"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    pack = load_pack(args.pack_file)
    if "gguf_pack" in args.modes and pack is None:
        raise ValueError("Mode gguf_pack requires --pack-file")

    payload = run_eval(
        lms_exe=args.lms_exe,
        model=args.model,
        sample_size=args.sample_size,
        seed=args.seed,
        timeout_s=args.timeout_s,
        ttl=args.ttl,
        modes=args.modes,
        pack=pack,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"lm_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    for mode, stats in payload["summary"].items():
        print(
            f"{mode}: accuracy={stats['accuracy']:.3f} ({stats['correct']}/{stats['total']}), "
            f"errors={stats['errors']}, avg_latency_s={stats['avg_latency_s']:.2f}"
        )


if __name__ == "__main__":
    main()
