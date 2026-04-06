from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from lojban_evolution.experiment import (
    LanguageSpec,
    Metrics,
    Problem,
    Proposal,
    evaluate,
    generate_dataset,
    is_pareto_improvement,
    split_dataset,
)
from lojban_evolution.repro import dataset_fingerprint, safe_git_commit, write_run_manifest
from lojban_evolution.storage import StoragePath, join_path, make_dirs, write_bytes, write_json, write_text


@dataclass
class VariantConfig:
    name: str
    use_meta_analysis: bool
    use_pareto_gate: bool
    use_retrain: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine_snapshot(repo_root: Path, out_root: StoragePath) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    qdir = str(join_path(out_root, f"quarantine_{ts}"))
    make_dirs(qdir, parents=True, exist_ok=True)

    include_paths = [
        repo_root / "README.md",
        repo_root / "Core Concept_ Self-Optimizing Language Evolution.txt",
    ]
    include_paths.extend(sorted((repo_root / "scripts").glob("*.py")))
    include_paths.extend(sorted((repo_root / "src" / "lojban_evolution").glob("*.py")))

    manifest: Dict[str, dict] = {"timestamp_utc": datetime.utcnow().isoformat() + "Z", "files": {}}
    for src in include_paths:
        if not src.exists() or not src.is_file():
            continue
        rel = src.relative_to(repo_root).as_posix()
        dst = join_path(qdir, rel)
        write_bytes(dst, src.read_bytes())
        manifest["files"][str(rel)] = {
            "sha256": sha256_file(src),
            "bytes": src.stat().st_size,
            "last_write_utc": datetime.utcfromtimestamp(src.stat().st_mtime).isoformat() + "Z",
        }

    write_json(join_path(qdir, "MANIFEST.json"), manifest, indent=2)
    return qdir


def all_ngrams(tokens: Sequence[str], min_n: int = 2, max_n: int = 4) -> Iterable[Tuple[str, ...]]:
    for n in range(min_n, max_n + 1):
        if n > len(tokens):
            break
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])


def encode_trace(trace: Sequence[str], language: LanguageSpec) -> List[str]:
    patterns = sorted(language.macros.items(), key=lambda kv: len(kv[1]), reverse=True)
    out: List[str] = []
    i = 0
    while i < len(trace):
        matched = False
        for macro_name, pattern in patterns:
            p_len = len(pattern)
            if p_len == 0:
                continue
            if tuple(trace[i : i + p_len]) == pattern:
                out.append(macro_name)
                i += p_len
                matched = True
                break
        if not matched:
            out.append(trace[i])
            i += 1
    return out


def phase1_seed_language() -> LanguageSpec:
    return LanguageSpec()


def phase2_generate_solutions(
    train: Sequence[Problem],
    language: LanguageSpec,
    use_retrain: bool,
) -> List[List[str]]:
    if use_retrain:
        return [encode_trace(p.trace, language) for p in train]
    return [list(p.trace) for p in train]


def phase3_meta_analysis(
    trace_sequences: Sequence[Sequence[str]],
    language: LanguageSpec,
    next_macro_index: int,
    rng: random.Random,
    use_meta_analysis: bool,
    top_k: int,
    min_support: int,
) -> tuple[List[Proposal], int]:
    existing_patterns = set(language.macros.values())
    counts: Dict[Tuple[str, ...], int] = {}
    unique: List[Tuple[str, ...]] = []
    seen = set()

    for seq in trace_sequences:
        for ng in all_ngrams(seq, min_n=2, max_n=4):
            if ng in existing_patterns:
                continue
            if any(tok in language.macros for tok in ng):
                continue
            counts[ng] = counts.get(ng, 0) + 1
            if ng not in seen:
                seen.add(ng)
                unique.append(ng)

    proposals: List[Proposal] = []
    if use_meta_analysis:
        ranked = [(pat, sup) for pat, sup in counts.items() if sup >= min_support]
        ranked.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    else:
        rng.shuffle(unique)
        ranked = [(pat, counts.get(pat, 1)) for pat in unique[: max(top_k * 5, top_k)]]

    for pattern, support in ranked[:top_k]:
        name = f"M{next_macro_index:03d}"
        proposals.append(Proposal(macro_name=name, pattern=pattern, support=support))
        next_macro_index += 1
    return proposals, next_macro_index


def phase4_eval_phase5_update(
    language: LanguageSpec,
    val: Sequence[Problem],
    proposals: Sequence[Proposal],
    use_pareto_gate: bool,
    max_accept: int,
) -> tuple[LanguageSpec, Metrics, List[Proposal]]:
    baseline = evaluate(val, language)
    accepted: List[Proposal] = []
    current = LanguageSpec(base_tokens=set(language.base_tokens), macros=dict(language.macros))

    for proposal in proposals:
        temp = LanguageSpec(base_tokens=set(current.base_tokens), macros=dict(current.macros))
        temp.macros[proposal.macro_name] = proposal.pattern
        candidate = evaluate(val, temp)
        ok = is_pareto_improvement(candidate, baseline) if use_pareto_gate else True
        if ok:
            current = temp
            baseline = candidate
            accepted.append(proposal)
            if len(accepted) >= max_accept:
                break
    return current, baseline, accepted


def phase6_retrain_proxy(train: Sequence[Problem], language: LanguageSpec) -> dict:
    encoded = [encode_trace(p.trace, language) for p in train]
    avg_tokens = (sum(len(x) for x in encoded) / len(encoded)) if encoded else 0.0
    return {"train_avg_encoded_tokens": avg_tokens, "train_examples": len(encoded)}


def run_variant(
    variant: VariantConfig,
    train: Sequence[Problem],
    val: Sequence[Problem],
    test: Sequence[Problem],
    iterations: int,
    max_accept: int,
    top_k: int,
    min_support: int,
    seed: int,
) -> dict:
    language = phase1_seed_language()
    next_macro_index = 1
    rng = random.Random(seed + hash(variant.name) % 10_000)
    history = []

    for i in range(iterations):
        before = evaluate(val, language)
        traces = phase2_generate_solutions(train, language, use_retrain=variant.use_retrain)
        proposals, next_macro_index = phase3_meta_analysis(
            trace_sequences=traces,
            language=language,
            next_macro_index=next_macro_index,
            rng=rng,
            use_meta_analysis=variant.use_meta_analysis,
            top_k=top_k,
            min_support=min_support,
        )
        language, after, accepted = phase4_eval_phase5_update(
            language=language,
            val=val,
            proposals=proposals,
            use_pareto_gate=variant.use_pareto_gate,
            max_accept=max_accept,
        )
        retrain_stats = phase6_retrain_proxy(train, language)
        history.append(
            {
                "iteration": i,
                "before": asdict(before),
                "after": asdict(after),
                "proposals": [asdict(p) for p in proposals],
                "accepted": [asdict(p) for p in accepted],
                "language_size": language.token_count(),
                "retrain": retrain_stats,
            }
        )

    test_metrics = evaluate(test, language)
    return {
        "variant": asdict(variant),
        "final_language": {
            "base_token_count": len(language.base_tokens),
            "macro_count": len(language.macros),
            "total_token_count": language.token_count(),
            "macros": {k: list(v) for k, v in sorted(language.macros.items())},
        },
        "history": history,
        "test_metrics": asdict(test_metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-phase language evolution plus ablations.")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--max-accept", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="artifacts/runs")
    return parser.parse_args()


def write_markdown_summary(path: StoragePath, payload: dict) -> None:
    lines = [
        "# Phase Ablation Summary",
        "",
        f"- Timestamp: {payload['timestamp_utc']}",
        f"- Quarantine snapshot: `{payload['quarantine_dir']}`",
        "",
        "## Variants",
        "",
    ]
    for v in payload["variants"]:
        m = v["test_metrics"]
        lines.append(
            f"- {v['variant']['name']}: acc={m['accuracy']:.4f}, "
            f"avg_tokens={m['avg_tokens']:.4f}, parse={m['parse_success_rate']:.4f}, "
            f"macros={v['final_language']['macro_count']}"
        )
    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = args.output_dir
    make_dirs(out_root, parents=True, exist_ok=True)

    quarantine_dir = quarantine_snapshot(repo_root=repo_root, out_root=out_root)

    problems = generate_dataset(size=args.dataset_size, seed=args.seed)
    train, val, test = split_dataset(problems)

    variants = [
        VariantConfig(name="full_phases", use_meta_analysis=True, use_pareto_gate=True, use_retrain=True),
        VariantConfig(name="ablate_meta_analysis", use_meta_analysis=False, use_pareto_gate=True, use_retrain=True),
        VariantConfig(name="ablate_pareto_gate", use_meta_analysis=True, use_pareto_gate=False, use_retrain=True),
        VariantConfig(name="ablate_retrain", use_meta_analysis=True, use_pareto_gate=True, use_retrain=False),
    ]

    results = []
    for v in variants:
        results.append(
            run_variant(
                variant=v,
                train=train,
                val=val,
                test=test,
                iterations=args.iterations,
                max_accept=args.max_accept,
                top_k=args.top_k,
                min_support=args.min_support,
                seed=args.seed,
            )
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = join_path(out_root, f"ablation_{ts}")
    make_dirs(run_dir, parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "dataset_size": args.dataset_size,
            "seed": args.seed,
            "iterations": args.iterations,
            "max_accept": args.max_accept,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        },
        "quarantine_dir": str(quarantine_dir),
        "variants": results,
    }

    write_json(join_path(run_dir, "ablation.json"), payload, indent=2)
    write_markdown_summary(join_path(run_dir, "summary.md"), payload)
    write_run_manifest(
        join_path(run_dir, "run_manifest.json"),
        {
            "script": "scripts/run_phase_ablation.py",
            "argv": sys.argv[1:],
            "git_commit": safe_git_commit(repo_root),
            "config": payload["config"],
            "dataset_fingerprint": dataset_fingerprint(problems),
            "environment": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "outputs": {
                "run_dir": str(run_dir),
                "ablation_json": str(join_path(run_dir, "ablation.json")),
                "summary_md": str(join_path(run_dir, "summary.md")),
                "quarantine_dir": str(quarantine_dir),
            },
        },
    )

    print(f"Wrote: {join_path(run_dir, 'ablation.json')}")
    print(f"Wrote: {join_path(run_dir, 'summary.md')}")
    for v in results:
        m = v["test_metrics"]
        print(
            f"{v['variant']['name']}: "
            f"acc={m['accuracy']:.4f} avg_tokens={m['avg_tokens']:.4f} parse={m['parse_success_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
