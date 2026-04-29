from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EngineRun:
    name: str
    kind: str
    description: str
    command: List[str]
    status: str
    return_code: Optional[int]
    output_dir: str
    metrics: Optional[Dict[str, object]]


def _run(cmd: List[str]) -> int:
    return int(subprocess.call(cmd))


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_subdir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _eval_metrics_from_gate(path: Path) -> Optional[Dict[str, object]]:
    payload = _read_json(path)
    if payload is None:
        return None
    mean_lifts = payload.get("mean_lifts", {})
    return {
        "metric_family": "hf_dual_mode_gate",
        "mean_final_lift": float(mean_lifts.get("final_answer", 0.0)),
        "mean_symbolic_lift": float(mean_lifts.get("symbolic", 0.0)),
        "gate_pass": bool(payload.get("gate_pass", False)),
        "seeds": payload.get("seeds", []),
        "sample_size": payload.get("sample_size"),
    }


def _metrics_from_phase_ablation(ablation_json: Path) -> Optional[Dict[str, object]]:
    payload = _read_json(ablation_json)
    if payload is None:
        return None
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        return None
    full = None
    for row in variants:
        if isinstance(row, dict) and row.get("variant", {}).get("name") == "full_phases":
            full = row
            break
    if full is None:
        return None
    tm = full.get("test_metrics", {})
    return {
        "metric_family": "phase_ablation_test_metrics",
        "accuracy": float(tm.get("accuracy", 0.0)),
        "parse_success_rate": float(tm.get("parse_success_rate", 0.0)),
        "avg_tokens": float(tm.get("avg_tokens", 0.0)),
        "macro_count": int(full.get("final_language", {}).get("macro_count", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three engines under one harness: English-first, Greedy-iterative, and Phase5-gradient."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="LoRA JSONL dataset for train_lora engines")
    parser.add_argument("--output-root", type=Path, default=Path("runs/three_engine_comparison"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--phase5-air-gapped-oracle", action="store_true")
    parser.add_argument("--phase5-freeze-symbolic-embeddings", action="store_true")
    parser.add_argument("--phase5-reward-temperature-beta", type=float, default=1.0)
    parser.add_argument("--trajectory-balance-weight", type=float, default=0.02)
    parser.add_argument("--embedding-anchor-weight", type=float, default=0.02)
    parser.add_argument("--execute", action="store_true", help="Run commands; otherwise only plan commands")
    return parser.parse_args()


def write_summary(path: Path, runs: List[EngineRun], payload: dict) -> None:
    lines: List[str] = []
    lines.append("# Three-Engine Comparison")
    lines.append("")
    lines.append("## Engines")
    lines.append("- `english_ce`: LoRA CE training focused on final-answer segments (trace/prompt downweighted).")
    lines.append("- `greedy_iterative`: symbolic language-evolution loop (proposal + greedy acceptance phases).")
    lines.append("- `phase5_gradient`: LoRA training with Phase-5 gradient objectives enabled.")
    lines.append("")
    lines.append("## Comparability")
    lines.append("- `english_ce` and `phase5_gradient` are directly compared via the same `eval_hf_dual_mode_gate.py` metrics.")
    lines.append("- `greedy_iterative` reports `run_phase_ablation.py` test metrics (accuracy/parse/tokens) and is adjacent, not strictly identical, to the HF gate metric family.")
    lines.append("")
    lines.append("| engine | status | return_code | metric_family | key_metrics |")
    lines.append("|---|---|---:|---|---|")
    for r in runs:
        fam = ""
        km = ""
        if isinstance(r.metrics, dict):
            fam = str(r.metrics.get("metric_family", ""))
            if fam == "hf_dual_mode_gate":
                km = (
                    f"final_lift={r.metrics.get('mean_final_lift', 0.0):+.3f}, "
                    f"symbolic_lift={r.metrics.get('mean_symbolic_lift', 0.0):+.3f}, "
                    f"gate={bool(r.metrics.get('gate_pass', False))}"
                )
            elif fam == "phase_ablation_test_metrics":
                km = (
                    f"acc={r.metrics.get('accuracy', 0.0):.3f}, "
                    f"parse={r.metrics.get('parse_success_rate', 0.0):.3f}, "
                    f"avg_tokens={r.metrics.get('avg_tokens', 0.0):.3f}"
                )
        lines.append(f"| `{r.name}` | `{r.status}` | `{r.return_code}` | `{fam}` | {km} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Use `--execute` to run the engines. Without it, this file is a reproducible plan artifact.")
    lines.append("- For strict apples-to-apples comparison, prioritize the HF gate rows (`english_ce`, `phase5_gradient`).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = args.output_root / ts
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    train_script = str(script_dir / "train_lora.py")
    gate_script = str(script_dir / "eval_hf_dual_mode_gate.py")
    phase_script = str(script_dir / "run_phase_ablation.py")

    runs: List[EngineRun] = []

    # Engine 1: English-first CE
    eng_out = root / "english_ce_adapter"
    eng_gate = root / "english_ce_gate.json"
    english_train_cmd = [
        sys.executable,
        train_script,
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(eng_out),
        "--epochs",
        str(args.epochs),
        "--max-length",
        str(args.max_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--trace-loss-multiplier",
        "0.0",
        "--prompt-loss-multiplier",
        "0.0",
        "--answer-loss-multiplier",
        "1.0",
    ]
    if args.local_files_only:
        english_train_cmd.append("--local-files-only")

    english_gate_cmd = [
        sys.executable,
        gate_script,
        "--base-model",
        args.base_model,
        "--adapter",
        str(eng_out),
        "--sample-size",
        str(args.sample_size),
        "--dataset-size",
        str(args.dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--output",
        str(eng_gate),
    ]
    if args.local_files_only:
        english_gate_cmd.append("--local-files-only")

    status = "planned"
    rc = None
    metrics = None
    if args.execute:
        rc_train = _run(english_train_cmd)
        if rc_train == 0:
            rc_gate = _run(english_gate_cmd)
            rc = rc_gate
            status = "ok" if rc_gate == 0 else "failed_eval"
            if rc_gate == 0:
                metrics = _eval_metrics_from_gate(eng_gate)
        else:
            rc = rc_train
            status = "failed_train"
    runs.append(
        EngineRun(
            name="english_ce",
            kind="lora_hf",
            description="CE-only adapter, answer-focused loss weighting",
            command=english_train_cmd + ["&&"] + english_gate_cmd,
            status=status,
            return_code=rc,
            output_dir=str(root / "english_ce_adapter"),
            metrics=metrics,
        )
    )

    # Engine 2: Greedy iterative system
    greedy_root = root / "greedy_iterative"
    greedy_cmd = [
        sys.executable,
        phase_script,
        "--dataset-size",
        str(args.dataset_size),
        "--seed",
        str(args.seeds[0] if args.seeds else 7),
        "--output-dir",
        str(greedy_root),
    ]
    status = "planned"
    rc = None
    metrics = None
    if args.execute:
        rc = _run(greedy_cmd)
        status = "ok" if rc == 0 else "failed"
        if rc == 0:
            latest = _latest_subdir(greedy_root, "ablation_")
            if latest is not None:
                metrics = _metrics_from_phase_ablation(latest / "ablation.json")
    runs.append(
        EngineRun(
            name="greedy_iterative",
            kind="symbolic_loop",
            description="proposal + greedy acceptance evolution loop",
            command=greedy_cmd,
            status=status,
            return_code=rc,
            output_dir=str(greedy_root),
            metrics=metrics,
        )
    )

    # Engine 3: Phase-5 gradient adapter
    p5_out = root / "phase5_gradient_adapter"
    p5_gate = root / "phase5_gradient_gate.json"
    p5_train_cmd = [
        sys.executable,
        train_script,
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(p5_out),
        "--epochs",
        str(args.epochs),
        "--max-length",
        str(args.max_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--phase5-objectives",
        "--trajectory-balance-weight",
        str(args.trajectory_balance_weight),
        "--embedding-anchor-weight",
        str(args.embedding_anchor_weight),
        "--phase5-reward-temperature-beta",
        str(args.phase5_reward_temperature_beta),
    ]
    if args.phase5_air_gapped_oracle:
        p5_train_cmd.append("--phase5-air-gapped-oracle")
    if args.phase5_freeze_symbolic_embeddings:
        p5_train_cmd.append("--phase5-freeze-symbolic-embeddings")
    if args.local_files_only:
        p5_train_cmd.append("--local-files-only")

    p5_gate_cmd = [
        sys.executable,
        gate_script,
        "--base-model",
        args.base_model,
        "--adapter",
        str(p5_out),
        "--sample-size",
        str(args.sample_size),
        "--dataset-size",
        str(args.dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--output",
        str(p5_gate),
    ]
    if args.local_files_only:
        p5_gate_cmd.append("--local-files-only")

    status = "planned"
    rc = None
    metrics = None
    if args.execute:
        rc_train = _run(p5_train_cmd)
        if rc_train == 0:
            rc_gate = _run(p5_gate_cmd)
            rc = rc_gate
            status = "ok" if rc_gate == 0 else "failed_eval"
            if rc_gate == 0:
                metrics = _eval_metrics_from_gate(p5_gate)
        else:
            rc = rc_train
            status = "failed_train"
    runs.append(
        EngineRun(
            name="phase5_gradient",
            kind="lora_hf",
            description="CE + Phase-5 differentiable objectives",
            command=p5_train_cmd + ["&&"] + p5_gate_cmd,
            status=status,
            return_code=rc,
            output_dir=str(p5_out),
            metrics=metrics,
        )
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_model": args.base_model,
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "max_length": args.max_length,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "sample_size": args.sample_size,
            "dataset_size": args.dataset_size,
            "seeds": args.seeds,
            "max_new_tokens": args.max_new_tokens,
            "local_files_only": args.local_files_only,
            "phase5_air_gapped_oracle": args.phase5_air_gapped_oracle,
            "phase5_freeze_symbolic_embeddings": args.phase5_freeze_symbolic_embeddings,
            "phase5_reward_temperature_beta": args.phase5_reward_temperature_beta,
            "trajectory_balance_weight": args.trajectory_balance_weight,
            "embedding_anchor_weight": args.embedding_anchor_weight,
            "execute": args.execute,
        },
        "engines": [
            {
                "name": r.name,
                "kind": r.kind,
                "description": r.description,
                "command": r.command,
                "status": r.status,
                "return_code": r.return_code,
                "output_dir": r.output_dir,
                "metrics": r.metrics,
            }
            for r in runs
        ],
    }

    json_out = root / "three_engine_comparison.json"
    md_out = root / "three_engine_comparison.md"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(md_out, runs, payload)
    print(f"Wrote: {json_out}")
    print(f"Wrote: {md_out}")
    for r in runs:
        print(f"{r.name}: {r.status}")


if __name__ == "__main__":
    main()
