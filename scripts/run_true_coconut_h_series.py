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
class HRun:
    run_id: str
    name: str
    status: str
    return_code: Optional[int]
    output: Optional[str]
    metrics: Optional[Dict[str, float]]
    command: List[str]


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _extract_metrics(path: Path) -> Optional[Dict[str, float]]:
    payload = _read_json(path)
    if payload is None:
        return None
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    step_means: List[float] = []
    for row in rows:
        trace = row.get("step_cosine")
        if isinstance(trace, list) and trace:
            vals = [float(s.get("cos_mean", 0.0)) for s in trace if isinstance(s, dict)]
            if vals:
                step_means.append(_safe_mean(vals))
    return {
        "base_acc": float(summary.get("base_acc", 0.0)),
        "handoff_acc": float(summary.get("coconut_acc", 0.0)),
        "handoff_lift": float(summary.get("coconut_lift", 0.0)),
        "errors": float(summary.get("errors", 0)),
        "mean_step_cosine": _safe_mean(step_means),
    }


def _run(cmd: List[str], execute: bool) -> tuple[str, Optional[int]]:
    if not execute:
        return "planned", None
    rc = int(subprocess.call(cmd))
    return ("ok", rc) if rc == 0 else ("failed", rc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run H-series ablations for True Coconut continuous feed.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--h1-window", type=int, default=4)
    p.add_argument("--h2-layer-index", type=int, default=12)
    p.add_argument("--h2-layer-scale", type=float, default=1.0)
    p.add_argument("--h3-adapter", type=Path, default=None)
    p.add_argument("--h3-bridge", type=Path, default=None)
    p.add_argument("--h3-layer-index", type=int, default=12)
    p.add_argument("--h3-layer-scale", type=float, default=1.0)
    p.add_argument("--only-runs", type=str, nargs="+", default=None, help="Subset of runs: H1 H2 H3")
    p.add_argument("--output-root", type=Path, default=Path("runs/true_coconut_h_series"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    true_coconut = Path(__file__).resolve().parent / "true_coconut.py"

    runs: List[HRun] = []

    def _seed_cmd(seed: int, mode: str, window: int, out: Path, adapter: Path, extra: Optional[List[str]] = None) -> List[str]:
        cmd = [
            sys.executable,
            str(true_coconut),
            "--base-model",
            args.base_model,
            "--adapter",
            str(adapter),
            "--sample-size",
            str(args.sample_size),
            "--seed",
            str(seed),
            "--dataset-size",
            str(args.dataset_size),
            "--max-logic-new-tokens",
            str(args.max_logic_new_tokens),
            "--max-final-new-tokens",
            str(args.max_final_new_tokens),
            "--virtual-token-window",
            str(window),
            "--injection-mode",
            mode,
            "--log-step-cosine",
            "--output",
            str(out),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        if extra:
            cmd.extend(extra)
        return cmd

    wanted = {r.upper() for r in (args.only_runs or ["H1", "H2", "H3"])}
    for run_id, name, mode, window, adapter, extra in [
        ("H1", "Multi-Vector Injection (Bandwidth)", "input", args.h1_window, args.adapter, None),
        (
            "H2",
            "Mid-Layer Injection (Depth)",
            "midlayer",
            1,
            args.adapter,
            ["--mid-layer-index", str(args.h2_layer_index), "--mid-layer-scale", str(args.h2_layer_scale)],
        ),
        (
            "H3",
            "SwiGLU Mid-Layer Bridge (Non-Linear Alignment)",
            "midlayer",
            1,
            args.h3_adapter if args.h3_adapter is not None else args.adapter,
            [
                "--mid-layer-index",
                str(args.h3_layer_index),
                "--mid-layer-scale",
                str(args.h3_layer_scale),
                "--swiglu-bridge",
                str(args.h3_bridge) if args.h3_bridge is not None else "",
            ],
        ),
    ]:
        if run_id not in wanted:
            continue
        if run_id == "H3" and (args.h3_adapter is None or args.h3_bridge is None):
            runs.append(HRun(run_id, name, "skipped", None, None, None, []))
            continue
        if run_id == "H3":
            extra = [x for x in (extra or []) if x != ""]
        seed_metrics: List[Dict[str, float]] = []
        status = "ok" if args.execute else "planned"
        last_rc: Optional[int] = None
        first_cmd: List[str] = []
        for seed in args.seeds:
            out = out_dir / f"{run_id.lower()}_seed{seed}.json"
            cmd = _seed_cmd(seed, mode, window, out, adapter, extra)
            if not first_cmd:
                first_cmd = cmd
            st, rc = _run(cmd, args.execute)
            last_rc = rc
            if st != "ok":
                status = st
                break
            m = _extract_metrics(out)
            if m is not None:
                seed_metrics.append(m)
        agg = None
        if seed_metrics:
            agg = {
                "base_acc": _safe_mean([m["base_acc"] for m in seed_metrics]),
                "handoff_acc": _safe_mean([m["handoff_acc"] for m in seed_metrics]),
                "handoff_lift": _safe_mean([m["handoff_lift"] for m in seed_metrics]),
                "mean_step_cosine": _safe_mean([m["mean_step_cosine"] for m in seed_metrics]),
                "errors": _safe_mean([m["errors"] for m in seed_metrics]),
            }
        runs.append(HRun(run_id, name, status, last_rc, str(out_dir / f"{run_id.lower()}_seed*.json"), agg, first_cmd))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "dataset_size": args.dataset_size,
        "execute": bool(args.execute),
        "runs": [r.__dict__ for r in runs],
    }
    manifest_path = out_dir / "run_h_series.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Run H Series",
        "",
        "| id | name | status | return_code | key metric |",
        "|---|---|---|---:|---|",
    ]
    for r in runs:
        key = ""
        if isinstance(r.metrics, dict):
            key = (
                f"base={r.metrics.get('base_acc', 0.0):.3f}, "
                f"handoff={r.metrics.get('handoff_acc', 0.0):.3f}, "
                f"lift={r.metrics.get('handoff_lift', 0.0):+.3f}, "
                f"mean_step_cos={r.metrics.get('mean_step_cosine', 0.0):.3f}"
            )
        lines.append(f"| `{r.run_id}` | `{r.name}` | `{r.status}` | `{r.return_code}` | {key} |")
    lines.append("")
    lines.append("- `Shock Tracking`: `mean_step_cos` is averaged from per-row step-wise cosine traces.")
    md_path = out_dir / "run_h_series.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
