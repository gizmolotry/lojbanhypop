from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from lojban_evolution.storage import StoragePath, is_s3_uri, join_path, make_dirs, write_bytes, write_json, write_text
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata


@dataclass
class RunRecord:
    run_id: str
    name: str
    status: str
    return_code: Optional[int]
    command: List[str]
    output: Optional[str]
    metrics: Optional[Dict[str, object]]
    notes: str


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run(cmd: List[str], execute: bool) -> tuple[str, Optional[int]]:
    if not execute:
        return "planned", None
    rc = int(subprocess.call(cmd))
    return ("ok", rc) if rc == 0 else ("failed", rc)


def _copy_to_output_if_s3(local_file: Path, output_root: StoragePath) -> str:
    if is_s3_uri(output_root):
        remote = join_path(output_root, local_file.name)
        write_bytes(remote, local_file.read_bytes())
        return str(remote)
    return str(local_file)


def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _metrics_from_dual_handoff(path: Path) -> Optional[Dict[str, object]]:
    payload = _read_json(path)
    if payload is None:
        return None
    per_seed = payload.get("per_seed", [])
    base_final = [float(x.get("final_answer", {}).get("base_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    adapter_final = [float(x.get("final_answer", {}).get("adapter_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    handoff_final = [float(x.get("final_answer", {}).get("handoff_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    base_symbolic = [float(x.get("symbolic", {}).get("base_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    adapter_symbolic = [float(x.get("symbolic", {}).get("adapter_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    handoff_symbolic = [float(x.get("symbolic", {}).get("handoff_acc", 0.0)) for x in per_seed if isinstance(x, dict)]
    return {
        "control_base_final_acc": _safe_mean(base_final),
        "control_base_symbolic_acc": _safe_mean(base_symbolic),
        "enhanced_t2t_final_acc": _safe_mean(adapter_final),
        "enhanced_t2t_symbolic_acc": _safe_mean(adapter_symbolic),
        # Backward-compatible aliases.
        "rigid_adapter_final_acc": _safe_mean(adapter_final),
        "rigid_adapter_symbolic_acc": _safe_mean(adapter_symbolic),
        "coconut_handoff_final_acc": _safe_mean(handoff_final),
        "coconut_handoff_symbolic_acc": _safe_mean(handoff_symbolic),
        "mean_lifts": payload.get("mean_lifts", {}),
    }


def _metrics_from_coconut_nope(paths: List[Path]) -> Dict[str, object]:
    rows = []
    for p in paths:
        payload = _read_json(p)
        if payload is not None:
            rows.append(payload)
    base_acc = [float(x.get("summary", {}).get("base_acc", 0.0)) for x in rows]
    handoff_acc = [float(x.get("summary", {}).get("handoff_acc", 0.0)) for x in rows]
    return {
        "seeds_evaluated": len(rows),
        "base_acc": _safe_mean(base_acc),
        "handoff_acc": _safe_mean(handoff_acc),
        "handoff_lift": _safe_mean(handoff_acc) - _safe_mean(base_acc),
    }


def _write_summary_md(path: StoragePath, records: List[RunRecord]) -> None:
    by_id = {r.run_id: r for r in records}

    def _h_metric(r: Optional[RunRecord]) -> str:
        if r is None or not isinstance(r.metrics, dict):
            return ""
        return (
            f"base={float(r.metrics.get('base_acc', 0.0)):.3f}, "
            f"handoff={float(r.metrics.get('handoff_acc', 0.0)):.3f}, "
            f"lift={float(r.metrics.get('handoff_lift', 0.0)):+.3f}, "
            f"mean_step_cos={float(r.metrics.get('mean_step_cosine', 0.0)):.3f}"
        )

    lines: List[str] = []
    lines.append("# Coconut Fusion Ablation Matrix")
    lines.append("")
    lines.append("| id | name | status | return_code | key metric |")
    lines.append("|---|---|---|---:|---|")
    for r in records:
        key_metric = ""
        if isinstance(r.metrics, dict):
            if r.run_id in {"A", "B.1", "B.2", "C"}:
                key_metric = (
                    f"base_final={r.metrics.get('control_base_final_acc', 0.0):.3f}, "
                    f"adapter_final={r.metrics.get('enhanced_t2t_final_acc', r.metrics.get('rigid_adapter_final_acc', 0.0)):.3f}, "
                    f"handoff_final={r.metrics.get('coconut_handoff_final_acc', 0.0):.3f}"
                )
            elif r.run_id == "D":
                key_metric = (
                    f"base={r.metrics.get('base_acc', 0.0):.3f}, "
                    f"handoff={r.metrics.get('handoff_acc', 0.0):.3f}, "
                    f"lift={r.metrics.get('handoff_lift', 0.0):+.3f}"
                )
            elif r.run_id == "E":
                ml = r.metrics.get("mean_lifts", {})
                key_metric = f"handoff_final_lift={float(ml.get('handoff_final_answer', 0.0)):+.3f}"
        lines.append(f"| `{r.run_id}` | `{r.name}` | `{r.status}` | `{r.return_code}` | {key_metric} |")

    lines.append("")
    lines.append("## Run H Series")
    lines.append("| id | name | status | key metric |")
    lines.append("|---|---|---|---|")
    h1 = by_id.get("H1")
    h2 = by_id.get("H2")
    h3 = by_id.get("H3")
    h4 = by_id.get("H4")
    lines.append(
        f"| `H1` | `Multi-Vector Injection (Bandwidth)` | "
        f"`{h1.status if h1 is not None else 'pending'}` | {_h_metric(h1)} |"
    )
    lines.append(
        f"| `H2` | `Mid-Layer Injection (Depth)` | "
        f"`{h2.status if h2 is not None else 'pending'}` | {_h_metric(h2)} |"
    )
    lines.append(
        f"| `H3` | `SwiGLU Mid-Layer Bridge (Non-Linear Alignment)` | "
        f"`{h3.status if h3 is not None else 'pending'}` | {_h_metric(h3)} |"
    )
    lines.append(
        f"| `H4` | `Persistent SwiGLU Injection (Continuous Anchor)` | "
        f"`{h4.status if h4 is not None else 'pending'}` | {_h_metric(h4)} |"
    )
    lines.append("")
    lines.append("- `Shock Tracking`: log per-step cosine for injected states (`step_cosine`) to observe persistence vs evaporation.")
    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formal A-G ablation matrix for Coconut Fusion.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True, help="Legacy text-to-text adapter (Run B.1/C).")
    p.add_argument(
        "--b2-adapter",
        type=Path,
        default=None,
        help="Enhanced-constraint text-to-text adapter for Run B.2 (no handoff path).",
    )
    p.add_argument("--drope-adapter", type=Path, default=None, help="NoPE-recalibrated adapter for Run D.")
    p.add_argument("--handoff-projection", type=Path, default=None, help="Projection checkpoint for Run E.")
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--output-root", type=str, default="runs/ablation/a_to_g")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("A-G", args.output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = join_path(args.output_root, ts)
    s3_output = is_s3_uri(out_dir)
    local_out_dir = Path(out_dir) if not s3_output else Path("runs/_staging_ablation_a_to_g") / ts
    make_dirs(local_out_dir, parents=True, exist_ok=True)

    eval_handoff = Path(__file__).resolve().parent / "eval_hf_dual_mode_handoff.py"
    eval_nope = Path(__file__).resolve().parent / "coconut_handoff.py"

    records: List[RunRecord] = []

    # A/B.1/C from one comparable run.
    abc_json = local_out_dir / "run_abc_dual_handoff.json"
    abc_cmd = [
        sys.executable,
        str(eval_handoff),
        "--base-model",
        args.base_model,
        "--adapter",
        str(args.adapter),
        "--sample-size",
        str(args.sample_size),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--dataset-size",
        str(args.dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output",
        str(abc_json),
    ]
    if args.local_files_only:
        abc_cmd.append("--local-files-only")
    abc_status, abc_rc = _run(abc_cmd, args.execute)
    abc_output = _copy_to_output_if_s3(abc_json, out_dir) if args.execute and abc_status == "ok" else str(abc_json)
    abc_metrics = _metrics_from_dual_handoff(abc_json) if abc_status == "ok" else None
    for rid, name, note in [
        ("A", "Control (English CoT -> English)", "Derived from base_acc in dual-handoff run."),
        (
            "B.1",
            "Legacy Text-to-Text (No Handoff)",
            "Derived from adapter_acc in dual-handoff run using --adapter.",
        ),
        ("C", "Coconut Fusion (Latent KV Handoff)", "Derived from handoff_acc in dual-handoff run."),
    ]:
        records.append(
            RunRecord(
                run_id=rid,
                name=name,
                status=abc_status,
                return_code=abc_rc,
                command=abc_cmd,
                output=abc_output,
                metrics=abc_metrics,
                notes=note,
            )
        )

    # B.2: enhanced constraints, text-to-text only (no handoff architecture).
    if args.b2_adapter is None:
        records.append(
            RunRecord(
                run_id="B.2",
                name="Enhanced Constraint Text-to-Text (No Handoff)",
                status="skipped",
                return_code=None,
                command=[],
                output=None,
                metrics=None,
                notes="Skipped: pass --b2-adapter to evaluate enhanced loss/constraint text-to-text path.",
            )
        )
    else:
        b2_json = local_out_dir / "run_b2_dual_handoff.json"
        b2_cmd = [
            sys.executable,
            str(eval_handoff),
            "--base-model",
            args.base_model,
            "--adapter",
            str(args.b2_adapter),
            "--sample-size",
            str(args.sample_size),
            "--seeds",
            *[str(s) for s in args.seeds],
            "--dataset-size",
            str(args.dataset_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(b2_json),
        ]
        if args.local_files_only:
            b2_cmd.append("--local-files-only")
        b2_status, b2_rc = _run(b2_cmd, args.execute)
        b2_output = _copy_to_output_if_s3(b2_json, out_dir) if args.execute and b2_status == "ok" else str(b2_json)
        b2_metrics = _metrics_from_dual_handoff(b2_json) if b2_status == "ok" else None
        records.append(
            RunRecord(
                run_id="B.2",
                name="Enhanced Constraint Text-to-Text (No Handoff)",
                status=b2_status,
                return_code=b2_rc,
                command=b2_cmd,
                output=b2_output,
                metrics=b2_metrics,
                notes="Derived from adapter_acc using --b2-adapter (constraint/loss enhancements only, no handoff).",
            )
        )

    # D: NoPE Fusion.
    nope_adapter = args.drope_adapter if args.drope_adapter is not None else args.adapter
    d_outputs: List[Path] = []
    d_status = "planned"
    d_rc = None
    if args.execute:
        d_status = "ok"
        for seed in args.seeds:
            out = local_out_dir / f"run_d_nope_seed{seed}.json"
            d_outputs.append(out)
            cmd = [
                sys.executable,
                str(eval_nope),
                "--base-model",
                args.base_model,
                "--adapter",
                str(nope_adapter),
                "--sample-size",
                str(args.sample_size),
                "--seed",
                str(seed),
                "--dataset-size",
                str(args.dataset_size),
                "--max-logic-new-tokens",
                str(args.max_new_tokens),
                "--max-final-new-tokens",
                str(args.max_new_tokens),
                "--output",
                str(out),
            ]
            if args.local_files_only:
                cmd.append("--local-files-only")
            rc = int(subprocess.call(cmd))
            d_rc = rc
            if rc != 0:
                d_status = "failed"
                break
            if s3_output:
                _copy_to_output_if_s3(out, out_dir)
    d_metrics = _metrics_from_coconut_nope(d_outputs) if d_status == "ok" else None
    d_output = (
        str(join_path(out_dir, "run_d_nope_seed*.json")) if s3_output else str(local_out_dir / "run_d_nope_seed*.json")
    )
    records.append(
        RunRecord(
            run_id="D",
            name="NoPE Fusion (DroPE + latent handoff)",
            status=d_status,
            return_code=d_rc,
            command=[
                sys.executable,
                str(eval_nope),
                "--base-model",
                args.base_model,
                "--adapter",
                str(nope_adapter),
                "--sample-size",
                str(args.sample_size),
                "--seed",
                "<each seed>",
            ],
            output=d_output,
            metrics=d_metrics,
            notes="Uses coconut_handoff.py (NoPE patch active).",
        )
    )

    # E: Babel Bridge (projection).
    e_json = local_out_dir / "run_e_babel_projection.json"
    if args.handoff_projection is None:
        e_status, e_rc, e_metrics = "skipped", None, None
        e_cmd = []
        e_note = "Skipped: pass --handoff-projection to execute Run E."
    else:
        e_cmd = [
            sys.executable,
            str(eval_handoff),
            "--base-model",
            args.base_model,
            "--adapter",
            str(args.adapter),
            "--sample-size",
            str(args.sample_size),
            "--seeds",
            *[str(s) for s in args.seeds],
            "--dataset-size",
            str(args.dataset_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--handoff-projection",
            str(args.handoff_projection),
            "--output",
            str(e_json),
        ]
        if args.local_files_only:
            e_cmd.append("--local-files-only")
        e_status, e_rc = _run(e_cmd, args.execute)
        e_output = _copy_to_output_if_s3(e_json, out_dir) if args.execute and e_status == "ok" else str(e_json)
        e_metrics = _metrics_from_dual_handoff(e_json) if e_status == "ok" else None
        e_note = "Projection applied to KV cache before adapter-off decode."
    if args.handoff_projection is None:
        e_output = None
    records.append(
        RunRecord(
            run_id="E",
            name="Babel Bridge (Projected latent handoff)",
            status=e_status,
            return_code=e_rc,
            command=e_cmd,
            output=e_output,
            metrics=e_metrics,
            notes=e_note,
        )
    )

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("A-G", "benchmark_ablation", "scripts/run_coconut_ablation_matrix.py"),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "b2_adapter": str(args.b2_adapter) if args.b2_adapter is not None else None,
        "drope_adapter": str(nope_adapter),
        "handoff_projection": str(args.handoff_projection) if args.handoff_projection is not None else None,
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "dataset_size": args.dataset_size,
        "max_new_tokens": args.max_new_tokens,
        "execute": bool(args.execute),
        "runs": [r.__dict__ for r in records],
    }
    manifest_path = join_path(out_dir, "ablation_matrix.json")
    summary_path = join_path(out_dir, "ablation_matrix.md")
    write_json(manifest_path, manifest, indent=2)
    _write_summary_md(summary_path, records)
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
