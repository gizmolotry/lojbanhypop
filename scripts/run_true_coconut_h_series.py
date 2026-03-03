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
    output_files: List[str]
    metrics: Optional[Dict[str, float]]
    command: List[str]
    config: Dict[str, object]


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


def _aggregate_metrics(rows: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not rows:
        return None
    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))]
        if vals:
            out[key] = _safe_mean(vals)
    out["seed_count"] = float(len(rows))
    return out


def _extract_provenance_metrics(path: Path) -> Dict[str, float]:
    payload = _read_json(path) or {}
    pm = payload.get("provenance_metrics", {}) if isinstance(payload, dict) else {}
    return {
        "provenance_mean_l2_delta": float(pm.get("mean_l2_delta", 0.0)),
        "provenance_exact_match_ratio_eps": float(pm.get("exact_match_ratio_eps", 0.0)),
        "provenance_anchor_mean_l2_delta": float(pm.get("anchor_mean_l2_delta", 0.0)),
    }


def _extract_ood_metrics(path: Path) -> Dict[str, float]:
    payload = _read_json(path) or {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    domains = payload.get("domains", {}) if isinstance(payload, dict) else {}
    spatial = domains.get("spatial", {}) if isinstance(domains, dict) else {}
    temporal = domains.get("temporal", {}) if isinstance(domains, dict) else {}
    return {
        "ood_accuracy": float(summary.get("accuracy", 0.0)),
        "ood_spatial_accuracy": float(spatial.get("accuracy", 0.0)),
        "ood_temporal_accuracy": float(temporal.get("accuracy", 0.0)),
    }


def _extract_dptr_metrics(path: Path) -> Dict[str, float]:
    payload = _read_json(path) or {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    ptr = payload.get("dynamic_pointer_metrics", {}) if isinstance(payload, dict) else {}
    return {
        "dptr_standard_accuracy": float(summary.get("standard_accuracy", 0.0)),
        "dptr_dynamic_accuracy": float(summary.get("dynamic_accuracy", 0.0)),
        "dptr_dynamic_minus_standard_accuracy": float(summary.get("dynamic_minus_standard_accuracy", 0.0)),
        "dptr_self_ref_rate": float(ptr.get("self_ref_rate", 0.0)),
    }


def _run(cmd: List[str], execute: bool) -> tuple[str, Optional[int]]:
    if not execute:
        return "planned", None
    try:
        rc = int(subprocess.run(cmd, check=False).returncode)
    except Exception:
        return "failed", -999
    return ("ok", rc) if rc == 0 else ("failed", rc)


def _validate_json_artifact(path: Path, required_keys: List[str]) -> bool:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return False
    return all(k in payload for k in required_keys)


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
    p.add_argument("--h4-bridge", type=Path, default=None)
    p.add_argument("--h4-layer-index", type=int, default=12)
    p.add_argument("--h4-layer-scale", type=float, default=1.0)
    p.add_argument("--h4-contrastive-alpha", type=float, default=1.0)
    p.add_argument(
        "--only-runs",
        type=str,
        nargs="+",
        default=None,
        help="Subset of runs: H1 H2 H3 H4 H5-PROV H5-OOD H5-DPTR",
    )
    p.add_argument("--fail-fast", action="store_true", help="Stop immediately when any selected run fails.")

    p.add_argument("--h5-base-model", type=str, default=None)
    p.add_argument("--h5-adapter", type=Path, default=None)
    p.add_argument("--h5-checkpoint", type=Path, default=None, help="Default H5 checkpoint/bridge for H5 runs.")

    p.add_argument("--h5-prov-base-model", type=str, default=None)
    p.add_argument("--h5-prov-adapter", type=Path, default=None)
    p.add_argument("--h5-prov-checkpoint", type=Path, default=None, help="Alias for --h5-prov-h53-checkpoint.")
    p.add_argument("--h5-prov-h53-checkpoint", type=Path, default=None)
    p.add_argument("--h5-prov-slice1-checkpoint", type=Path, default=None)
    p.add_argument("--h5-prov-top-k", type=int, default=64)

    p.add_argument("--h5-ood-base-model", type=str, default=None)
    p.add_argument("--h5-ood-adapter", type=Path, default=None)
    p.add_argument("--h5-ood-checkpoint", type=Path, default=None)
    p.add_argument("--h5-ood-per-domain-limit", type=int, default=10)
    p.add_argument("--h5-ood-max-logic-new-tokens", type=int, default=None)
    p.add_argument("--h5-ood-max-final-new-tokens", type=int, default=None)
    p.add_argument("--h5-ood-layer-index", type=int, default=12)
    p.add_argument("--h5-ood-inject-scale", type=float, default=1.0)
    p.add_argument("--h5-ood-relation-bias", type=float, default=0.0)
    p.add_argument("--h5-ood-use-iron-collar", action="store_true")

    p.add_argument("--h5-dptr-base-model", type=str, default=None)
    p.add_argument("--h5-dptr-adapter", type=Path, default=None)
    p.add_argument("--h5-dptr-checkpoint", type=Path, default=None)
    p.add_argument("--h5-dptr-sample-size", type=int, default=8)
    p.add_argument("--h5-dptr-pointer-window", type=int, default=16)
    p.add_argument("--h5-dptr-layer-index", type=int, default=12)
    p.add_argument("--h5-dptr-inject-scale", type=float, default=1.0)
    p.add_argument("--h5-dptr-relation-bias", type=float, default=0.0)
    p.add_argument("--h5-dptr-use-iron-collar", action="store_true")

    p.add_argument("--output-root", type=Path, default=Path("runs/true_coconut_h_series"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    true_coconut = script_dir / "true_coconut.py"
    prov_script = script_dir / "trace_h5_provenance.py"
    ood_script = script_dir / "eval_h5_ood_stress.py"
    dptr_script = script_dir / "eval_h5_dynamic_pointer_refactor.py"

    runs: List[HRun] = []
    abort_remaining = False

    def _seed_cmd(
        seed: int,
        *,
        base_model: str,
        adapter: Path,
        sample_size: int,
        dataset_size: int,
        max_logic_new_tokens: int,
        max_final_new_tokens: int,
        mode: str,
        window: int,
        out: Path,
        log_step_cosine: bool,
        extra: Optional[List[str]] = None,
    ) -> List[str]:
        cmd = [
            sys.executable,
            str(true_coconut),
            "--base-model",
            base_model,
            "--adapter",
            str(adapter),
            "--sample-size",
            str(sample_size),
            "--seed",
            str(seed),
            "--dataset-size",
            str(dataset_size),
            "--max-logic-new-tokens",
            str(max_logic_new_tokens),
            "--max-final-new-tokens",
            str(max_final_new_tokens),
            "--virtual-token-window",
            str(window),
            "--injection-mode",
            mode,
            "--output",
            str(out),
        ]
        if log_step_cosine:
            cmd.append("--log-step-cosine")
        if args.local_files_only:
            cmd.append("--local-files-only")
        if extra:
            cmd.extend(extra)
        return cmd

    wanted = {r.upper() for r in (args.only_runs or ["H1", "H2", "H3", "H4", "H5-PROV", "H5-OOD", "H5-DPTR"])}
    run_specs = [
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
        (
            "H4",
            "Persistent SwiGLU Injection (Continuous Anchor)",
            "midlayer_persistent",
            1,
            args.h3_adapter if args.h3_adapter is not None else args.adapter,
            [
                "--mid-layer-index",
                str(args.h4_layer_index),
                "--mid-layer-scale",
                str(args.h4_layer_scale),
                "--swiglu-bridge",
                str(args.h4_bridge) if args.h4_bridge is not None else "",
                "--contrastive-alpha",
                str(args.h4_contrastive_alpha),
            ],
        ),
    ]
    for run_id, name, mode, window, adapter, extra in run_specs:
        if abort_remaining and args.fail_fast:
            break
        if run_id not in wanted:
            continue
        if run_id == "H3" and (args.h3_adapter is None or args.h3_bridge is None):
            runs.append(HRun(run_id, name, "skipped", None, None, [], None, [], {}))
            continue
        if run_id == "H4" and args.h4_bridge is None:
            runs.append(HRun(run_id, name, "skipped", None, None, [], None, [], {}))
            continue
        if run_id in {"H3", "H4"}:
            extra = [x for x in (extra or []) if x != ""]
        seed_metrics: List[Dict[str, float]] = []
        output_files: List[str] = []
        status = "ok" if args.execute else "planned"
        last_rc: Optional[int] = None
        first_cmd: List[str] = []
        run_base_model = args.base_model
        run_sample_size = args.sample_size
        run_dataset_size = args.dataset_size
        run_max_logic_new_tokens = args.max_logic_new_tokens
        run_max_final_new_tokens = args.max_final_new_tokens
        run_seeds = args.seeds
        for seed in run_seeds:
            out = out_dir / f"{run_id.lower()}_seed{seed}.json"
            output_files.append(str(out))
            cmd = _seed_cmd(
                seed,
                base_model=run_base_model,
                adapter=adapter,
                sample_size=run_sample_size,
                dataset_size=run_dataset_size,
                max_logic_new_tokens=run_max_logic_new_tokens,
                max_final_new_tokens=run_max_final_new_tokens,
                mode=mode,
                window=window,
                out=out,
                log_step_cosine=True,
                extra=extra,
            )
            if not first_cmd:
                first_cmd = cmd
            st, rc = _run(cmd, args.execute)
            last_rc = rc
            if st == "failed":
                status = st
                break
            if st == "planned":
                continue
            m = _extract_metrics(out)
            if m is not None:
                seed_metrics.append(m)
        agg = _aggregate_metrics(seed_metrics)
        run_config: Dict[str, object] = {
            "base_model": run_base_model,
            "adapter": str(adapter),
            "seeds": run_seeds,
            "sample_size": run_sample_size,
            "dataset_size": run_dataset_size,
            "max_logic_new_tokens": run_max_logic_new_tokens,
            "max_final_new_tokens": run_max_final_new_tokens,
            "injection_mode": mode,
            "virtual_token_window": window,
        }
        runs.append(
            HRun(
                run_id,
                name,
                status,
                last_rc,
                str(out_dir / f"{run_id.lower()}_seed*.json"),
                output_files,
                agg,
                first_cmd,
                run_config,
            )
        )
        if status == "failed" and args.fail_fast:
            abort_remaining = True
            break

    # Dedicated H5 extensions (single-run jobs integrated in the same manifest/report structure)
    h53_ckpt = args.h5_prov_h53_checkpoint or args.h5_prov_checkpoint or args.h5_checkpoint
    s1_ckpt = args.h5_prov_slice1_checkpoint
    if "H5-PROV" in wanted and not abort_remaining:
        out = out_dir / "h5-prov.json"
        if h53_ckpt is None or s1_ckpt is None:
            runs.append(HRun("H5-PROV", "Provenance Trace", "skipped", None, str(out), [str(out)], None, [], {"reason": "missing_h53_or_slice1_checkpoint"}))
        else:
            cmd = [sys.executable, str(prov_script), "--h53-checkpoint", str(h53_ckpt), "--slice1-checkpoint", str(s1_ckpt), "--top-k", str(args.h5_prov_top_k), "--output", str(out)]
            st, rc = _run(cmd, args.execute)
            if st == "ok" and not _validate_json_artifact(out, ["provenance_metrics", "provenance_map_topk"]):
                st, rc = "failed", -2
            metrics = _extract_provenance_metrics(out) if st == "ok" else None
            runs.append(HRun("H5-PROV", "Provenance Trace", st, rc, str(out), [str(out)], metrics, cmd, {"h53_checkpoint": str(h53_ckpt), "slice1_checkpoint": str(s1_ckpt), "top_k": args.h5_prov_top_k}))
            if st == "failed" and args.fail_fast:
                abort_remaining = True

    if "H5-OOD" in wanted and not abort_remaining:
        out = out_dir / "h5-ood.json"
        ood_base = args.h5_ood_base_model or args.h5_base_model or args.base_model
        ood_adapter = args.h5_ood_adapter or args.h5_adapter or args.adapter
        ood_ckpt = args.h5_ood_checkpoint or args.h5_checkpoint
        if ood_ckpt is None:
            runs.append(HRun("H5-OOD", "OOD Stress Test", "skipped", None, str(out), [str(out)], None, [], {"reason": "missing_h5_ood_checkpoint"}))
        else:
            cmd = [
                sys.executable, str(ood_script),
                "--base-model", str(ood_base),
                "--adapter", str(ood_adapter),
                "--checkpoint", str(ood_ckpt),
                "--per-domain-limit", str(args.h5_ood_per_domain_limit),
                "--max-logic-new-tokens", str(args.h5_ood_max_logic_new_tokens if args.h5_ood_max_logic_new_tokens is not None else args.max_logic_new_tokens),
                "--max-final-new-tokens", str(args.h5_ood_max_final_new_tokens if args.h5_ood_max_final_new_tokens is not None else args.max_final_new_tokens),
                "--layer-index", str(args.h5_ood_layer_index),
                "--inject-scale", str(args.h5_ood_inject_scale),
                "--relation-bias", str(args.h5_ood_relation_bias),
                "--output", str(out),
            ]
            if args.h5_ood_use_iron_collar:
                cmd.append("--use-iron-collar")
            if args.local_files_only:
                cmd.append("--local-files-only")
            st, rc = _run(cmd, args.execute)
            if st == "ok" and not _validate_json_artifact(out, ["summary", "domains", "samples"]):
                st, rc = "failed", -2
            metrics = _extract_ood_metrics(out) if st == "ok" else None
            runs.append(HRun("H5-OOD", "OOD Stress Test", st, rc, str(out), [str(out)], metrics, cmd, {"base_model": str(ood_base), "adapter": str(ood_adapter), "checkpoint": str(ood_ckpt)}))
            if st == "failed" and args.fail_fast:
                abort_remaining = True

    if "H5-DPTR" in wanted and not abort_remaining:
        out = out_dir / "h5-dptr.json"
        dptr_base = args.h5_dptr_base_model or args.h5_base_model or args.base_model
        dptr_adapter = args.h5_dptr_adapter or args.h5_adapter or args.adapter
        dptr_ckpt = args.h5_dptr_checkpoint or args.h5_checkpoint
        if dptr_ckpt is None:
            runs.append(HRun("H5-DPTR", "Dynamic Pointer Refactor Eval", "skipped", None, str(out), [str(out)], None, [], {"reason": "missing_h5_dptr_checkpoint"}))
        else:
            cmd = [
                sys.executable, str(dptr_script),
                "--base-model", str(dptr_base),
                "--adapter", str(dptr_adapter),
                "--checkpoint", str(dptr_ckpt),
                "--sample-size", str(args.h5_dptr_sample_size),
                "--pointer-window", str(args.h5_dptr_pointer_window),
                "--max-logic-new-tokens", str(args.max_logic_new_tokens),
                "--max-final-new-tokens", str(args.max_final_new_tokens),
                "--layer-index", str(args.h5_dptr_layer_index),
                "--inject-scale", str(args.h5_dptr_inject_scale),
                "--relation-bias", str(args.h5_dptr_relation_bias),
                "--output", str(out),
            ]
            if args.h5_dptr_use_iron_collar:
                cmd.append("--use-iron-collar")
            if args.local_files_only:
                cmd.append("--local-files-only")
            st, rc = _run(cmd, args.execute)
            if st == "ok" and not _validate_json_artifact(out, ["summary", "dynamic_pointer_metrics", "samples"]):
                st, rc = "failed", -2
            metrics = _extract_dptr_metrics(out) if st == "ok" else None
            runs.append(HRun("H5-DPTR", "Dynamic Pointer Refactor Eval", st, rc, str(out), [str(out)], metrics, cmd, {"base_model": str(dptr_base), "adapter": str(dptr_adapter), "checkpoint": str(dptr_ckpt)}))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "dataset_size": args.dataset_size,
        "execute": bool(args.execute),
        "h5": {
            "base_model": args.h5_base_model,
            "adapter": str(args.h5_adapter) if args.h5_adapter is not None else None,
            "checkpoint": str(args.h5_checkpoint) if args.h5_checkpoint is not None else None,
        },
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
            if r.run_id == "H5-PROV":
                key = (
                    f"mean_l2_delta={r.metrics.get('provenance_mean_l2_delta', 0.0):.3f}, "
                    f"exact_match={r.metrics.get('provenance_exact_match_ratio_eps', 0.0):.3f}"
                )
            elif r.run_id == "H5-OOD":
                key = (
                    f"ood_acc={r.metrics.get('ood_accuracy', 0.0):.3f}, "
                    f"spatial={r.metrics.get('ood_spatial_accuracy', 0.0):.3f}, "
                    f"temporal={r.metrics.get('ood_temporal_accuracy', 0.0):.3f}"
                )
            elif r.run_id == "H5-DPTR":
                key = (
                    f"dyn_acc={r.metrics.get('dptr_dynamic_accuracy', 0.0):.3f}, "
                    f"base_acc={r.metrics.get('dptr_standard_accuracy', 0.0):.3f}, "
                    f"delta={r.metrics.get('dptr_dynamic_minus_standard_accuracy', 0.0):+.3f}"
                )
            else:
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
