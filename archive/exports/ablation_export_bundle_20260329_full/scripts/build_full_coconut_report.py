from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_h3_scale(run_obj: dict, source: Optional[str] = None) -> Optional[float]:
    cmd = run_obj.get("command", [])
    if not isinstance(cmd, list):
        cmd = []
    for i, tok in enumerate(cmd):
        if tok in {"--h3-layer-scale", "--mid-layer-scale"} and i + 1 < len(cmd):
            try:
                return float(cmd[i + 1])
            except Exception:
                pass
    if source:
        parts = source.replace("\\", "/").split("/")
        for p in parts:
            if p.startswith("scale_"):
                try:
                    return float(p.replace("scale_", ""))
                except Exception:
                    return None
    return None


def _extract_bridge_path(run_obj: dict) -> Optional[str]:
    cmd = run_obj.get("command", [])
    if not isinstance(cmd, list):
        return None
    for i, tok in enumerate(cmd):
        if tok == "--swiglu-bridge" and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return None


def _load_h_series(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    payload = _read_json(path)
    if payload is None:
        return out
    for r in payload.get("runs", []):
        rid = str(r.get("run_id", "")).upper()
        out[rid] = r
    return out


def _collect_sweep(root: Path) -> List[dict]:
    by_key: Dict[tuple[str, Optional[float]], dict] = {}
    if not root.exists():
        return []
    for p in sorted(root.rglob("run_h_series.json")):
        payload = _read_json(p)
        if payload is None:
            continue
        for r in payload.get("runs", []):
            if str(r.get("run_id", "")).upper() != "H3":
                continue
            m = r.get("metrics") or {}
            source = str(p)
            scale = _extract_h3_scale(r, source=source)
            bridge = _extract_bridge_path(r)
            expansion = 2
            if bridge and "exp4" in bridge.lower():
                expansion = 4
            row = {
                "source": source,
                "scale": scale,
                "bridge": bridge,
                "expansion_factor": expansion,
                "base_acc": _safe(m.get("base_acc")),
                "handoff_acc": _safe(m.get("handoff_acc")),
                "lift": _safe(m.get("handoff_lift")),
                "mean_step_cosine": _safe(m.get("mean_step_cosine")),
                "errors": _safe(m.get("errors")),
            }
            key = (str(bridge), scale)
            # Keep latest source path if duplicates exist for same config.
            prev = by_key.get(key)
            if prev is None or str(source) > str(prev.get("source", "")):
                by_key[key] = row
    rows = list(by_key.values())
    rows.sort(key=lambda x: (x["lift"], x["handoff_acc"], -x["expansion_factor"]), reverse=True)
    return rows


def _write_md(path: Path, report: dict) -> None:
    lines: List[str] = []
    lines.append("# Coconut Comprehensive Report")
    lines.append("")
    lines.append(f"- generated_at: `{report['generated_at']}`")
    lines.append("")
    lines.append("## Core Results")
    core = report["core"]
    lines.append(f"- Run A base final: `{core['run_a_base_final']:.3f}`")
    lines.append(f"- Run B rigid final: `{core['run_b_rigid_final']:.3f}`")
    lines.append(f"- Run C kv handoff final: `{core['run_c_kv_final']:.3f}`")
    lines.append(f"- Run E babel final: `{core['run_e_babel_final']:.3f}`")
    lines.append(f"- Run F self-correct final: `{core['run_f_final']:.3f}`")
    lines.append(f"- Run G true-coconut final: `{core['run_g_true_coconut_final']:.3f}`")
    lines.append("")
    lines.append("## H-Series")
    h = report["h_series"]
    for rid in ["H1", "H2", "H3", "H4"]:
        row = h.get(rid)
        if row is None:
            continue
        m = row.get("metrics") or {}
        lines.append(
            f"- {rid}: handoff=`{_safe(m.get('handoff_acc')):.3f}`, "
            f"lift=`{_safe(m.get('handoff_lift')):+.3f}`, "
            f"mean_step_cos=`{_safe(m.get('mean_step_cosine')):.3f}`"
        )
    lines.append("")
    lines.append("## H3 Sweep")
    lines.append("| rank | exp | scale | bridge | handoff | lift | step_cos |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|")
    for i, r in enumerate(report["h3_sweep"], start=1):
        bridge = r.get("bridge") or ""
        lines.append(
            f"| {i} | {r.get('expansion_factor')} | {r.get('scale')} | `{bridge}` | {r['handoff_acc']:.3f} | {r['lift']:+.3f} | {r['mean_step_cosine']:.3f} |"
        )
    lines.append("")
    ctrl = report.get("english_control_duel")
    if isinstance(ctrl, dict):
        lines.append("## English Control Duel")
        lines.append(f"- base_acc: `{_safe(ctrl.get('base_acc')):.3f}`")
        lines.append(f"- english_cot_acc: `{_safe(ctrl.get('english_cot_acc')):.3f}`")
        lines.append(f"- lojban_adapter_acc: `{_safe(ctrl.get('lojban_adapter_acc')):.3f}`")
        lines.append(f"- english_minus_lojban: `{_safe(ctrl.get('english_minus_lojban')):+.3f}`")
        lines.append("")

    vq = report.get("vq_pilot")
    if isinstance(vq, dict):
        lines.append("## VQ Pilot")
        lines.append(f"- codebook_size: `{vq.get('codebook_size')}`")
        lines.append(f"- codes_used: `{vq.get('codes_used')}` (ratio `{_safe(vq.get('usage_ratio')):.3f}`)")
        lines.append(f"- train_steps: `{vq.get('train_steps')}`")
        lines.append(f"- loss start/end: `{_safe(vq.get('loss_start')):.3f}` -> `{_safe(vq.get('loss_end')):.3f}`")
        lines.append("")

    sw_runs = report.get("swiglu_train_runs") or []
    if sw_runs:
        lines.append("## SwiGLU Train")
        lines.append("| exp | train_examples | train_steps | loss start -> end | bridge |")
        lines.append("|---:|---:|---:|---|---|")
        for sw in sw_runs:
            lh = sw.get("loss_history") or []
            ls = ""
            if lh:
                ls = f"{float(lh[0]):.3f} -> {float(lh[-1]):.3f}"
            lines.append(
                f"| {sw.get('expansion_factor')} | {sw.get('train_examples')} | {sw.get('train_steps')} | "
                f"`{ls}` | `{sw.get('bridge_path')}` |"
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append(
        f"- Best final accuracy remains Run B (`{core['run_b_rigid_final']:.3f}`) and Run F (`{core['run_f_final']:.3f}`) among recovery paths."
    )
    lines.append(
        "- Mid-layer transport preserves geometry (high step-cos), but semantic alignment remains unresolved in final-answer decoding."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build comprehensive Coconut report from gathered artifacts.")
    p.add_argument(
        "--science-pack",
        type=Path,
        default=Path("runs/ablation/a_to_g/20260226_090029/science_metrics_pack.json"),
    )
    p.add_argument(
        "--run-g-summary",
        type=Path,
        default=Path("runs/true_coconut_run_g_summary_s24_s7s11.json"),
    )
    p.add_argument(
        "--h-series-files",
        type=Path,
        nargs="+",
        default=[
            Path("runs/h_series/20260228_031421/run_h_series.json"),
            Path("runs/h_series/20260228_190640/run_h_series.json"),
            Path("runs/h_series/20260301_132657/run_h_series.json"),
        ],
    )
    p.add_argument("--h3-sweep-root", type=Path, required=True)
    p.add_argument(
        "--swiglu-reports",
        type=Path,
        nargs="+",
        default=[Path("runs/swiglu_bridge_report_h3.json"), Path("runs/swiglu_bridge_report_h3_exp4.json")],
    )
    p.add_argument(
        "--english-control-comparison",
        type=Path,
        default=Path("runs/english_cot_control_duel/20260301_205229/control_duel_comparison.json"),
    )
    p.add_argument(
        "--vq-pilot-report",
        type=Path,
        default=Path("runs/vq_reasoning_pilot_report_k200.json"),
    )
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    science = _read_json(args.science_pack) or {}
    run_g = _read_json(args.run_g_summary) or {}
    h_base: Dict[str, dict] = {}
    for p in args.h_series_files:
        for k, v in _load_h_series(p).items():
            h_base[k] = v
    sweep = _collect_sweep(args.h3_sweep_root)
    swiglu_runs = []
    for p in args.swiglu_reports:
        j = _read_json(p)
        if j is not None:
            swiglu_runs.append(j)
    swiglu_runs.sort(key=lambda x: int(x.get("expansion_factor", 0)))
    english_control = _read_json(args.english_control_comparison)
    vq_pilot = _read_json(args.vq_pilot_report)

    core = {
        "run_a_base_final": _safe(science.get("trinity", {}).get("run_abc", {}).get("final_answer", {}).get("control_base", {}).get("mean")),
        "run_b_rigid_final": _safe(science.get("trinity", {}).get("run_abc", {}).get("final_answer", {}).get("rigid_text_to_text", {}).get("mean")),
        "run_c_kv_final": _safe(science.get("trinity", {}).get("run_abc", {}).get("final_answer", {}).get("coconut_handoff", {}).get("mean")),
        "run_e_babel_final": _safe(science.get("trinity", {}).get("run_e", {}).get("final_answer", {}).get("babel_handoff", {}).get("mean")),
        "run_f_final": _safe(science.get("trinity", {}).get("run_f", {}).get("overall", {}).get("mean_run_f_acc")),
        "run_g_true_coconut_final": _safe(run_g.get("aggregate", {}).get("mean_coconut_acc")),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "science_pack": str(args.science_pack),
            "run_g_summary": str(args.run_g_summary),
            "h_series_files": [str(p) for p in args.h_series_files],
            "h3_sweep_root": str(args.h3_sweep_root),
            "swiglu_reports": [str(p) for p in args.swiglu_reports],
            "english_control_comparison": str(args.english_control_comparison),
            "vq_pilot_report": str(args.vq_pilot_report),
        },
        "core": core,
        "h_series": h_base,
        "h3_sweep": sweep,
        "swiglu_train_runs": swiglu_runs,
        "english_control_duel": english_control,
        "vq_pilot": vq_pilot,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(args.output_md, report)
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_md}")


if __name__ == "__main__":
    main()
