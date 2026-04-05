from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _ci95(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    m = _mean(xs)
    if len(xs) == 1:
        return {"mean": m, "low": m, "high": m, "n": 1}
    se = _std(xs) / math.sqrt(len(xs))
    delta = 1.96 * se
    return {"mean": m, "low": m - delta, "high": m + delta, "n": len(xs)}


def _task_type(question: str) -> str:
    q = question.lower()
    if "knave" in q or "knight" in q:
        return "knights_knaves"
    if "because they" in q:
        return "winograd_pronoun"
    if "where does" in q and "ball" in q:
        return "belief_tracking"
    return "other"


def _extract_run_abc_metrics(run_abc: dict) -> dict:
    per_seed = run_abc.get("per_seed", [])
    base_final = [float(x.get("final_answer", {}).get("base_acc", 0.0)) for x in per_seed]
    adapter_final = [float(x.get("final_answer", {}).get("adapter_acc", 0.0)) for x in per_seed]
    handoff_final = [float(x.get("final_answer", {}).get("handoff_acc", 0.0)) for x in per_seed]
    base_sym = [float(x.get("symbolic", {}).get("base_acc", 0.0)) for x in per_seed]
    adapter_sym = [float(x.get("symbolic", {}).get("adapter_acc", 0.0)) for x in per_seed]
    handoff_sym = [float(x.get("symbolic", {}).get("handoff_acc", 0.0)) for x in per_seed]
    return {
        "final_answer": {
            "control_base": _ci95(base_final),
            "rigid_text_to_text": _ci95(adapter_final),
            "coconut_handoff": _ci95(handoff_final),
            "lift_adapter_minus_base": _ci95([a - b for a, b in zip(adapter_final, base_final)]),
            "lift_handoff_minus_base": _ci95([h - b for h, b in zip(handoff_final, base_final)]),
        },
        "symbolic": {
            "control_base": _ci95(base_sym),
            "rigid_text_to_text": _ci95(adapter_sym),
            "coconut_handoff": _ci95(handoff_sym),
            "lift_adapter_minus_base": _ci95([a - b for a, b in zip(adapter_sym, base_sym)]),
            "lift_handoff_minus_base": _ci95([h - b for h, b in zip(handoff_sym, base_sym)]),
        },
        "per_seed": per_seed,
    }


def _extract_run_e_metrics(run_e: dict) -> dict:
    per_seed = run_e.get("per_seed", [])
    base_final = [float(x.get("final_answer", {}).get("base_acc", 0.0)) for x in per_seed]
    handoff_final = [float(x.get("final_answer", {}).get("handoff_acc", 0.0)) for x in per_seed]
    handoff_sym = [float(x.get("symbolic", {}).get("handoff_acc", 0.0)) for x in per_seed]
    return {
        "final_answer": {
            "babel_handoff": _ci95(handoff_final),
            "lift_babel_minus_base": _ci95([h - b for h, b in zip(handoff_final, base_final)]),
        },
        "symbolic": {
            "babel_handoff": _ci95(handoff_sym),
        },
        "gate_pass_handoff": bool(run_e.get("gate_pass_handoff", False)),
        "projection": run_e.get("handoff_projection"),
    }


def _extract_run_f_metrics(run_f: dict) -> dict:
    rows = run_f.get("rows", [])
    by_type = defaultdict(lambda: {"n": 0, "base_ok": 0, "run_f_ok": 0})
    for r in rows:
        t = _task_type(str(r.get("question", "")))
        by_type[t]["n"] += 1
        by_type[t]["base_ok"] += int(bool(r.get("base_ok", False)))
        by_type[t]["run_f_ok"] += int(bool(r.get("run_f_ok", False)))
    by_type_out = {}
    for t, v in by_type.items():
        n = max(1, int(v["n"]))
        by_type_out[t] = {
            "n": int(v["n"]),
            "base_acc": v["base_ok"] / n,
            "run_f_acc": v["run_f_ok"] / n,
            "lift": (v["run_f_ok"] - v["base_ok"]) / n,
        }
    return {
        "overall": {
            "mean_base_acc": float(run_f.get("mean_base_acc", 0.0)),
            "mean_run_f_acc": float(run_f.get("mean_run_f_acc", 0.0)),
            "mean_run_f_lift": float(run_f.get("mean_run_f_lift", 0.0)),
        },
        "per_seed": run_f.get("per_seed", []),
        "by_task_type": by_type_out,
    }


def _extract_shock_metrics(shock: dict) -> dict:
    mean_cos = float(shock.get("mean_cosine_similarity", 0.0))
    layers = [float(x) for x in shock.get("mean_layer_cosine_similarity", [])]
    early = _mean(layers[:8]) if layers else 0.0
    mid = _mean(layers[8:16]) if len(layers) > 8 else 0.0
    late = _mean(layers[16:]) if len(layers) > 16 else 0.0
    return {
        "mean_cosine_similarity": mean_cos,
        "disconnect_verdict": shock.get("disconnect_verdict"),
        "early_layer_mean_cos": early,
        "mid_layer_mean_cos": mid,
        "late_layer_mean_cos": late,
        "layer_count": len(layers),
    }


def _write_md(path: Path, pack: dict) -> None:
    tr = pack["trinity"]
    lines = [
        "# Science Metrics Pack",
        "",
        f"- generated_at: `{pack['generated_at']}`",
        f"- run_dir: `{pack['run_dir']}`",
        "",
        "## A/B/C",
        f"- final base mean: `{tr['run_abc']['final_answer']['control_base']['mean']:.3f}`",
        f"- final rigid mean: `{tr['run_abc']['final_answer']['rigid_text_to_text']['mean']:.3f}`",
        f"- final handoff mean: `{tr['run_abc']['final_answer']['coconut_handoff']['mean']:.3f}`",
        "",
        "## E (Babel)",
        f"- final babel mean: `{tr['run_e']['final_answer']['babel_handoff']['mean']:.3f}`",
        f"- final lift vs base: `{tr['run_e']['final_answer']['lift_babel_minus_base']['mean']:+.3f}`",
        f"- gate pass: `{tr['run_e']['gate_pass_handoff']}`",
        "",
        "## F (Self-Correct)",
        f"- mean run_f acc: `{tr['run_f']['overall']['mean_run_f_acc']:.3f}`",
        f"- mean run_f lift: `{tr['run_f']['overall']['mean_run_f_lift']:+.3f}`",
        "",
        "## Shock",
        f"- mean cosine: `{tr['shock']['mean_cosine_similarity']:.4f}`",
        f"- verdict: `{tr['shock']['disconnect_verdict']}`",
        f"- early/mid/late: `{tr['shock']['early_layer_mean_cos']:.3f}` / `{tr['shock']['mid_layer_mean_cos']:.3f}` / `{tr['shock']['late_layer_mean_cos']:.3f}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build expanded metrics pack for science bots.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_abc = _read_json(run_dir / "run_abc_dual_handoff.json")
    run_e = _read_json(run_dir / "run_e_babel_trained.json")
    run_f = _read_json(run_dir / "run_f_self_correct.json")
    shock = _read_json(run_dir / "shock_analysis.json")

    pack = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "trinity": {
            "run_abc": _extract_run_abc_metrics(run_abc),
            "run_e": _extract_run_e_metrics(run_e),
            "run_f": _extract_run_f_metrics(run_f),
            "shock": _extract_shock_metrics(shock),
        },
    }

    output_json = args.output_json or (run_dir / "science_metrics_pack.json")
    output_md = args.output_md or (run_dir / "science_metrics_pack.md")
    output_json.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    _write_md(output_md, pack)
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()
