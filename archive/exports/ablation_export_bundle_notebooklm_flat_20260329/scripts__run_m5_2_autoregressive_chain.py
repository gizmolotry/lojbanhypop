from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run isolated M5.2 autoregressive matrix-chain test.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=4)
    p.add_argument("--semantic-dataset-size", type=int, default=48)
    p.add_argument("--winograd-pack-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-profile", type=str, default="semantic_bench_v1")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--max-chain-steps", type=int, default=4)
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m5_2_autoregressive_chain"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m5_2_autoregressive_chain"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.l_output_root)
    assert_output_path_allowed("M", args.report_output_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    l_run_root = args.l_output_root / ts / "m5_2_smoke"
    report_root = args.report_output_root / f"m5_2_{ts}"
    validate_series_outputs("M", [args.l_output_root, args.report_output_root], [l_run_root, report_root])
    l_run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    trainer = Path(__file__).resolve().parent / "train_m5_2_autoregressive_chain.py"
    cmd = [
        sys.executable,
        str(trainer),
        "--base-model",
        str(args.base_model),
        "--adapter",
        str(args.adapter),
        "--output-root",
        str(l_run_root),
        "--train-steps",
        str(int(args.train_steps)),
        "--semantic-dataset-size",
        str(int(args.semantic_dataset_size)),
        "--winograd-pack-size",
        str(int(args.winograd_pack_size)),
        "--seed",
        str(int(args.seed)),
        "--dataset-profile",
        str(args.dataset_profile),
        "--difficulty-tier",
        str(args.difficulty_tier),
        "--layer-index",
        str(int(args.layer_index)),
        "--max-chain-steps",
        str(int(args.max_chain_steps)),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")

    rc = int(subprocess.run(cmd, check=False).returncode)
    latest = _latest_child_dir(l_run_root)
    summary_path = latest / "m5_2_autoregressive_chain_summary.json" if latest is not None else None
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M5.2.autoregressive_chain.run", "scripts/run_m5_2_autoregressive_chain.py"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "status": "failed",
        "return_code": rc,
    }
    if rc == 0 and summary_path is not None and summary_path.exists():
        payload["status"] = "ok"
        payload["summary_path"] = str(summary_path).replace("\\", "/")
        payload["summary"] = _read_json(summary_path)

    report_json = report_root / "m5_2_autoregressive_chain_report.json"
    report_md = report_root / "m5_2_autoregressive_chain_report.md"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# M5.2 Autoregressive Chain",
        "",
        f"- status: {payload['status']}",
        f"- return_code: {payload['return_code']}",
    ]
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    final_step = summary.get("final_step", {}) if isinstance(summary.get("final_step"), dict) else {}
    eval_block = summary.get("eval", {}) if isinstance(summary.get("eval"), dict) else {}
    if final_step:
        lines.extend(
            [
                "",
                "## Final Step",
                f"- mean_chain_length: {final_step.get('mean_chain_length', 'n/a')}",
                f"- mean_stop_prob: {final_step.get('mean_stop_prob', 'n/a')}",
                f"- slot_usage_mean: {final_step.get('slot_usage_mean', 'n/a')}",
                f"- operator_entropy_batch: {final_step.get('operator_entropy_batch', 'n/a')}",
                f"- top1_op_share_batch: {final_step.get('top1_op_share_batch', 'n/a')}",
                f"- winograd_accuracy_batch: {final_step.get('winograd_accuracy_batch', 'n/a')}",
            ]
        )
    if eval_block:
        lines.extend(
            [
                "",
                "## Eval",
                f"- winograd_val: {eval_block.get('winograd_val', {})}",
                f"- winograd_eval: {eval_block.get('winograd_eval', {})}",
                f"- semantic_val: {eval_block.get('semantic_val', {})}",
            ]
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_json}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
