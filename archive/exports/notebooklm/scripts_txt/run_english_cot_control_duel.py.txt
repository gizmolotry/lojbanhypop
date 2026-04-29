from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _run(cmd: List[str], execute: bool) -> tuple[str, Optional[int]]:
    if not execute:
        return "planned", None
    rc = int(subprocess.call(cmd))
    return ("ok", rc) if rc == 0 else ("failed", rc)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apples-to-apples duel: English-CoT LoRA vs baseline (optionally vs Lojban LoRA).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/english_cot_control_duel"))
    p.add_argument("--dataset-size", type=int, default=1200)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-sample-size", type=int, default=24)
    p.add_argument("--eval-seed", type=int, default=7)
    p.add_argument("--eval-dataset-size", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--lojban-adapter", type=Path, default=None, help="Optional reference adapter for side-by-side eval.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    build_ds = Path(__file__).resolve().parent / "build_english_cot_control_dataset.py"
    train = Path(__file__).resolve().parent / "train_lora.py"
    eval_adapter = Path(__file__).resolve().parent / "eval_hf_adapter.py"

    ds_path = out_dir / "english_cot_control.jsonl"
    adapter_out = out_dir / "english_cot_adapter"
    eval_out = out_dir / "english_cot_eval.json"
    eval_lojban_out = out_dir / "lojban_reference_eval.json"

    build_cmd = [
        sys.executable,
        str(build_ds),
        "--output",
        str(ds_path),
        "--dataset-size",
        str(args.dataset_size),
        "--seeds",
        *[str(s) for s in args.seeds],
    ]
    train_cmd = [
        sys.executable,
        str(train),
        "--base-model",
        args.base_model,
        "--dataset",
        str(ds_path),
        "--output-dir",
        str(adapter_out),
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
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
    ]
    eval_cmd = [
        sys.executable,
        str(eval_adapter),
        "--base-model",
        args.base_model,
        "--adapter",
        str(adapter_out),
        "--sample-size",
        str(args.eval_sample_size),
        "--seed",
        str(args.eval_seed),
        "--dataset-size",
        str(args.eval_dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output",
        str(eval_out),
        "--prompt-style",
        "final_answer",
    ]
    if args.local_files_only:
        train_cmd.append("--local-files-only")
        eval_cmd.append("--local-files-only")

    manifest: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "build_dataset": {"cmd": build_cmd, "status": "planned", "return_code": None, "output": str(ds_path)},
        "train_english_cot": {"cmd": train_cmd, "status": "planned", "return_code": None, "output": str(adapter_out)},
        "eval_english_cot": {"cmd": eval_cmd, "status": "planned", "return_code": None, "output": str(eval_out)},
        "eval_lojban_reference": None,
        "comparison": None,
    }

    st, rc = _run(build_cmd, args.execute)
    manifest["build_dataset"]["status"] = st
    manifest["build_dataset"]["return_code"] = rc

    if st == "ok":
        st, rc = _run(train_cmd, args.execute)
        manifest["train_english_cot"]["status"] = st
        manifest["train_english_cot"]["return_code"] = rc

    if manifest["train_english_cot"]["status"] == "ok":
        st, rc = _run(eval_cmd, args.execute)
        manifest["eval_english_cot"]["status"] = st
        manifest["eval_english_cot"]["return_code"] = rc

    if args.lojban_adapter is not None:
        ref_cmd = [
            sys.executable,
            str(eval_adapter),
            "--base-model",
            args.base_model,
            "--adapter",
            str(args.lojban_adapter),
            "--sample-size",
            str(args.eval_sample_size),
            "--seed",
            str(args.eval_seed),
            "--dataset-size",
            str(args.eval_dataset_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(eval_lojban_out),
            "--prompt-style",
            "final_answer",
        ]
        if args.local_files_only:
            ref_cmd.append("--local-files-only")
        entry = {"cmd": ref_cmd, "status": "planned", "return_code": None, "output": str(eval_lojban_out)}
        st, rc = _run(ref_cmd, args.execute)
        entry["status"] = st
        entry["return_code"] = rc
        manifest["eval_lojban_reference"] = entry

    eng_eval = _read_json(eval_out)
    loj_eval = _read_json(eval_lojban_out) if args.lojban_adapter is not None else None
    if eng_eval is not None:
        base_acc = float(eng_eval.get("summary", {}).get("base", {}).get("accuracy", 0.0))
        eng_acc = float(eng_eval.get("summary", {}).get("adapter", {}).get("accuracy", 0.0))
        cmp = {
            "base_acc": base_acc,
            "english_cot_adapter_acc": eng_acc,
            "english_lift_vs_base": eng_acc - base_acc,
        }
        if loj_eval is not None:
            loj_acc = float(loj_eval.get("summary", {}).get("adapter", {}).get("accuracy", 0.0))
            cmp["lojban_adapter_acc"] = loj_acc
            cmp["english_minus_lojban"] = eng_acc - loj_acc
        manifest["comparison"] = cmp

    out = out_dir / "english_cot_control_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    if isinstance(manifest.get("comparison"), dict):
        c = manifest["comparison"]
        print(f"base_acc: {c.get('base_acc', 0.0):.3f}")
        print(f"english_cot_adapter_acc: {c.get('english_cot_adapter_acc', 0.0):.3f}")
        if "lojban_adapter_acc" in c:
            print(f"lojban_adapter_acc: {c.get('lojban_adapter_acc', 0.0):.3f}")


if __name__ == "__main__":
    main()
