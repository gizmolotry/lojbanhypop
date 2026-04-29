from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import sys as _sys

_sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
_sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lojban_evolution.m18_family import M18_FAMILY_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M18 controller family with explicit train and audit artifacts.")
    parser.add_argument("--base-model", default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path(M18_FAMILY_REGISTRY["M18"]["output_root"]))
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/models/m18/frontier"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    assert_output_path_allowed("M", args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_output_root / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    selector_path = model_dir / "salience_v0.pt"
    interpreter_u_path = model_dir / "interpreter_u_v0.pt"
    interpreter_l_path = model_dir / "interpreter_l_v0.pt"
    joint_u_dir = model_dir / "joint_u_v0"
    joint_l_dir = model_dir / "joint_l_v0"

    if not bool(args.skip_train):
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "m18" / "train_salience.py"),
                "--base-model",
                str(args.base_model),
                "--data-path",
                str(args.data_path),
                "--output-path",
                str(selector_path),
                "--epochs",
                str(int(args.epochs)),
            ],
            repo_root,
        )
        for ontology, interpreter_path in (("U", interpreter_u_path), ("L", interpreter_l_path)):
            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "m18" / "train_interpreter.py"),
                    "--base-model",
                    str(args.base_model),
                    "--data-path",
                    str(args.data_path),
                    "--selector-path",
                    str(selector_path),
                    "--ontology",
                    ontology,
                    "--output-path",
                    str(interpreter_path),
                    "--epochs",
                    str(int(args.epochs)),
                ],
                repo_root,
            )
        for ontology, interpreter_path, joint_dir in (("U", interpreter_u_path, joint_u_dir), ("L", interpreter_l_path, joint_l_dir)):
            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "m18" / "train_joint_controller.py"),
                    "--base-model",
                    str(args.base_model),
                    "--data-path",
                    str(args.data_path),
                    "--selector-path",
                    str(selector_path),
                    "--interpreter-path",
                    str(interpreter_path),
                    "--ontology",
                    ontology,
                    "--output-dir",
                    str(joint_dir),
                    "--epochs",
                    str(int(args.epochs)),
                ],
                repo_root,
            )

    sapir_path = run_dir / "sapir_whorf_audit_report.json"
    harmonized_path = run_dir / "harmonized_audit_report.json"
    hybrid_path = run_dir / "hybrid_cot_audit_report.json"
    for script_name, output_path in (
        ("run_sapir_whorf_audit.py", sapir_path),
        ("run_harmonized_audit.py", harmonized_path),
        ("run_hybrid_cot_audit.py", hybrid_path),
    ):
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "m18" / script_name),
                "--base-model",
                str(args.base_model),
                "--output-path",
                str(output_path),
            ],
            repo_root,
        )

    sapir = json.loads(sapir_path.read_text(encoding="utf-8"))
    harmonized = json.loads(harmonized_path.read_text(encoding="utf-8"))
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": "M18",
        "run_id": run_id,
        "series": {
            "series_id": "M",
            "track": "M18",
            "script": "scripts/m18/run_m18_controller_family.py",
        },
        "family_contract": {
            "family_name": M18_FAMILY_REGISTRY["M18"]["family"],
            "implementation_label": M18_FAMILY_REGISTRY["M18"]["implementation_label"],
            "runner_script": "scripts/m18/run_m18_controller_family.py",
            "dag": M18_FAMILY_REGISTRY["M18"]["dag"],
        },
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path),
            "epochs": int(args.epochs),
            "skip_train": bool(args.skip_train),
        },
        "artifact_paths": {
            "selector_path": str(selector_path).replace("\\", "/"),
            "interpreter_u_path": str(interpreter_u_path).replace("\\", "/"),
            "interpreter_l_path": str(interpreter_l_path).replace("\\", "/"),
            "joint_u_dir": str(joint_u_dir).replace("\\", "/"),
            "joint_l_dir": str(joint_l_dir).replace("\\", "/"),
            "sapir_whorf_report": str(sapir_path).replace("\\", "/"),
            "harmonized_report": str(harmonized_path).replace("\\", "/"),
            "hybrid_cot_report": str(hybrid_path).replace("\\", "/"),
        },
        "metrics": {
            "sapir_english_accuracy": sapir.get("metrics", {}).get("english_accuracy"),
            "sapir_chinese_accuracy": sapir.get("metrics", {}).get("chinese_accuracy"),
            "harmonized_en_concise_accuracy": harmonized.get("metrics", {}).get("en_concise_accuracy"),
            "harmonized_u_typed_accuracy": harmonized.get("metrics", {}).get("u_typed_accuracy"),
            "harmonized_l_typed_accuracy": harmonized.get("metrics", {}).get("l_typed_accuracy"),
            "hybrid_en_cot_accuracy": hybrid.get("metrics", {}).get("en_cot_accuracy"),
            "hybrid_u_typed_accuracy": hybrid.get("metrics", {}).get("u_typed_accuracy"),
            "hybrid_l_typed_accuracy": hybrid.get("metrics", {}).get("l_typed_accuracy"),
        },
    }
    manifest_path = run_dir / "m18_family_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"M18 controller family complete: {run_dir}")


if __name__ == "__main__":
    main()
