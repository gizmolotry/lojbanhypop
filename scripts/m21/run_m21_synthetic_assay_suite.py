from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_m21_dynamic_bridi_suite import parse_args, run_suite  # noqa: E402
from lojban_evolution.m21.family import M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import validate_series_outputs  # noqa: E402


if __name__ == "__main__":
    args = parse_args()
    registry = M21_REGISTRY["M21"]
    if Path(args.output_root).as_posix() == Path(registry["output_roots"]["suite"]).as_posix():
        args.output_root = Path(registry["output_roots"]["synthetic_assay"])
    payload = run_suite(args)
    payload["series"]["track"] = "M21.1.synthetic_assay"
    payload["registry"]["runner_script"] = registry["runner_scripts"]["synthetic_assay"]
    payload["registry"]["output_root"] = registry["output_roots"]["synthetic_assay"]
    report_path = Path(payload["run_dir"]) / registry["report_names"]["synthetic_assay"]
    validate_series_outputs("M", [registry["output_roots"]["synthetic_assay"], str(Path(payload["run_dir"]))], [report_path])
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M21 synthetic assay report written to {report_path}")
