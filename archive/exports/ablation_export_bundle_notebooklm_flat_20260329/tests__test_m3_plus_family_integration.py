from __future__ import annotations

import json
from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_m3_plus_family.py"
_SPEC = importlib.util.spec_from_file_location("run_m3_plus_family", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_m3_plus_family_emits_baseline_and_lineage(monkeypatch) -> None:
    uid = uuid.uuid4().hex
    l_output_root = Path("runs/l_series/m3_plus") / f"test_integration_{uid}"
    report_output_root = Path("runs/m_series/m3_plus") / f"test_integration_{uid}"
    baseline_manifest = Path("docs/baselines") / f"test_m_series_baseline_{uid}.json"
    j5_summary = Path("runs/j_series") / f"test_j5_summary_{uid}.json"
    adapter_path = Path("runs/phase5_two_stage_recovery_anchors") / f"test_adapter_{uid}"
    adapter_path.mkdir(parents=True, exist_ok=True)

    _write_json(
        baseline_manifest,
        {
            "series_id": "M",
            "baseline_id": f"M_BASE_TEST_{uid}",
            "upstream_best": {"j_series": "M1.4/J-5", "l_series": "M2.C/L6-C"},
            "m_base": {
                "dataset": "runs/j_series/test_j5_clean.jsonl",
                "constraints": "L6-C",
                "identity_reg": "swap_test",
                "curriculum": "recursive_scope_depth_ramp_with_minimal_edit_foils",
                "optimizer": "AdamW_lr_2e-4_seed_7",
            },
        },
    )
    _write_json(
        j5_summary,
        {
            "metrics": {
                "accepted_foil_pair_accuracy": 0.8125,
                "foil_auc": 0.8125,
            },
            "samples": [
                {"true_score": 1.0, "false_score": 0.0},
                {"true_score": 0.8, "false_score": 0.3},
            ],
        },
    )

    args = Namespace(
        base_model="dummy-base",
        adapter=adapter_path,
        train_steps=1,
        dataset_size=4,
        dataset_profile="legacy",
        difficulty_tier="all",
        seed=7,
        local_files_only=True,
        l_output_root=l_output_root,
        report_output_root=report_output_root,
        j5_summary=j5_summary,
        dynamic_arity_signatures=False,
        operator_arity_json=None,
        default_relation_arity=2,
        arity_enforcement_mode="crystallization",
        track_label="M3+",
        baseline_manifest=baseline_manifest,
    )

    monkeypatch.setattr(mod, "parse_args", lambda: args)

    def _fake_run(cmd: list[str], check: bool = False):
        output_root = Path(cmd[cmd.index("--output-root") + 1])
        child = output_root / "20260310_000000"
        child.mkdir(parents=True, exist_ok=True)
        run_label = output_root.name
        summary = {
            "final_step": {
                "constraint_arity_strict": 0.0,
                "constraint_scope": 0.09 if run_label == "m3_4" else 0.25,
                "constraint_scope_unbound": 0.01 if run_label == "m3_4" else 0.08,
                "constraint_identity": 0.0,
            },
            "tier_b_enabled": True,
            "tier_c_enabled": True,
        }
        _write_json(child / "l_series_summary.json", summary)
        (child / "l_series_checkpoint.pt").write_bytes(b"checkpoint")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    mod.main()

    report_dirs = sorted(p for p in report_output_root.iterdir() if p.is_dir())
    assert report_dirs, "expected M3+ report directory"
    report = json.loads((report_dirs[-1] / "m3_plus_report.json").read_text(encoding="utf-8"))

    assert report["series"]["series_id"] == "M"
    assert report["inputs"]["baseline_id"] == f"M_BASE_TEST_{uid}"
    assert report["declared_l_output_root"].startswith(str(l_output_root).replace("\\", "/"))
    assert report["declared_report_output_root"].startswith(str(report_output_root).replace("\\", "/"))
    assert report["gate_eval"]["gate_target_run_id"] == "M3.4"
    for row in report["rows"]:
        assert row["lineage"]["mode"] == "train"
        if row["status"] == "ok":
            assert str(row["lineage"]["checkpoint_out"]).startswith(str(l_output_root).replace("\\", "/"))
