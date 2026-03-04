from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import uuid

from conftest import load_script_module


mod = load_script_module("run_true_coconut_h_series_testmod", "scripts/run_true_coconut_h_series.py")


def _mk_tmp_root() -> Path:
    path = Path("artifacts/test_tmp") / f"run_true_coconut_h_series_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base_args(tmp_root: Path, only_runs: list[str], execute: bool, fail_fast: bool = False) -> Namespace:
    return Namespace(
        base_model="dummy-base",
        adapter=tmp_root / "dummy-adapter",
        sample_size=24,
        seeds=[7, 11],
        dataset_size=1000,
        max_logic_new_tokens=48,
        max_final_new_tokens=48,
        h1_window=4,
        h2_layer_index=12,
        h2_layer_scale=1.0,
        h3_adapter=None,
        h3_bridge=None,
        h3_layer_index=12,
        h3_layer_scale=1.0,
        h4_bridge=None,
        h4_layer_index=12,
        h4_layer_scale=1.0,
        h4_contrastive_alpha=1.0,
        only_runs=only_runs,
        fail_fast=fail_fast,
        h5_base_model=None,
        h5_adapter=None,
        h5_checkpoint=None,
        h5_prov_base_model=None,
        h5_prov_adapter=None,
        h5_prov_checkpoint=None,
        h5_prov_h53_checkpoint=None,
        h5_prov_slice1_checkpoint=None,
        h5_prov_top_k=64,
        h5_ood_base_model=None,
        h5_ood_adapter=None,
        h5_ood_checkpoint=None,
        h5_ood_per_domain_limit=10,
        h5_ood_max_logic_new_tokens=None,
        h5_ood_max_final_new_tokens=None,
        h5_ood_layer_index=12,
        h5_ood_inject_scale=1.0,
        h5_ood_relation_bias=0.0,
        h5_ood_use_iron_collar=False,
        h5_dptr_base_model=None,
        h5_dptr_adapter=None,
        h5_dptr_checkpoint=None,
        h5_dptr_sample_size=8,
        h5_dptr_pointer_window=16,
        h5_dptr_layer_index=12,
        h5_dptr_inject_scale=1.0,
        h5_dptr_relation_bias=0.0,
        h5_dptr_use_iron_collar=False,
        output_root=tmp_root / "runs",
        local_files_only=True,
        execute=execute,
    )


def _latest_manifest(output_root: Path) -> dict:
    run_dirs = sorted([p for p in output_root.iterdir() if p.is_dir()])
    assert run_dirs
    manifest_path = run_dirs[-1] / "run_h_series.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_planned_mode_keeps_j2_planned_when_j1_selected(monkeypatch) -> None:
    args = _base_args(_mk_tmp_root(), only_runs=["J-1", "J-2"], execute=False)
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    mod.main()

    payload = _latest_manifest(args.output_root)
    rows = {r["run_id"]: r for r in payload["runs"]}
    assert rows["J-1"]["status"] == "planned"
    assert rows["J-2"]["status"] == "planned"


def test_fail_fast_aborts_after_h5_dptr_failure_before_j_runs(monkeypatch) -> None:
    args = _base_args(_mk_tmp_root(), only_runs=["H5-DPTR", "J-1"], execute=True, fail_fast=True)
    args.h5_dptr_checkpoint = args.output_root / "ckpt.pt"
    monkeypatch.setattr(mod, "parse_args", lambda: args)

    def fake_run(cmd: list[str], execute: bool):
        if "eval_h5_dynamic_pointer_refactor.py" in " ".join(cmd):
            return "failed", 2
        return "ok", 0

    monkeypatch.setattr(mod, "_run", fake_run)
    mod.main()

    payload = _latest_manifest(args.output_root)
    ids = [r["run_id"] for r in payload["runs"]]
    assert "H5-DPTR" in ids
    assert "J-1" not in ids


def test_j4_fails_if_dataset_sidecar_missing(monkeypatch) -> None:
    args = _base_args(_mk_tmp_root(), only_runs=["J-4"], execute=True)
    monkeypatch.setattr(mod, "parse_args", lambda: args)

    def fake_run(cmd: list[str], execute: bool):
        cmd_s = " ".join(cmd)
        if "eval_j_4.py" in cmd_s:
            out_path = Path(cmd[cmd.index("--output") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "summary": {"run_id": "J-4"},
                        "metrics": {"sample_count": 1, "operator_count": 1},
                        "operator_histogram": {"equality": 1},
                    }
                ),
                encoding="utf-8",
            )
            return "ok", 0
        return "ok", 0

    monkeypatch.setattr(mod, "_run", fake_run)
    mod.main()

    payload = _latest_manifest(args.output_root)
    rows = {r["run_id"]: r for r in payload["runs"]}
    assert rows["J-4"]["status"] == "failed"
    assert rows["J-4"]["return_code"] == -3
