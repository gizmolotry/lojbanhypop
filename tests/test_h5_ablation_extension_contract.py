from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from conftest import load_script_module


mod = load_script_module("run_coconut_ablation_matrix", "scripts/run_coconut_ablation_matrix.py")

REQUIRED_EXTENSION_RUN_IDS = {"H5-PROV", "H5-OOD", "H5-DPTR", "J-1", "J-2", "J-3", "J-4"}
REQUIRED_EXTENSION_KEYS = {"run_id", "name", "status", "return_code", "output", "metrics", "notes"}


def _validate_h5_extension_manifest(payload: dict) -> None:
    rows = payload.get("h5_extensions")
    assert isinstance(rows, list), "h5_extensions must be a list."
    ids = {str(r.get("run_id", "")) for r in rows if isinstance(r, dict)}
    assert REQUIRED_EXTENSION_RUN_IDS.issubset(ids), "Missing one or more required H5/J extension rows."
    for row in rows:
        assert isinstance(row, dict), "Each H5 extension row must be an object."
        missing = REQUIRED_EXTENSION_KEYS.difference(row.keys())
        assert not missing, f"H5 extension row missing keys: {sorted(missing)}"
        assert isinstance(row["metrics"], dict), "H5 extension row metrics must be an object."


def test_h5_extension_manifest_includes_required_rows_and_keys() -> None:
    manifest = {
        "timestamp": "2026-03-03T00:00:00+00:00",
        "h5_extensions": [
            {
                "run_id": "H5-PROV",
                "name": "H5 Provenance",
                "status": "ok",
                "return_code": 0,
                "output": "runs/ablation/a_to_g/h5_prov.json",
                "metrics": {"final_acc": 0.39},
                "notes": "Lineage and grounding checks.",
            },
            {
                "run_id": "H5-OOD",
                "name": "H5 Out-of-Distribution",
                "status": "ok",
                "return_code": 0,
                "output": "runs/ablation/a_to_g/h5_ood.json",
                "metrics": {"final_acc": 0.31},
                "notes": "Distribution-shift stress test.",
            },
            {
                "run_id": "H5-DPTR",
                "name": "H5 Distillation Pointer Transfer",
                "status": "ok",
                "return_code": 0,
                "output": "runs/ablation/a_to_g/h5_dptr.json",
                "metrics": {"final_acc": 0.36},
                "notes": "Pointer transfer/distillation check.",
            },
            {
                "run_id": "J-1",
                "name": "Graph Target (Factor Schema)",
                "status": "ok",
                "return_code": 0,
                "output": "runs/j_series/<timestamp>/j-1.json",
                "metrics": {"schema_valid_rate": 1.0},
                "notes": "Graph schema extraction.",
            },
            {
                "run_id": "J-2",
                "name": "Paraphrase Explosion (Invariance)",
                "status": "ok",
                "return_code": 0,
                "output": "runs/j_series/<timestamp>/j-2.json",
                "metrics": {"invariance_rate": 0.9},
                "notes": "Paraphrase invariance stress.",
            },
            {
                "run_id": "J-3",
                "name": "Stop-Grad Isolation Gate",
                "status": "ok",
                "return_code": 0,
                "output": "runs/j_series/<timestamp>/j-3.json",
                "metrics": {"stopgrad_contract_pass": 1},
                "notes": "Isolation gate check.",
            },
            {
                "run_id": "J-4",
                "name": "Operator Curriculum Build",
                "status": "ok",
                "return_code": 0,
                "output": "runs/j_series/<timestamp>/j-4.json",
                "metrics": {"operator_count": 5},
                "notes": "Operator curriculum generation.",
            },
        ],
    }

    _validate_h5_extension_manifest(manifest)


def test_h5_extension_manifest_missing_required_row_fails() -> None:
    manifest = {
        "h5_extensions": [
            {
                "run_id": "H5-PROV",
                "name": "H5 Provenance",
                "status": "ok",
                "return_code": 0,
                "output": "runs/ablation/a_to_g/h5_prov.json",
                "metrics": {"final_acc": 0.39},
                "notes": "Lineage and grounding checks.",
            },
            {
                "run_id": "H5-OOD",
                "name": "H5 Out-of-Distribution",
                "status": "ok",
                "return_code": 0,
                "output": "runs/ablation/a_to_g/h5_ood.json",
                "metrics": {"final_acc": 0.31},
                "notes": "Distribution-shift stress test.",
            },
        ]
    }

    with pytest.raises(AssertionError, match="required H5/J extension rows"):
        _validate_h5_extension_manifest(manifest)


def test_markdown_summary_includes_h5_extension_run_rows() -> None:
    records = [
        mod.RunRecord("A", "Control (English CoT -> English)", "ok", 0, [], None, None, ""),
        mod.RunRecord("B", "Rigid Lojban (Text-to-Text)", "ok", 0, [], None, None, ""),
        mod.RunRecord("C", "Coconut Fusion (Latent KV Handoff)", "ok", 0, [], None, None, ""),
        mod.RunRecord("D", "NoPE Fusion (DroPE + latent handoff)", "ok", 0, [], None, None, ""),
        mod.RunRecord("E", "Babel Bridge (Projected latent handoff)", "ok", 0, [], None, None, ""),
        mod.RunRecord("H5-PROV", "H5 Provenance", "ok", 0, [], None, {"final_acc": 0.39}, ""),
        mod.RunRecord("H5-OOD", "H5 Out-of-Distribution", "ok", 0, [], None, {"final_acc": 0.31}, ""),
        mod.RunRecord("H5-DPTR", "H5 Distillation Pointer Transfer", "ok", 0, [], None, {"final_acc": 0.36}, ""),
        mod.RunRecord("J-1", "Graph Target (Factor Schema)", "ok", 0, [], None, {"schema_valid_rate": 1.0}, ""),
        mod.RunRecord("J-2", "Paraphrase Explosion (Invariance)", "ok", 0, [], None, {"invariance_rate": 1.0}, ""),
        mod.RunRecord("J-3", "Stop-Grad Isolation Gate", "ok", 0, [], None, {"stopgrad_contract_pass": 1.0}, ""),
        mod.RunRecord("J-4", "Operator Curriculum Build", "ok", 0, [], None, {"operator_count": 5.0}, ""),
    ]

    out_dir = Path("artifacts/test_tmp") / f"h5_ext_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ablation_matrix.md"
    mod._write_summary_md(str(out), records)
    text = out.read_text(encoding="utf-8")

    assert "| `H5-PROV` | `H5 Provenance` | `ok` | `0` |" in text
    assert "| `H5-OOD` | `H5 Out-of-Distribution` | `ok` | `0` |" in text
    assert "| `H5-DPTR` | `H5 Distillation Pointer Transfer` | `ok` | `0` |" in text
    assert "| `J-1` | `Graph Target (Factor Schema)` | `ok` | `0` |" in text
    assert "| `J-2` | `Paraphrase Explosion (Invariance)` | `ok` | `0` |" in text
    assert "| `J-3` | `Stop-Grad Isolation Gate` | `ok` | `0` |" in text
    assert "| `J-4` | `Operator Curriculum Build` | `ok` | `0` |" in text
