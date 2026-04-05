from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from conftest import load_script_module


def test_mixed_curriculum_integrity(monkeypatch):
    mod = load_script_module("build_mixed_curriculum_dataset", "scripts/build_mixed_curriculum_dataset.py")
    out_dir = Path(__file__).resolve().parents[1] / "src" / ".tmp_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mixed_curriculum_integrity.jsonl"
    if out.exists():
        out.unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mixed_curriculum_dataset.py",
            "--output",
            str(out),
            "--dataset-size",
            "60",
            "--seeds",
            "7",
            "--copies-per-problem",
            "1",
            "--fluid-ratio",
            "1.0",
            "--max-samples",
            "0",
        ],
    )
    mod.main()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    crystal_rows = [r for r in rows if r.get("mode") == "crystal"]
    fluid_rows = [r for r in rows if r.get("mode") == "fluid"]

    assert len(crystal_rows) == len(fluid_rows), "Dataset is imbalanced; will induce mode collapse."

    for row in rows:
        assert "trace_loss_multiplier" in row, "Missing entropy control weights."
        assert "answer_loss_multiplier" in row, "Missing communication anchors."
        assert "prompt_loss_multiplier" in row, "Missing prompt weight."
        assert "trace_anchor" in row
        assert "answer_anchor" in row

    shutil.rmtree(out_dir, ignore_errors=True)
