from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from lojban_evolution.experiment import Problem
from lojban_evolution.repro import dataset_fingerprint, write_run_manifest


def test_dataset_fingerprint_is_stable() -> None:
    rows = [
        Problem(problem_id=1, prompt="p1", answer="a1", trace=("X", "Y")),
        Problem(problem_id=2, prompt="p2", answer="a2", trace=("Y", "Z")),
    ]
    first = dataset_fingerprint(rows)
    second = dataset_fingerprint(rows)
    assert first == second
    assert len(first) == 64


def test_write_run_manifest() -> None:
    tmp_dir = Path("artifacts/test_tmp") / f"manifest_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "run_manifest.json"
    write_run_manifest(out, {"script": "scripts/run_experiment.py", "config": {"seed": 7}})
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["script"] == "scripts/run_experiment.py"
    assert payload["config"]["seed"] == 7
    shutil.rmtree(tmp_dir, ignore_errors=True)
