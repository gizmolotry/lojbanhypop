from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import types
import uuid

from lojban_evolution.experiment import Problem, run_experiment
from lojban_evolution.repro import dataset_fingerprint, write_run_manifest
from lojban_evolution.storage import _s3_client, join_path, write_text


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


def test_write_text_to_uri_local_string_path() -> None:
    tmp_dir = Path("artifacts/test_tmp") / f"local_uri_{uuid.uuid4().hex}"
    out = tmp_dir / "payload.json"
    write_text(str(out), '{"ok": true}')
    assert out.exists()
    assert out.read_text(encoding="utf-8") == '{"ok": true}'
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_write_run_manifest_s3_uri_uses_boto3_put_object(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeS3Client:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    fake_boto3 = types.SimpleNamespace(client=lambda service: FakeS3Client() if service == "s3" else None)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    _s3_client.cache_clear()

    write_run_manifest("s3://unit-test-bucket/manifests/run_manifest.json", {"script": "scripts/run_experiment.py"})

    assert len(calls) == 1
    call = calls[0]
    assert call["Bucket"] == "unit-test-bucket"
    assert call["Key"] == "manifests/run_manifest.json"
    payload = json.loads(call["Body"].decode("utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["script"] == "scripts/run_experiment.py"
    _s3_client.cache_clear()


def test_run_experiment_s3_writes_history_and_summary(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeS3Client:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    fake_boto3 = types.SimpleNamespace(client=lambda service: FakeS3Client() if service == "s3" else None)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    _s3_client.cache_clear()

    payload = run_experiment(
        output_root="s3://unit-test-bucket/runs",
        iterations=1,
        seed=7,
        dataset_size=20,
        max_accept_per_iteration=1,
    )
    write_run_manifest(join_path(payload["run_dir"], "run_manifest.json"), {"script": "scripts/run_experiment.py"})

    keys = {call["Key"] for call in calls}
    assert any(key.endswith("/history.json") for key in keys)
    assert any(key.endswith("/summary.md") for key in keys)
    assert any(key.endswith("/run_manifest.json") for key in keys)
    _s3_client.cache_clear()
