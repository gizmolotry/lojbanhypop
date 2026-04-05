from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from lojban_evolution.storage import _s3_client, exists, join_path, write_bytes


def test_join_path_s3_normalizes_slashes() -> None:
    out = join_path("s3://my-bucket//root//", "/child/", "file.json")
    assert out == "s3://my-bucket/root/child/file.json"


def test_join_path_local_returns_path() -> None:
    out = join_path(Path("artifacts"), "runs", "x.json")
    assert isinstance(out, Path)
    assert out.as_posix().endswith("artifacts/runs/x.json")


def test_write_bytes_s3_root_uri_raises_value_error() -> None:
    with pytest.raises(ValueError):
        write_bytes("s3://bucket-only", b"x")


def test_exists_s3_handles_head_object_failure_false(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeS3Client:
        def head_object(self, **kwargs):
            raise RuntimeError("not found")

    fake_boto3 = types.SimpleNamespace(client=lambda service: FakeS3Client() if service == "s3" else None)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    _s3_client.cache_clear()
    assert exists("s3://my-bucket/path/to/object.json") is False
    _s3_client.cache_clear()
