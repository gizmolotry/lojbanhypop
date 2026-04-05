from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


StoragePath = str | Path


def _as_str(path: StoragePath) -> str:
    return str(path)


def is_s3_uri(path: StoragePath) -> bool:
    return _as_str(path).startswith("s3://")


def _split_s3_uri(path: StoragePath) -> tuple[str, str]:
    raw = _as_str(path)
    if not raw.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {raw}")
    tail = raw[5:]
    bucket, _, key = tail.partition("/")
    if not bucket:
        raise ValueError(f"Missing s3 bucket in uri: {raw}")
    return bucket, key


@lru_cache(maxsize=1)
def _s3_client():
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("S3 support requires boto3 to be installed.") from exc
    return boto3.client("s3")


def join_path(base: StoragePath, *parts: str) -> StoragePath:
    if is_s3_uri(base):
        bucket, key = _split_s3_uri(base)
        clean_parts = [p.strip("/") for p in parts if p]
        key_prefix = key.strip("/")
        joined = "/".join([x for x in [key_prefix, *clean_parts] if x])
        return f"s3://{bucket}/{joined}" if joined else f"s3://{bucket}"
    out = Path(base)
    for part in parts:
        out = out / part
    return out


def make_dirs(path: StoragePath, parents: bool = True, exist_ok: bool = True) -> None:
    if is_s3_uri(path):
        return
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def write_bytes(path: StoragePath, data: bytes) -> None:
    if is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        if not key:
            raise ValueError(f"S3 object key is required for write: {path}")
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def read_bytes(path: StoragePath) -> bytes:
    if is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        if not key:
            raise ValueError(f"S3 object key is required for read: {path}")
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    return Path(path).read_bytes()


def exists(path: StoragePath) -> bool:
    if is_s3_uri(path):
        bucket, key = _split_s3_uri(path)
        if not key:
            raise ValueError(f"S3 object key is required for exists check: {path}")
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(path).exists()


def write_text(path: StoragePath, text: str, encoding: str = "utf-8") -> None:
    write_bytes(path, text.encode(encoding))


def read_text(path: StoragePath, encoding: str = "utf-8") -> str:
    return read_bytes(path).decode(encoding)


def write_json(path: StoragePath, payload: Any, indent: int = 2) -> None:
    write_text(path, json.dumps(payload, indent=indent), encoding="utf-8")
