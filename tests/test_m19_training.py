from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from lojban_evolution.m19 import artifact_contract
from lojban_evolution.m19.training import (
    checkpoint_selection_score,
    maybe_apply_surface_augmentations,
    select_best_checkpoint,
)


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_replication_suite = _load_script_module("run_m19_replication_suite.py")
_prepare_selection_slices = _replication_suite._prepare_selection_slices
_selection_surface_size = _replication_suite._selection_surface_size


def test_report_command_contract_prevents_stale_artifact_reuse(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    calls: list[tuple[list[str], str, bool]] = []

    def fake_run(cmd: list[str], cwd: str, check: bool) -> None:
        calls.append((cmd, cwd, check))
        report_path.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr(artifact_contract.subprocess, "run", fake_run)
    cmd_a = ["python", "script.py", "--eval-size", "80"]
    cmd_b = ["python", "script.py", "--eval-size", "200"]

    assert artifact_contract.run_if_needed(report_path, cmd_a, tmp_path) is True
    assert len(calls) == 1
    assert artifact_contract.command_contract_matches(report_path, cmd_a) is True

    calls.clear()
    assert artifact_contract.run_if_needed(report_path, cmd_a, tmp_path) is False
    assert calls == []

    assert artifact_contract.run_if_needed(report_path, cmd_b, tmp_path) is True
    assert len(calls) == 1
    assert artifact_contract.command_contract_matches(report_path, cmd_b) is True


def test_report_command_contract_includes_script_and_input_hashes(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    script_path = tmp_path / "child.py"
    dataset_path = tmp_path / "eval.jsonl"
    script_path.write_text("print('v1')\n", encoding="utf-8")
    dataset_path.write_text('{"x": 1}\n', encoding="utf-8")
    cmd = ["python", str(script_path), "--eval-data-path", str(dataset_path)]

    report_path.write_text("{}", encoding="utf-8")
    artifact_contract.write_command_contract(report_path, cmd, tmp_path)

    assert artifact_contract.command_contract_matches(report_path, cmd, tmp_path) is True

    dataset_path.write_text('{"x": 2}\n', encoding="utf-8")

    assert artifact_contract.command_contract_matches(report_path, cmd, tmp_path) is False


def test_surface_augmentations_can_compose_entity_and_format_changes() -> None:
    rng = random.Random(0)
    question, answer, flags = maybe_apply_surface_augmentations(
        "Alice asked Bob: Where is the key?",
        "Bob",
        entity_rename_probability=1.0,
        format_flatten_probability=1.0,
        rng=rng,
    )

    assert question == "avery asked blake where is the key?"
    assert answer == "blake"
    assert flags == {"entity_renamed": True, "format_flattened": True}


def test_checkpoint_selection_prefers_purged_and_audit_under_audit_purged_policy() -> None:
    candidates = [
        {
            "epoch": 1,
            "checkpoint_path": "epoch1.pt",
            "purged_accuracy": 0.30,
            "audit_qformer_accuracy": 0.90,
            "final_mean_loss": 0.70,
        },
        {
            "epoch": 2,
            "checkpoint_path": "epoch2.pt",
            "purged_accuracy": 0.42,
            "audit_qformer_accuracy": 0.60,
            "final_mean_loss": 0.80,
        },
    ]

    best = select_best_checkpoint(candidates, "audit_purged")

    assert best is not None
    assert best["checkpoint_path"] == "epoch2.pt"
    assert checkpoint_selection_score(
        purged_accuracy=0.42,
        audit_qformer_accuracy=0.60,
        final_mean_loss=0.80,
        policy="audit_purged",
    ) > checkpoint_selection_score(
        purged_accuracy=0.30,
        audit_qformer_accuracy=0.90,
        final_mean_loss=0.70,
        policy="audit_purged",
    )


def test_checkpoint_selection_uses_loss_for_final_only_policy() -> None:
    score_low_loss = checkpoint_selection_score(
        purged_accuracy=None,
        audit_qformer_accuracy=None,
        final_mean_loss=0.4,
        policy="final_only",
    )
    score_high_loss = checkpoint_selection_score(
        purged_accuracy=None,
        audit_qformer_accuracy=None,
        final_mean_loss=0.9,
        policy="final_only",
    )

    assert score_low_loss is not None
    assert score_high_loss is not None
    assert score_low_loss > score_high_loss


def test_checkpoint_selection_can_prefer_format_robust_epoch() -> None:
    candidates = [
        {
            "epoch": 1,
            "checkpoint_path": "epoch1.pt",
            "purged_accuracy": 0.40,
            "format_accuracy": 0.18,
            "audit_qformer_accuracy": 0.90,
            "final_mean_loss": 0.70,
        },
        {
            "epoch": 2,
            "checkpoint_path": "epoch2.pt",
            "purged_accuracy": 0.38,
            "format_accuracy": 0.30,
            "audit_qformer_accuracy": 0.80,
            "final_mean_loss": 0.72,
        },
    ]

    best = select_best_checkpoint(candidates, "audit_purged_format")

    assert best is not None
    assert best["checkpoint_path"] == "epoch2.pt"
    assert checkpoint_selection_score(
        purged_accuracy=0.38,
        format_accuracy=0.30,
        audit_qformer_accuracy=0.80,
        final_mean_loss=0.72,
        policy="audit_purged_format",
    ) > checkpoint_selection_score(
        purged_accuracy=0.40,
        format_accuracy=0.18,
        audit_qformer_accuracy=0.90,
        final_mean_loss=0.70,
        policy="audit_purged_format",
    )


def test_checkpoint_selection_can_bias_toward_weak_seed_surface_robustness() -> None:
    candidates = [
        {
            "epoch": 1,
            "checkpoint_path": "epoch1.pt",
            "purged_accuracy": 0.45,
            "entity_accuracy": 0.41,
            "format_accuracy": 0.35,
            "entity_renamed_accuracy": 0.47,
            "numeric_accuracy": 0.39,
            "audit_qformer_accuracy": 0.55,
            "arity_violation_rate": 0.10,
            "masked_pointer_zero_rate": 1.0,
            "final_mean_loss": 0.72,
        },
        {
            "epoch": 2,
            "checkpoint_path": "epoch2.pt",
            "purged_accuracy": 0.47,
            "entity_accuracy": 0.19,
            "format_accuracy": 0.24,
            "entity_renamed_accuracy": 0.31,
            "numeric_accuracy": 0.22,
            "audit_qformer_accuracy": 0.80,
            "arity_violation_rate": 0.10,
            "masked_pointer_zero_rate": 1.0,
            "final_mean_loss": 0.70,
        },
    ]

    best = select_best_checkpoint(candidates, "audit_purged_surface_arity_weakseed")

    assert best is not None
    assert best["checkpoint_path"] == "epoch1.pt"
    assert checkpoint_selection_score(
        purged_accuracy=0.45,
        entity_accuracy=0.41,
        format_accuracy=0.35,
        entity_renamed_accuracy=0.47,
        numeric_accuracy=0.39,
        audit_qformer_accuracy=0.55,
        arity_violation_rate=0.10,
        masked_pointer_zero_rate=1.0,
        final_mean_loss=0.72,
        policy="audit_purged_surface_arity_weakseed",
    ) > checkpoint_selection_score(
        purged_accuracy=0.47,
        entity_accuracy=0.19,
        format_accuracy=0.24,
        entity_renamed_accuracy=0.31,
        numeric_accuracy=0.22,
        audit_qformer_accuracy=0.80,
        arity_violation_rate=0.10,
        masked_pointer_zero_rate=1.0,
        final_mean_loss=0.70,
        policy="audit_purged_surface_arity_weakseed",
    )


def test_checkpoint_selection_penalizes_purged_to_worst_surface_gap() -> None:
    robust = checkpoint_selection_score(
        purged_accuracy=0.42,
        entity_accuracy=0.38,
        format_accuracy=0.36,
        entity_renamed_accuracy=0.39,
        numeric_accuracy=0.37,
        audit_qformer_accuracy=0.55,
        arity_violation_rate=0.10,
        masked_pointer_zero_rate=1.0,
        final_mean_loss=0.72,
        policy="audit_purged_surface_arity_weakseed",
    )
    brittle = checkpoint_selection_score(
        purged_accuracy=0.48,
        entity_accuracy=0.18,
        format_accuracy=0.24,
        entity_renamed_accuracy=0.31,
        numeric_accuracy=0.20,
        audit_qformer_accuracy=0.55,
        arity_violation_rate=0.10,
        masked_pointer_zero_rate=1.0,
        final_mean_loss=0.70,
        policy="audit_purged_surface_arity_weakseed",
    )

    assert robust is not None
    assert brittle is not None
    assert robust > brittle


def test_selection_surface_size_defaults_to_200_and_writes_numeric_slice(tmp_path: Path) -> None:
    args = argparse.Namespace(selection_surface_size=200, selection_purged_eval_size=None)
    assert _selection_surface_size(args) == 200

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text("", encoding="utf-8")
    rows = [
        {"prompt": f"Alice moved {idx} boxes to Bob.", "answer": "Alice"}
        for idx in range(5)
    ]
    eval_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    paths = _prepare_selection_slices(
        run_dir=tmp_path / "run",
        train_path=train_path,
        eval_path=eval_path,
        selection_surface_size=5,
    )

    assert paths["purged_count"] == 5
    assert paths["format_count"] == 5
    assert paths["entity_count"] == 5
    assert paths["entity_renamed_count"] == 5
    assert paths["numeric_count"] == 5
    assert paths["selection_requested_count"] == 5
    assert paths["selection_shortfall"] == 0
    assert Path(paths["numeric_eval_path"]).exists()
    assert Path(paths["selection_manifest_path"]).exists()


def test_selection_surface_tops_up_after_overlap_purge(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    eval_rows = [
        {"prompt": f"Alice moved item {idx} to Bob.", "answer": "Alice"}
        for idx in range(5)
    ]
    train_rows = eval_rows[:2]
    train_path.write_text("\n".join(json.dumps(row) for row in train_rows), encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(row) for row in eval_rows), encoding="utf-8")

    paths = _prepare_selection_slices(
        run_dir=tmp_path / "run",
        train_path=train_path,
        eval_path=eval_path,
        selection_surface_size=3,
    )

    assert paths["selection_requested_count"] == 3
    assert paths["overlap_count"] == 2
    assert paths["available_purged_count"] == 3
    assert paths["purged_count"] == 3
    assert paths["format_count"] == 3
    assert paths["numeric_count"] == 3
    assert paths["selection_shortfall"] == 0
