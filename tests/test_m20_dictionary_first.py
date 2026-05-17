from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from lojban_evolution.m20.dictionary import (
    M20_LOCKS,
    M20SoftDictionaryModel,
    build_vocab,
    generate_synthetic_world_examples,
    tokenize,
    train_m20_dictionary,
)
from lojban_evolution.m20.family import M20_REGISTRY, m20_default_grid


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m20_registry_declares_six_cell_grid_and_ledger_paths() -> None:
    spec = M20_REGISTRY["M20"]
    for key in ("train", "predicate_induction", "lock_suite", "suite"):
        assert key in spec["runner_scripts"]
        assert key in spec["dags"]
        assert key in spec["output_roots"]
        assert key in spec["report_names"]
        assert "artifacts/runs/telemetry/raw/ablation/hypercube" in spec["output_roots"][key]

    grid = m20_default_grid()
    assert [cell["cell_key"] for cell in grid] == ["A", "B", "C", "D", "E", "F"]
    assert {cell["lock"] for cell in grid} == set(M20_LOCKS)
    assert all(cell["cell_id"].startswith("M20.1.") for cell in grid)


def test_synthetic_world_has_counterfactual_surfaces_and_floating_brivi_controls() -> None:
    rows = generate_synthetic_world_examples(240, seed=7, floating_fraction=0.25)
    surfaces = {row.surface for row in rows}
    assert "floating" in surfaces
    assert {"renamed", "anonymized", "flattened", "numeric"} & surfaces

    grounded = [row for row in rows if row.has_argument]
    groups: dict[str, set[str]] = {}
    for row in grounded:
        groups.setdefault(row.counterfactual_group, set()).add(row.entity_signature)
    assert any(len(signatures) >= 2 for signatures in groups.values())

    floating = [row for row in rows if not row.has_argument]
    assert floating
    assert all(row.slot_targets == (0, 0, 0) for row in floating)


def test_soft_dictionary_temperature_anneal_sharpens_assignments() -> None:
    rows = generate_synthetic_world_examples(32, seed=11)
    vocab = build_vocab(rows)
    model = M20SoftDictionaryModel(vocab_size=len(vocab), codebook_size=64, embedding_dim=16, hidden_dim=24)
    encoded = []
    for row in rows[:8]:
        ids = [vocab.get(tok, 1) for tok in tokenize(row.prompt)][:16]
        ids.extend([0] * (16 - len(ids)))
        encoded.append(ids)
    input_ids = torch.tensor(encoded, dtype=torch.long)
    high = model(input_ids, temperature=2.0)["code_probs"]
    low = model(input_ids, temperature=0.2)["code_probs"]
    high_entropy = -(high.clamp_min(1e-8) * high.clamp_min(1e-8).log()).sum(dim=-1).mean()
    low_entropy = -(low.clamp_min(1e-8) * low.clamp_min(1e-8).log()).sum(dim=-1).mean()
    assert low_entropy < high_entropy


def test_m20_training_smoke_exposes_lock_and_token_metrics() -> None:
    result = train_m20_dictionary(
        train_size=96,
        eval_size=48,
        epochs=1,
        batch_size=24,
        seed=5,
        codebook_size=64,
        embedding_dim=16,
        hidden_dim=24,
    )
    metrics = result["metrics"]
    for key in (
        "strict_accuracy",
        "phrase_accuracy",
        "accuracy_per_token",
        "predicate_identity_stability",
        "brivi_gate_accuracy",
        "argument_binding_accuracy",
        "masked_accuracy",
    ):
        assert key in metrics
    assert result["history"]


def test_m20_suite_smoke_writes_cells_report() -> None:
    output_root_rel = Path("runs") / "m_series" / "pytest_m20_suite"
    output_root = REPO_ROOT / output_root_rel
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "m20" / "run_m20_dictionary_first_suite.py"),
                "--seed-list",
                "23",
                "--cell-list",
                "A",
                "--train-size",
                "80",
                "--eval-size",
                "40",
                "--epochs",
                "1",
                "--batch-size",
                "20",
                "--codebook-size",
                "64",
                "--embedding-dim",
                "16",
                "--hidden-dim",
                "24",
                "--output-root",
                str(output_root_rel),
                "--run-id",
                "pytest_m20_suite",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        report = output_root / "pytest_m20_suite" / "m20_dictionary_first_suite_report.json"
        assert report.exists()
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["track"] == "M20.1"
        assert payload["canonical_accuracy"] == "strict_accuracy"
        assert payload["diagnostic_only"] == ["phrase_accuracy"]
        assert "A" in payload["cells"]
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)
