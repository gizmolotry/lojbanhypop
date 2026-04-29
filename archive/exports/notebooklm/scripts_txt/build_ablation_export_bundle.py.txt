from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "archive" / "exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a comprehensive export bundle for the ablation program.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("ablation_export_bundle_%Y%m%d_%H%M%S")
    bundle_root = args.output_root / run_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    included: list[dict[str, str]] = []

    patterns = [
        "docs/**/*.md",
        "docs/**/*.json",
        "docs/**/*.txt",
        "configs/**/*.json",
        "scripts/**/*.py",
        "scripts/**/*.ps1",
        "airflow/dags/**/*.py",
        "src/lojban_evolution/**/*.py",
        "tests/**/*.py",
    ]
    for pattern in patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if _should_skip(path):
                continue
            _copy_into_bundle(path, bundle_root)
            included.append({"source": _rel(path), "flat_name": _flat_name(path)})

    artifact_roots = [
        "ablation_history_backfill",
        "ablation_program_map",
        "m_bridge_ablation_test_suite",
        "m14_symbiote_scratchpad",
        "m3_19_d_mainline_grid",
        "m3_18_decoder_reentry_resume",
        "m3_17_advisor_reentry_bridge",
        "m11_discriminative_suite",
        "m5_autoformalization",
        "m5_padded_nary",
        "m5_2_autoregressive_chain",
        "m5_3_masked_pair_chain",
        "m4_0_semantic_probe",
        "m4_2_predicate_grounding",
        "m4",
        "m3_plus",
        "m3_9_primitive_probe",
        "m3_10_ood_accuracy",
        "m3_11_winograd_failure_anatomy",
        "m2_hypercube_20260304",
        "m2_hypercube_20260304_bundle",
        "grid_20260305_full",
    ]
    telemetry_root = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube"
    for root_name in artifact_roots:
        root = telemetry_root / root_name
        if not root.exists():
            continue
        latest = _latest_leaf_run_dir(root)
        if latest is None:
            continue
        for path in sorted(latest.rglob("*")):
            if not path.is_file() or _should_skip(path):
                continue
            _copy_into_bundle(path, bundle_root)
            included.append({"source": _rel(path), "flat_name": _flat_name(path)})

    archive_patterns = [
        "archive/results/m6/**/*.json",
        "archive/results/m7/**/*.json",
        "archive/results/m8/**/*.json",
        "archive/results/m9/**/*.json",
        "archive/results/m10/**/*.json",
        "archive/results/m10/**/*.md",
        "archive/reports/**/*.md",
    ]
    for pattern in archive_patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if _should_skip(path):
                continue
            _copy_into_bundle(path, bundle_root)
            included.append({"source": _rel(path), "flat_name": _flat_name(path)})

    unique_entries = _dedupe_entries(included)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": _rel(bundle_root),
        "file_count": len(unique_entries),
        "included_files": unique_entries,
        "notes": [
            "This bundle is flattened for direct multi-file upload.",
            "Filenames are path-encoded so NotebookLM-style upload can be done from one directory.",
        ],
    }
    manifest_path = bundle_root / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path = bundle_root / "EXPORT_SUMMARY.md"
    summary_path.write_text(_render_summary(manifest), encoding="utf-8")

    print(f"Wrote bundle: {bundle_root}")
    print(f"Files copied: {manifest['file_count']}")


def _latest_leaf_run_dir(root: Path) -> Path | None:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _copy_into_bundle(path: Path, bundle_root: Path) -> None:
    dest = bundle_root / _flat_name(path)
    shutil.copy2(path, dest)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _should_skip(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if "__pycache__" in lowered_parts:
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def _flat_name(path: Path) -> str:
    rel = _rel(path)
    stemmed = re.sub(r"[^A-Za-z0-9._-]+", "__", rel)
    return stemmed


def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for entry in entries:
        seen[entry["source"]] = entry
    return [seen[key] for key in sorted(seen)]


def _render_summary(manifest: dict[str, object]) -> str:
    lines = [
        "# Ablation Export Bundle",
        "",
        f"- Generated UTC: `{manifest['generated_utc']}`",
        f"- File count: `{manifest['file_count']}`",
        "",
        "## Included Categories",
        "",
        "- docs, ledgers, specs, taxonomy, and cleanup docs",
        "- configs and baseline manifests",
        "- scripts for historical backfill, M-series experiments, and export tooling",
        "- Airflow DAG orchestration across letter-era and M-era families",
        "- core `src/lojban_evolution` program modules",
        "- selected latest telemetry manifests and summaries",
        "- archive-backed M6-M10 result reports",
        "- all files flattened into this single folder",
        "",
        "## Manifest",
        "",
        "- `EXPORT_MANIFEST.json` contains the full file list.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
