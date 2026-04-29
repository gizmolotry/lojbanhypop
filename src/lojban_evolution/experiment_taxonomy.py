from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .repo_paths import REPO_ROOT, repo_relative


TAXONOMY_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_taxonomy.json"


def load_taxonomy_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or TAXONOMY_CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def enrich_history_entries(entries: list[dict[str, Any]], taxonomy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    taxonomy = taxonomy or load_taxonomy_config()
    major_families = taxonomy.get("major_families", {})
    entry_overrides = taxonomy.get("entry_overrides", {})
    comparison_index = build_comparison_index(taxonomy)

    for entry in entries:
        normalized = _normalized_identity(entry, entry_overrides)
        family_key = f"M{normalized['series_major']}" if normalized.get("series_major") is not None else None
        family_manifest = major_families.get(family_key or "", {})
        override = entry_overrides.get(str(entry.get("canonical_id") or ""), {})

        if override.get("legacy_aliases_to_add"):
            for alias in override["legacy_aliases_to_add"]:
                if alias not in entry.get("aliases", []):
                    entry.setdefault("aliases", []).append(alias)
                if alias not in entry.get("lookup_aliases", []):
                    entry.setdefault("lookup_aliases", []).append(alias)

        entry["normalized_canonical_id"] = normalized.get("normalized_canonical_id")
        entry["series_major"] = normalized.get("series_major")
        entry["series_minor"] = normalized.get("series_minor")
        entry["series_cell"] = normalized.get("series_cell")
        entry["question_boundary"] = family_key
        entry["architectural_thesis"] = family_manifest.get("architectural_thesis")
        entry["allowed_ablation_axes"] = list(family_manifest.get("allowed_ablation_axes", []))
        entry["forbidden_drift_axes"] = list(family_manifest.get("forbidden_drift_axes", []))
        entry["promotion_basis"] = list(override.get("promotion_basis", family_manifest.get("promotion_basis", [])))
        entry["metrics_primary"] = list(family_manifest.get("metrics_primary", []))
        entry["metrics_guardrail"] = list(family_manifest.get("metrics_guardrail", []))
        entry["baseline_manifest"] = family_manifest.get("baseline_manifest")
        entry["inherits_from"] = list(override.get("inherits_from", []))
        entry["inherits_components"] = list(override.get("inherits_components", []))
        entry["frozen_components"] = list(override.get("frozen_components", []))
        entry["changed_components"] = list(override.get("changed_components", []))
        entry["dropped_components"] = list(override.get("dropped_components", []))
        entry["component_inventory"] = dict(override.get("component_inventory", {}))
        comparison_contract = deepcopy_dict(comparison_index.get(family_key or "", {}))
        entry["comparison_contract"] = comparison_contract
        entry["required_test_contracts"] = list(comparison_contract.get("required_test_contract_ids", []))
        entry["historical_comparison_families"] = list(comparison_contract.get("historical_comparison_families", []))
        entry["active_doc_path"] = _derive_active_doc_path(entry)
        entry["archive_path"] = _derive_archive_path(entry)
    return entries


def build_transition_index(taxonomy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    taxonomy = taxonomy or load_taxonomy_config()
    return list(taxonomy.get("transition_manifests", []))


def build_comparison_index(taxonomy: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    taxonomy = taxonomy or load_taxonomy_config()
    defaults = taxonomy.get("comparison_defaults", {})
    contracts = taxonomy.get("family_comparison_contracts", {})
    test_catalog = taxonomy.get("test_contract_catalog", {})
    transition_map = {str(row.get("to_major") or ""): row for row in taxonomy.get("transition_manifests", [])}
    major_families = taxonomy.get("major_families", {})

    output: dict[str, dict[str, Any]] = {}
    for family_key in major_families:
        contract = dict(defaults)
        contract.update(contracts.get(family_key, {}))
        ancestors = _ancestor_majors(family_key, transition_map) if contract.get("compare_against_ancestors", True) else []
        comparison_targets: list[dict[str, Any]] = []
        if contract.get("compare_within_family", True):
            comparison_targets.append({"kind": "same_major_family", "target": family_key, "reason": "peer ablations inside the same question boundary"})
        for ancestor in ancestors:
            transition = transition_map.get(ancestor["child"])
            comparison_targets.append(
                {
                    "kind": "ancestor_major_family",
                    "target": ancestor["major"],
                    "selected_upstream": transition.get("selected_upstream") if isinstance(transition, dict) else None,
                    "reason": "upstream promoted family in the declared lineage chain",
                }
            )
        for legacy_family in contract.get("historical_comparison_families", []):
            comparison_targets.append(
                {"kind": "legacy_family", "target": legacy_family, "reason": "historical comparator family carried forward by policy"}
            )
        for explicit_entry in contract.get("explicit_compare_entries", []):
            comparison_targets.append({"kind": "explicit_entry", "target": explicit_entry, "reason": "family-specific explicit comparator"})

        required_ids = list(contract.get("required_test_contracts", []))
        if contract.get("inherit_required_test_contracts", True):
            for ancestor in ancestors:
                required_ids.extend(contracts.get(ancestor["major"], {}).get("required_test_contracts", []))
        required_ids = unique_strings(required_ids)
        output[family_key] = {
            "family_key": family_key,
            "compare_within_family": bool(contract.get("compare_within_family", True)),
            "compare_against_ancestors": bool(contract.get("compare_against_ancestors", True)),
            "inherit_required_test_contracts": bool(contract.get("inherit_required_test_contracts", True)),
            "historical_comparison_families": list(contract.get("historical_comparison_families", [])),
            "comparison_targets": _dedupe_comparison_targets(comparison_targets),
            "required_test_contract_ids": required_ids,
            "required_test_contracts": [deepcopy_dict(test_catalog.get(test_id, {"test_id": test_id})) for test_id in required_ids],
        }
    return output


def _normalized_identity(entry: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    canonical_id = str(entry.get("canonical_id") or "")
    override = overrides.get(canonical_id, {})
    if override:
        normalized_id = override.get("normalized_canonical_id")
        series_major = override.get("series_major")
        series_minor = override.get("series_minor")
        series_cell = override.get("series_cell")
        if normalized_id:
            if series_major is None or series_minor is None:
                major, minor, cell = _parse_normalized_id(normalized_id)
                series_major = series_major if series_major is not None else major
                series_minor = series_minor if series_minor is not None else minor
                series_cell = series_cell if series_cell is not None else cell
            return {
                "normalized_canonical_id": normalized_id,
                "series_major": series_major,
                "series_minor": series_minor,
                "series_cell": series_cell,
            }

    inferred = _infer_normalized_from_canonical(canonical_id)
    if inferred is not None:
        return inferred
    return {
        "normalized_canonical_id": canonical_id,
        "series_major": None,
        "series_minor": None,
        "series_cell": None,
    }


def _infer_normalized_from_canonical(canonical_id: str) -> dict[str, Any] | None:
    patterns = [
        (r"^m\.track\.m(\d+)_([0-9]+[a-z]?)\.([a-z]\d*)$", True),
        (r"^m\.track\.m(\d+)_([0-9]+[a-z]?)$", "minor_family"),
        (r"^m\.track\.m(\d+)\.([a-z]\d*)$", False),
        (r"^m\.track\.m(\d+)$", "major_family"),
        (r"^l\.branch\.m(\d+)_([0-9]+[a-z]?)\.([a-z]\d*)$", True),
        (r"^l\.branch\.m(\d+)_([0-9]+[a-z]?)$", "minor_family_l"),
        (r"^l\.branch\.m(\d+)\.m(\d+)_(\d+)$", "flat_family"),
        (r"^m\.family\.m(\d+)$", "family_only"),
    ]
    for pattern, mode in patterns:
        match = re.match(pattern, canonical_id)
        if not match:
            continue
        if mode is True:
            major = int(match.group(1))
            minor = match.group(2)
            cell = match.group(3).upper()
            return {
                "normalized_canonical_id": f"M{major}.{minor}.{cell}",
                "series_major": major,
                "series_minor": minor,
                "series_cell": cell,
            }
        if mode is False:
            major = int(match.group(1))
            cell = match.group(2).upper()
            return {
                "normalized_canonical_id": f"M{major}.{cell}",
                "series_major": major,
                "series_minor": None,
                "series_cell": cell,
            }
        if mode == "minor_family":
            major = int(match.group(1))
            minor = match.group(2)
            return {
                "normalized_canonical_id": f"M{major}.{minor}",
                "series_major": major,
                "series_minor": minor,
                "series_cell": None,
            }
        if mode == "major_family":
            major = int(match.group(1))
            return {
                "normalized_canonical_id": f"M{major}",
                "series_major": major,
                "series_minor": None,
                "series_cell": None,
            }
        if mode == "minor_family_l":
            major = int(match.group(1))
            minor = match.group(2)
            return {
                "normalized_canonical_id": f"M{major}.{minor}",
                "series_major": major,
                "series_minor": minor,
                "series_cell": None,
            }
        if mode == "flat_family":
            major = int(match.group(1))
            minor = int(match.group(3))
            return {
                "normalized_canonical_id": f"M{major}.{minor}",
                "series_major": major,
                "series_minor": minor,
                "series_cell": None,
            }
        if mode == "family_only":
            major = int(match.group(1))
            return {
                "normalized_canonical_id": f"M{major}",
                "series_major": major,
                "series_minor": None,
                "series_cell": None,
            }
    return None


def _parse_normalized_id(value: str) -> tuple[int | None, int | str | None, str | None]:
    match = re.match(r"^M(\d+)(?:\.([0-9]+[a-z]?))?(?:\.([A-Z]\d*))?$", str(value))
    if not match:
        return None, None, None
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) and match.group(2).isdigit() else match.group(2)
    cell = match.group(3) if match.group(3) else None
    return major, minor, cell


def _derive_active_doc_path(entry: dict[str, Any]) -> str | None:
    for record in entry.get("evidence_records", []):
        for path in record.get("source_paths", []):
            if str(path).startswith("docs/"):
                return str(path)
    return None


def _derive_archive_path(entry: dict[str, Any]) -> str | None:
    for record in entry.get("evidence_records", []):
        for path in record.get("source_paths", []):
            if str(path).startswith("archive/"):
                return str(path)
    for root in entry.get("artifact_roots", []):
        if str(root).startswith("archive/"):
            return str(root)
        path_obj = Path(str(root))
        try:
            rel = path_obj.relative_to(REPO_ROOT)
            if str(rel).startswith("archive"):
                return repo_relative(path_obj)
        except ValueError:
            continue
    return None


def _ancestor_majors(family_key: str, transition_map: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    ancestors: list[dict[str, str]] = []
    current = family_key
    seen: set[str] = set()
    while current in transition_map:
        transition = transition_map[current]
        parent = str(transition.get("from_major") or "").strip()
        if not parent or parent in seen:
            break
        ancestors.append({"major": parent, "child": current})
        seen.add(parent)
        current = parent
    return ancestors


def unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def deepcopy_dict(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _dedupe_comparison_targets(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in values:
        target = str(row.get("target", "")).strip()
        if not target or target in seen:
            continue
        seen.add(target)
        output.append(row)
    return output
