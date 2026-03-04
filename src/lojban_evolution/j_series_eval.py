from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

_ALLOWED_RELATIONS = {
    "inside",
    "contains",
    "north_of",
    "south_of",
    "east_of",
    "west_of",
    "implies",
    "equals",
    "corefers",
    "and",
    "or",
    "not",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", text.strip().lower())


def _entity_name(token: str) -> str:
    out = _norm_token(token)
    return out if out else "_unk"


def _sample_text(row: Mapping[str, Any]) -> str:
    for key in ("question", "prompt", "input", "text"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def load_text_samples(path: Path | None) -> List[str]:
    if path is None:
        return [
            "The blue box is inside the red box.",
            "A is north of B. B is east of C.",
            "If P then Q.",
            "Alice equals WitnessA.",
        ]
    payload = _read_json(path)
    samples: List[str] = []
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                txt = _sample_text(row)
                if txt:
                    samples.append(txt)
    if not samples and isinstance(payload, dict):
        txt = _sample_text(payload)
        if txt:
            samples.append(txt)
    return samples


def parse_text_to_graph(text: str) -> Dict[str, Any]:
    s = re.sub(r"\s+", " ", text.strip())
    entities: set[str] = set()
    edges: List[Dict[str, str]] = []

    patterns: List[Tuple[str, str]] = [
        (r"\b([A-Za-z][\w-]*)\s+is\s+inside\s+the\s+([A-Za-z][\w-]*)\b", "inside"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+inside\s+([A-Za-z][\w-]*)\b", "inside"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+north\s+of\s+([A-Za-z][\w-]*)\b", "north_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+south\s+of\s+([A-Za-z][\w-]*)\b", "south_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+east\s+of\s+([A-Za-z][\w-]*)\b", "east_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+west\s+of\s+([A-Za-z][\w-]*)\b", "west_of"),
        (r"\bif\s+([A-Za-z][\w-]*)\s+then\s+([A-Za-z][\w-]*)\b", "implies"),
        (r"\b([A-Za-z][\w-]*)\s+equals\s+([A-Za-z][\w-]*)\b", "equals"),
        (r"\b([A-Za-z][\w-]*)\s+same\s+as\s+([A-Za-z][\w-]*)\b", "equals"),
    ]
    for pattern, rel in patterns:
        for a, b in re.findall(pattern, s, flags=re.IGNORECASE):
            src = _entity_name(a)
            dst = _entity_name(b)
            entities.add(src)
            entities.add(dst)
            edges.append({"src": src, "rel": rel, "dst": dst})

    # Lightweight coreference heuristic.
    coref_match = re.search(r"\b([A-Za-z][\w-]*)\b.*\b(it|they|them)\b", s, flags=re.IGNORECASE)
    if coref_match:
        src = _entity_name(coref_match.group(1))
        dst = "pronoun_ref"
        entities.add(src)
        entities.add(dst)
        edges.append({"src": src, "rel": "corefers", "dst": dst})

    if not edges:
        toks = [t for t in re.findall(r"[A-Za-z][\w-]*", s) if t.lower() not in {"the", "is", "of", "if", "then"}]
        toks = [_entity_name(t) for t in toks[:2]]
        if len(toks) == 2:
            entities.update(toks)
            edges.append({"src": toks[0], "rel": "and", "dst": toks[1]})

    return {"entities": sorted(entities), "edges": edges}


def canonical_edges(graph: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    edges = graph.get("edges") if isinstance(graph, Mapping) else None
    if isinstance(edges, list):
        for e in edges:
            if isinstance(e, Mapping):
                src = _entity_name(str(e.get("src", "")))
                rel = _entity_name(str(e.get("rel", "")))
                dst = _entity_name(str(e.get("dst", "")))
                out.append((src, rel, dst))
    return sorted(out)


def validate_graph_schema(graph: Mapping[str, Any]) -> bool:
    entities = graph.get("entities")
    edges = graph.get("edges")
    if not isinstance(entities, list) or not isinstance(edges, list):
        return False
    ent_set = {_entity_name(str(x)) for x in entities}
    if not ent_set:
        return False
    for edge in edges:
        if not isinstance(edge, Mapping):
            return False
        src = _entity_name(str(edge.get("src", "")))
        rel = _entity_name(str(edge.get("rel", "")))
        dst = _entity_name(str(edge.get("dst", "")))
        if rel not in _ALLOWED_RELATIONS:
            return False
        if src not in ent_set or dst not in ent_set:
            return False
    return True


def run_j1_graph_target(input_artifact: Path | None, output: Path) -> Dict[str, Any]:
    texts = load_text_samples(input_artifact)
    graphs = [parse_text_to_graph(t) for t in texts]
    valid = sum(1 for g in graphs if validate_graph_schema(g))
    rel_counts: Dict[str, int] = {}
    for g in graphs:
        for _, rel, _ in canonical_edges(g):
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

    metrics = {
        "graph_count": len(graphs),
        "schema_valid_count": valid,
        "schema_valid_rate": float(valid) / float(max(1, len(graphs))),
        "mean_entities": float(sum(len(g["entities"]) for g in graphs)) / float(max(1, len(graphs))),
        "mean_edges": float(sum(len(g["edges"]) for g in graphs)) / float(max(1, len(graphs))),
        "unique_relation_count": float(len(rel_counts)),
    }
    payload = {
        "summary": {
            "run_id": "J-1",
            "name": "Graph Target",
            "generated_utc": now_utc_iso(),
            "input_artifact": str(input_artifact) if input_artifact is not None else None,
        },
        "metrics": metrics,
        "graphs": graphs,
        "relation_histogram": rel_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _rename_graph(graph: Mapping[str, Any], mapping: Mapping[str, str]) -> Dict[str, Any]:
    entities = [mapping.get(_entity_name(str(e)), _entity_name(str(e))) for e in graph.get("entities", [])]
    edges: List[Dict[str, str]] = []
    for src, rel, dst in canonical_edges(graph):
        edges.append(
            {
                "src": mapping.get(src, src),
                "rel": rel,
                "dst": mapping.get(dst, dst),
            }
        )
    return {"entities": sorted(set(entities)), "edges": edges}


def _graph_to_text(graph: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for src, rel, dst in canonical_edges(graph):
        if rel == "inside":
            parts.append(f"{src} is inside {dst}")
        elif rel == "north_of":
            parts.append(f"{src} is north of {dst}")
        elif rel == "east_of":
            parts.append(f"{src} is east of {dst}")
        elif rel == "implies":
            parts.append(f"if {src} then {dst}")
        elif rel == "equals":
            parts.append(f"{src} equals {dst}")
        elif rel == "corefers":
            parts.append(f"{src} corefers {dst}")
        else:
            parts.append(f"{src} {rel} {dst}")
    return ". ".join(parts) + ("." if parts else "")


def run_j2_paraphrase_explosion(j1_artifact: Path, output: Path, variants_per_graph: int, seed: int = 7) -> Dict[str, Any]:
    payload = _read_json(j1_artifact)
    graphs = payload.get("graphs", []) if isinstance(payload, dict) else []
    rng = random.Random(seed)

    total = 0
    invariant = 0
    sample_rows: List[Dict[str, Any]] = []

    for g in graphs:
        if not isinstance(g, Mapping):
            continue
        base_edges = canonical_edges(g)
        entities = [_entity_name(str(e)) for e in g.get("entities", [])]
        if not entities:
            continue
        for _ in range(max(1, variants_per_graph)):
            labels = [f"e{i}" for i in range(len(entities))]
            rng.shuffle(labels)
            mapping = dict(zip(entities, labels))
            paraphrased = _rename_graph(g, mapping)
            text = _graph_to_text(paraphrased)
            reparsed = parse_text_to_graph(text)
            reverse = {v: k for k, v in mapping.items()}
            restored = _rename_graph(reparsed, reverse)
            ok = canonical_edges(restored) == base_edges
            total += 1
            invariant += int(ok)
            if len(sample_rows) < 8:
                sample_rows.append({"text": text, "invariant": ok})

    metrics = {
        "variant_count": float(total),
        "invariant_count": float(invariant),
        "invariance_rate": float(invariant) / float(max(1, total)),
        "variants_per_graph": float(max(1, variants_per_graph)),
    }
    out = {
        "summary": {
            "run_id": "J-2",
            "name": "Paraphrase Explosion",
            "generated_utc": now_utc_iso(),
            "j1_artifact": str(j1_artifact),
        },
        "metrics": metrics,
        "samples": sample_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_j3_stopgrad_isolation(source_script: Path, output: Path) -> Dict[str, Any]:
    text = source_script.read_text(encoding="utf-8")
    has_disabled_ctx = bool(re.search(r"with\s+adapter_disabled\(model\)\s*:", text))
    has_frozen_params = bool(
        re.search(r"for\s+p\s+in\s+model\.parameters\(\)\s*:\s*p\.requires_grad\s*=\s*False", text)
    )
    has_quantize_detach = bool(re.search(r"\(\s*z_q\s*-\s*z\s*\)\.detach\(\)", text))

    metrics = {
        "has_adapter_disabled_context": float(1 if has_disabled_ctx else 0),
        "has_model_freeze_guard": float(1 if has_frozen_params else 0),
        "has_quantize_detach": float(1 if has_quantize_detach else 0),
        "stopgrad_contract_pass": float(1 if (has_disabled_ctx and has_frozen_params and has_quantize_detach) else 0),
    }
    out = {
        "summary": {
            "run_id": "J-3",
            "name": "Stop-Grad Isolation Gate",
            "generated_utc": now_utc_iso(),
            "source_script": str(source_script),
        },
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_j4_operator_curriculum(output: Path, dataset_output: Path, per_operator: int, seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    operators = {
        "equality": ["{a} equals {b}", "{a} is the same as {b}"],
        "coreference": ["{a} met {b}. Later it thanked {b}", "{a} saw {b} and then they left"],
        "containment": ["{a} is inside {b}", "{a} is contained by {b}"],
        "transitivity": ["{a} is north of {b}. {b} is north of {c}", "if {a} then {b}. if {b} then {c}"],
        "negation": ["not {a}", "it is false that {a}"],
    }

    rows: List[Dict[str, Any]] = []
    for op, templates in operators.items():
        for i in range(max(1, per_operator)):
            a = f"x{i}"
            b = f"y{i}"
            c = f"z{i}"
            template = rng.choice(templates)
            prompt = template.format(a=a, b=b, c=c)
            rows.append({"operator": op, "prompt": prompt})

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    with dataset_output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    op_counts: Dict[str, int] = {}
    for row in rows:
        op = str(row["operator"])
        op_counts[op] = op_counts.get(op, 0) + 1

    out = {
        "summary": {
            "run_id": "J-4",
            "name": "Operator Curriculum",
            "generated_utc": now_utc_iso(),
            "dataset_output": str(dataset_output),
        },
        "metrics": {
            "sample_count": float(len(rows)),
            "operator_count": float(len(op_counts)),
            "min_operator_coverage": float(min(op_counts.values()) if op_counts else 0),
            "max_operator_coverage": float(max(op_counts.values()) if op_counts else 0),
        },
        "operator_histogram": op_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
