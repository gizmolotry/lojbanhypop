from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from lojban_evolution.l_series import compute_scope_violation_components
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata

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
    "before",
    "after",
    "permits",
    "forbids",
    "parent_of",
    "child_of",
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
    hyperedges: List[Dict[str, Any]] = []

    # Function-style n-ary predicates, e.g. contains(a,b,c)
    for rel, args_blob in re.findall(r"\b([A-Za-z][\w-]*)\s*\(\s*([^)]+)\s*\)", s):
        rel_n = _entity_name(rel)
        args = [_entity_name(x) for x in re.split(r"\s*,\s*", args_blob) if _entity_name(x)]
        if len(args) >= 2 and rel_n in _ALLOWED_RELATIONS:
            entities.update(args)
            hyperedges.append({"rel": rel_n, "args": args})

    patterns: List[Tuple[str, str]] = [
        (r"\b([A-Za-z][\w-]*)\s+contains\s+([A-Za-z][\w-]*)\b", "contains"),
        (r"\binside\s+the\s+([A-Za-z][\w-]*),\s+there\s+is\s+(?:a|an)\s+([A-Za-z][\w-]*)\b", "contains"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+situated\s+within\s+([A-Za-z][\w-]*)\b", "inside"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+inside\s+the\s+([A-Za-z][\w-]*)\b", "inside"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+inside\s+([A-Za-z][\w-]*)\b", "inside"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+north\s+of\s+([A-Za-z][\w-]*)\b", "north_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+south\s+of\s+([A-Za-z][\w-]*)\b", "south_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+east\s+of\s+([A-Za-z][\w-]*)\b", "east_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+west\s+of\s+([A-Za-z][\w-]*)\b", "west_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+before\s+([A-Za-z][\w-]*)\b", "before"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+after\s+([A-Za-z][\w-]*)\b", "after"),
        (r"\b([A-Za-z][\w-]*)\s+permits\s+([A-Za-z][\w-]*)\b", "permits"),
        (r"\b([A-Za-z][\w-]*)\s+forbids\s+([A-Za-z][\w-]*)\b", "forbids"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+parent\s+of\s+([A-Za-z][\w-]*)\b", "parent_of"),
        (r"\b([A-Za-z][\w-]*)\s+is\s+child\s+of\s+([A-Za-z][\w-]*)\b", "child_of"),
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

    if hyperedges:
        return {"entities": sorted(entities), "edges": hyperedges}
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


def canonical_hyperedges(graph: Mapping[str, Any]) -> List[Tuple[str, Tuple[str, ...]]]:
    out: List[Tuple[str, Tuple[str, ...]]] = []
    edges = graph.get("edges") if isinstance(graph, Mapping) else None
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            rel = _entity_name(str(e.get("rel", "")))
            args_raw = e.get("args")
            if isinstance(args_raw, list) and args_raw:
                args = tuple(_entity_name(str(a)) for a in args_raw if _entity_name(str(a)))
                if len(args) >= 2:
                    out.append((rel, args))
                continue
            src = _entity_name(str(e.get("src", "")))
            dst = _entity_name(str(e.get("dst", "")))
            if src and dst:
                out.append((rel, (src, dst)))
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
        rel = _entity_name(str(edge.get("rel", "")))
        if rel not in _ALLOWED_RELATIONS:
            return False
        args_raw = edge.get("args")
        if isinstance(args_raw, list):
            args = [_entity_name(str(a)) for a in args_raw]
            if len(args) < 2:
                return False
            if any(a not in ent_set for a in args):
                return False
        else:
            src = _entity_name(str(edge.get("src", "")))
            dst = _entity_name(str(edge.get("dst", "")))
            if src not in ent_set or dst not in ent_set:
                return False
    return True


def run_j1_graph_target(input_artifact: Path | None, output: Path) -> Dict[str, Any]:
    assert_output_path_allowed("J", output)
    texts = load_text_samples(input_artifact)
    graphs = [parse_text_to_graph(t) for t in texts]
    valid = sum(1 for g in graphs if validate_graph_schema(g))
    rel_counts: Dict[str, int] = {}
    for g in graphs:
        for rel, _args in canonical_hyperedges(g):
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
            **series_metadata("J", "invariance_data", "scripts/eval_j_1.py"),
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
    for rel, args in canonical_hyperedges(graph):
        mapped_args = [mapping.get(a, a) for a in args]
        if len(mapped_args) == 2:
            edges.append({"src": mapped_args[0], "rel": rel, "dst": mapped_args[1]})
        else:
            edges.append({"rel": rel, "args": mapped_args})
    return {"entities": sorted(set(entities)), "edges": edges}


def _graph_to_text(graph: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for rel, args in canonical_hyperedges(graph):
        if len(args) == 2:
            src, dst = args
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
        else:
            args_s = ", ".join(args)
            parts.append(f"{rel}({args_s})")
    return ". ".join(parts) + ("." if parts else "")


def run_j2_paraphrase_explosion(j1_artifact: Path, output: Path, variants_per_graph: int, seed: int = 7) -> Dict[str, Any]:
    assert_output_path_allowed("J", output)
    payload = _read_json(j1_artifact)
    graphs = payload.get("graphs", []) if isinstance(payload, dict) else []
    rng = random.Random(seed)

    total = 0
    invariant = 0
    sample_rows: List[Dict[str, Any]] = []

    for g in graphs:
        if not isinstance(g, Mapping):
            continue
        base_edges = canonical_hyperedges(g)
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
            ok = canonical_hyperedges(restored) == base_edges
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
            **series_metadata("J", "invariance_data", "scripts/eval_j_2.py"),
        },
        "metrics": metrics,
        "samples": sample_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_j3_stopgrad_isolation(source_script: Path, output: Path) -> Dict[str, Any]:
    assert_output_path_allowed("J", output)
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
            **series_metadata("J", "invariance_data", "scripts/eval_j_3.py"),
        },
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_j4_operator_curriculum(output: Path, dataset_output: Path, per_operator: int, seed: int = 7) -> Dict[str, Any]:
    assert_output_path_allowed("J", output)
    assert_output_path_allowed("J", dataset_output)
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
            **series_metadata("J", "invariance_data", "scripts/eval_j_4.py"),
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


def _depth_level_weights() -> List[Tuple[int, float]]:
    # Scheduled depth mix: 40/30/20/10 for levels 1..4.
    return [(1, 0.40), (2, 0.30), (3, 0.20), (4, 0.10)]


def _sample_depth(rng: random.Random) -> int:
    levels = _depth_level_weights()
    r = rng.random()
    acc = 0.0
    for depth, w in levels:
        acc += w
        if r <= acc:
            return depth
    return levels[-1][0]


def _entity_pool(domain: str, idx: int) -> List[str]:
    if domain == "kinship":
        return [f"parent{idx}", f"child{idx}", f"sibling{idx}", f"guardian{idx}"]
    if domain == "temporal":
        return [f"event{idx}", f"milestone{idx}", f"deadline{idx}", f"review{idx}"]
    if domain == "legal_permission":
        return [f"actor{idx}", f"role{idx}", f"permit{idx}", f"policy{idx}"]
    return [f"obj{idx}", f"container{idx}", f"node{idx}", f"state{idx}"]


def _rel_templates(rel: str) -> List[str]:
    m = {
        "contains": [
            "{a} contains {b}",
            "inside the {a}, there is a {b}",
            "{b} is situated within {a}",
        ],
        "inside": [
            "{a} is inside {b}",
            "{a} is situated within {b}",
            "inside the {b}, there is a {a}",
        ],
        "implies": [
            "if {a} then {b}",
            "whenever {a}, then {b}",
            "{a} implies {b}",
        ],
        "and": [
            "{a} and {b}",
            "both {a} and {b}",
            "{a} together with {b}",
        ],
        "or": [
            "{a} or {b}",
            "either {a} or {b}",
            "{a}, alternatively {b}",
        ],
        "before": [
            "{a} is before {b}",
            "{a} happens before {b}",
            "before {b}, {a}",
        ],
        "permits": [
            "{a} permits {b}",
            "{a} allows {b}",
            "{b} is authorized by {a}",
        ],
        "forbids": [
            "{a} forbids {b}",
            "{a} disallows {b}",
            "{b} is forbidden by {a}",
        ],
        "parent_of": [
            "{a} is parent of {b}",
            "{a} is the parent of {b}",
            "{b} is child of {a}",
        ],
    }
    return m.get(rel, ["{a} " + rel + " {b}"])


def _render_sentence(rel: str, a: str, b: str, template_idx: int) -> str:
    templates = _rel_templates(rel)
    t = templates[template_idx % len(templates)]
    return t.format(a=a, b=b)


def _render_graph_sentences(graph: Mapping[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rel, args in canonical_hyperedges(graph):
        if len(args) == 2:
            src, dst = args
            templates = _rel_templates(rel)
            tidx = rng.randint(0, max(0, len(templates) - 1))
            text = _render_sentence(rel, src, dst, tidx)
            out.append({"args": [src, dst], "rel": rel, "template_idx": tidx, "text": text})
        else:
            text = f"{rel}(" + ", ".join(args) + ")"
            out.append({"args": list(args), "rel": rel, "template_idx": 0, "text": text})
    return out


def _sentences_to_prompt(rows: List[Dict[str, Any]]) -> str:
    chunks = [str(r.get("text", "")).strip().rstrip(".") for r in rows]
    chunks = [c for c in chunks if c]
    return ". ".join(chunks) + ("." if chunks else "")


def _minimal_edit_mutation(row: Mapping[str, Any]) -> Dict[str, str]:
    args = row.get("args")
    if isinstance(args, list) and len(args) >= 2:
        arg_list = [str(x) for x in args]
    else:
        arg_list = [str(row.get("src", "a")), str(row.get("dst", "b"))]
    src = arg_list[0]
    dst = arg_list[1]
    rel = str(row.get("rel", "implies"))
    tidx = int(row.get("template_idx", 0))
    if isinstance(args, list) and len(args) >= 3:
        swapped = list(arg_list)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        return {"rel": rel, "text": f"{rel}(" + ", ".join(swapped) + ")"}
    if rel in {"before", "after", "permits", "forbids", "parent_of", "child_of", "implies", "contains", "inside"}:
        # Single truth-critical mutation: argument swap in the same template.
        return {"rel": rel, "text": _render_sentence(rel, dst, src, tidx)}
    if rel == "and":
        return {"rel": "or", "text": _render_sentence("or", src, dst, tidx)}
    if rel == "or":
        return {"rel": "and", "text": _render_sentence("and", src, dst, tidx)}
    return {"rel": "not", "text": _render_sentence("not", src, dst, 0)}


def _scope_trace_for_depth(graph: Mapping[str, Any], depth: int) -> List[str]:
    vars_sorted = sorted({_entity_name(str(e)) for e in graph.get("entities", [])})
    if len(vars_sorted) < 2:
        vars_sorted = (vars_sorted + ["v0", "v1"])[:2]
    v0 = f"VAR_{vars_sorted[0]}"
    v1 = f"VAR_{vars_sorted[1]}"
    if depth <= 1:
        return ["FORALL", v0, "SCOPE_OPEN", v0, v1, "SCOPE_CLOSE"]
    if depth == 2:
        return ["FORALL", v0, "SCOPE_OPEN", "EXISTS", v1, "SCOPE_OPEN", v0, v1, "SCOPE_CLOSE", "SCOPE_CLOSE"]
    if depth == 3:
        return [
            "FORALL",
            v0,
            "SCOPE_OPEN",
            "EXISTS",
            v1,
            "SCOPE_OPEN",
            "NOT",
            v0,
            "AND",
            v1,
            "SCOPE_CLOSE",
            "SCOPE_CLOSE",
        ]
    # Depth 4: quantifier nesting with explicit balanced closes.
    v2 = f"VAR_{vars_sorted[2] if len(vars_sorted) > 2 else vars_sorted[0] + '_2'}"
    return [
        "FORALL",
        v0,
        "SCOPE_OPEN",
        "EXISTS",
        v1,
        "SCOPE_OPEN",
        "FORALL",
        v2,
        "SCOPE_OPEN",
        v0,
        "IMPLIES",
        v1,
        "AND",
        v2,
        "SCOPE_CLOSE",
        "SCOPE_CLOSE",
        "SCOPE_CLOSE",
    ]


def _build_problem(domain: str, depth: int, idx: int) -> Dict[str, Any]:
    ents = _entity_pool(domain, idx)
    a, b, c, d = [_entity_name(x) for x in ents]

    if domain == "kinship":
        base_rel = "parent_of"
    elif domain == "temporal":
        base_rel = "before"
    elif domain == "legal_permission":
        base_rel = "permits"
    else:
        base_rel = "contains"

    edges: List[Dict[str, str]] = []
    if depth == 1:
        edges.append({"src": a, "rel": base_rel, "dst": b})
    elif depth == 2:
        edges.extend(
            [
                {"src": a, "rel": base_rel, "dst": b},
                {"src": b, "rel": "implies", "dst": c},
            ]
        )
    elif depth == 3:
        edges.extend(
            [
                {"rel": base_rel, "args": [a, b]},
                {"rel": "implies", "args": [b, c]},
                {"rel": "before", "args": [c, d]},
                {"rel": "and", "args": [a, b, c]},
            ]
        )
    else:
        edges.extend(
            [
                {"rel": "implies", "args": [a, b]},
                {"rel": base_rel, "args": [b, c]},
                {"rel": "implies", "args": [c, d]},
                {"rel": "before", "args": [a, d]},
                {"rel": "or", "args": [a, b, c, d]},
            ]
        )
    return {"entities": sorted({a, b, c, d}), "edges": edges}


def _make_foil_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    edges = list(canonical_hyperedges(graph))
    if not edges:
        return {"entities": list(graph.get("entities", [])), "edges": []}
    rel, args = edges[0]
    args_l = list(args)
    if len(args_l) >= 2:
        args_l[0], args_l[1] = args_l[1], args_l[0]
    if rel == "and":
        rel2 = "or"
    elif rel == "or":
        rel2 = "and"
    else:
        rel2 = rel
    foil_edges = [(rel2, tuple(args_l))] + edges[1:]
    foil_graph = {
        "entities": sorted({_entity_name(e) for e in graph.get("entities", [])}),
        "edges": [{"rel": r, "args": list(a)} for r, a in foil_edges],
    }
    return foil_graph


def _edge_jaccard(a: Iterable[Tuple[str, str, str]], b: Iterable[Tuple[str, str, str]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(max(1, len(sa | sb)))


def _truth_score(candidate_graph: Mapping[str, Any], target_graph: Mapping[str, Any]) -> float:
    # Soft score in [0,1], used for foil discrimination proxy.
    return _edge_jaccard(canonical_hyperedges(candidate_graph), canonical_hyperedges(target_graph))


def run_j5_adversarial_synthesis(
    output: Path,
    dataset_output: Path,
    sample_count: int = 256,
    seed: int = 7,
    novelty_threshold: float = 0.30,
    strict_depth_balance: bool = True,
    max_attempt_multiplier: int = 100,
) -> Dict[str, Any]:
    assert_output_path_allowed("J", output)
    assert_output_path_allowed("J", dataset_output)
    rng = random.Random(seed)
    domains = ["kinship", "temporal", "legal_permission"]

    accepted_rows: List[Dict[str, Any]] = []
    seen_hashes: set[Tuple[Tuple[str, str, str], ...]] = set()
    by_depth_total: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    by_depth_accept: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    target_by_depth: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    base = int(sample_count) // 4
    rem = int(sample_count) % 4
    for depth in (1, 2, 3, 4):
        target_by_depth[depth] = base
    for depth in (1, 2, 3, 4):
        if rem <= 0:
            break
        target_by_depth[depth] += 1
        rem -= 1

    schema_valid_count = 0
    solver_consistent_count = 0
    dedupe_reject_count = 0
    novelty_values: List[float] = []
    foil_pairs_ok_total = 0
    foil_pairs_ok_accept = 0
    edit_distance_ok = 0
    scope_component_sums = {
        "scope_unbalanced": 0.0,
        "scope_lifetime": 0.0,
        "scope_unbound": 0.0,
        "scope_quantifier_assoc": 0.0,
        "scope_shadowing": 0.0,
    }
    scope_component_by_depth = {
        d: {
            "scope_unbalanced": 0.0,
            "scope_lifetime": 0.0,
            "scope_unbound": 0.0,
            "scope_quantifier_assoc": 0.0,
            "scope_shadowing": 0.0,
            "n": 0.0,
        }
        for d in (1, 2, 3, 4)
    }

    total_target = max(1, int(sample_count))
    max_attempts = max(total_target, total_target * max(1, int(max_attempt_multiplier)))
    attempts = 0
    i = 0
    while attempts < max_attempts:
        if strict_depth_balance:
            if all(by_depth_accept[d] >= target_by_depth[d] for d in (1, 2, 3, 4)):
                break
            underfilled = [d for d in (1, 2, 3, 4) if by_depth_accept[d] < target_by_depth[d]]
            depth = int(rng.choice(underfilled))
        else:
            if len(accepted_rows) >= total_target:
                break
            depth = _sample_depth(rng)

        attempts += 1
        by_depth_total[depth] += 1
        domain = domains[i % len(domains)]
        i += 1

        graph = _build_problem(domain=domain, depth=depth, idx=i)
        sentence_rows = _render_graph_sentences(graph, rng)
        prompt = _sentences_to_prompt(sentence_rows)
        reparsed = parse_text_to_graph(prompt)
        foil_graph = _make_foil_graph(graph)
        foil_rows = list(sentence_rows)
        if foil_rows:
            foil_rows[0] = {
                **foil_rows[0],
                **_minimal_edit_mutation(foil_rows[0]),
            }
        foil_prompt = _sentences_to_prompt(foil_rows)
        foil_reparsed = parse_text_to_graph(foil_prompt)
        edit_distance_ok += int(abs(len(prompt) - len(foil_prompt)) <= 4)

        schema_valid = int(validate_graph_schema(graph) and validate_graph_schema(reparsed))
        schema_valid_count += schema_valid

        target_edges = canonical_hyperedges(graph)
        reparsed_edges = canonical_hyperedges(reparsed)
        solver_consistent = int(target_edges == reparsed_edges)
        solver_consistent_count += solver_consistent

        h = tuple(target_edges)
        duplicate = h in seen_hashes
        if duplicate:
            dedupe_reject_count += 1

        if seen_hashes:
            novelty = 1.0 - max(_edge_jaccard(target_edges, prev) for prev in seen_hashes)
        else:
            novelty = 1.0
        novelty_values.append(float(novelty))

        true_score = _truth_score(reparsed, graph)
        false_score = _truth_score(foil_reparsed, graph)
        foil_good = int(true_score > false_score)
        foil_pairs_ok_total += foil_good

        scope_trace = _scope_trace_for_depth(graph, depth)
        scope_comp = compute_scope_violation_components(scope_trace)
        for k in scope_component_sums:
            scope_component_sums[k] += float(scope_comp[k])
            scope_component_by_depth[depth][k] += float(scope_comp[k])
        scope_component_by_depth[depth]["n"] += 1.0

        accepted = bool(schema_valid and solver_consistent and (not duplicate) and (novelty >= float(novelty_threshold)))
        if strict_depth_balance and accepted and by_depth_accept[depth] >= target_by_depth[depth]:
            accepted = False
        if accepted:
            seen_hashes.add(h)
            by_depth_accept[depth] += 1
            foil_pairs_ok_accept += foil_good
            accepted_rows.append(
                {
                    "domain": domain,
                    "depth": depth,
                    "prompt": prompt,
                    "foil_prompt": foil_prompt,
                    "factor_graph": graph,
                    "foil_graph": foil_graph,
                    "novelty": novelty,
                    "true_score": true_score,
                    "false_score": false_score,
                    "scope_trace": scope_trace,
                    "scope_components": scope_comp,
                }
            )

    total = max(1, attempts)
    accepted_total = max(1, len(accepted_rows))
    accept_rate_by_depth = {
        str(k): float(by_depth_accept[k]) / float(max(1, by_depth_total[k]))
        for k in sorted(by_depth_total.keys())
    }
    scope_by_depth_components = {}
    for depth, vals in scope_component_by_depth.items():
        n = max(1.0, float(vals["n"]))
        scope_by_depth_components[str(depth)] = {
            "scope_unbalanced": float(vals["scope_unbalanced"]) / n,
            "scope_lifetime": float(vals["scope_lifetime"]) / n,
            "scope_unbound": float(vals["scope_unbound"]) / n,
            "scope_quantifier_assoc": float(vals["scope_quantifier_assoc"]) / n,
            "scope_shadowing": float(vals["scope_shadowing"]) / n,
        }

    metrics = {
        "problem_count": float(sample_count),
        "attempt_count": float(attempts),
        "accepted_count": float(len(accepted_rows)),
        "generator_accept_rate": float(len(accepted_rows)) / float(max(1, attempts)),
        "depth_balance_target": {str(k): float(v) for k, v in target_by_depth.items()},
        "depth_accept_count": {str(k): float(by_depth_accept[k]) for k in sorted(by_depth_accept.keys())},
        "schema_valid_rate": float(schema_valid_count) / float(total),
        "solver_consistency_rate": float(solver_consistent_count) / float(total),
        "novelty_mean": float(sum(novelty_values) / float(max(1, len(novelty_values)))),
        "novelty_threshold": float(novelty_threshold),
        "dedupe_reject_count": float(dedupe_reject_count),
        "accept_rate_by_depth": accept_rate_by_depth,
        # Deprecated alias retained for downstream compatibility. This is acceptance rate by depth, not scope quality.
        "scope_by_depth": accept_rate_by_depth,
        "scope_components_mean": {k: float(v) / float(total) for k, v in scope_component_sums.items()},
        "scope_components_by_depth": scope_by_depth_components,
        "foil_minimal_edit_rate": float(edit_distance_ok) / float(total),
        "foil_pair_accuracy_total": float(foil_pairs_ok_total) / float(total),
        "foil_pair_accuracy_accepted": float(foil_pairs_ok_accept) / float(accepted_total),
        "accepted_foil_pair_accuracy": float(foil_pairs_ok_accept) / float(accepted_total),
        # Deprecated alias retained for downstream compatibility. This is accepted-pair accuracy, not ROC-AUC.
        "foil_auc": float(foil_pairs_ok_accept) / float(accepted_total),
        "domain_coverage": float(len({row["domain"] for row in accepted_rows})),
    }

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    with dataset_output.open("w", encoding="utf-8") as f:
        for row in accepted_rows:
            f.write(json.dumps(row) + "\n")

    out = {
        "summary": {
            "run_id": "J-5",
            "name": "Adversarial Problem Synthesis",
            "generated_utc": now_utc_iso(),
            "dataset_output": str(dataset_output),
            **series_metadata("J", "invariance_data", "scripts/eval_j_5.py"),
        },
        "metrics": metrics,
        "samples": accepted_rows[:16],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
