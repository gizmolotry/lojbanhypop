from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from .storage import StoragePath, join_path, make_dirs, write_text


SEED_LANGUAGE = {
    "ASSUME",
    "CHECK",
    "VERIFY",
    "REF",
    "DEF",
    "RECALL",
    "AND",
    "OR",
    "NOT",
    "IF",
    "THEN",
    "ELSE",
    "BIND",
    "POINTER",
    "STATE",
    "EVENT",
    "BELIEF",
    "CLAIM",
    "CONSISTENT",
    "INCONSISTENT",
    "ANS",
}


@dataclass(frozen=True)
class Problem:
    problem_id: int
    prompt: str
    answer: str
    trace: Tuple[str, ...]


@dataclass
class LanguageSpec:
    base_tokens: set[str] = field(default_factory=lambda: set(SEED_LANGUAGE))
    macros: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def token_count(self) -> int:
        return len(self.base_tokens) + len(self.macros)


@dataclass
class Metrics:
    accuracy: float
    avg_tokens: float
    parse_success_rate: float

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "avg_tokens": self.avg_tokens,
            "parse_success_rate": self.parse_success_rate,
        }


@dataclass
class Proposal:
    macro_name: str
    pattern: Tuple[str, ...]
    support: int

    def to_dict(self) -> dict:
        return {
            "macro_name": self.macro_name,
            "pattern": list(self.pattern),
            "support": self.support,
        }


@dataclass
class IterationRecord:
    iteration: int
    before: Metrics
    after: Metrics
    accepted: List[Proposal]
    language_size: int

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "accepted": [p.to_dict() for p in self.accepted],
            "language_size": self.language_size,
        }


def split_dataset(problems: Sequence[Problem]) -> tuple[list[Problem], list[Problem], list[Problem]]:
    n = len(problems)
    train = list(problems[: int(n * 0.6)])
    val = list(problems[int(n * 0.6) : int(n * 0.8)])
    test = list(problems[int(n * 0.8) :])
    return train, val, test


def generate_dataset(size: int = 1000, seed: int = 7) -> list[Problem]:
    rng = random.Random(seed)
    problems: list[Problem] = []
    for i in range(size):
        mode = rng.choice(["winograd", "multi_agent", "knights_knaves"])
        if mode == "winograd":
            prompt, answer, trace = _generate_winograd_problem(rng)
        elif mode == "multi_agent":
            prompt, answer, trace = _generate_multi_agent_problem(rng)
        else:
            prompt, answer, trace = _generate_knights_knaves_problem(rng)
        problems.append(Problem(problem_id=i, prompt=prompt, answer=answer, trace=trace))
    return problems


def _generate_winograd_problem(rng: random.Random) -> tuple[str, str, Tuple[str, ...]]:
    schemas = [
        (
            "The city councilmen refused the demonstrators a permit because they feared violence. Who feared violence?",
            "councilmen",
            "E1",
        ),
        (
            "The city councilmen refused the demonstrators a permit because they advocated revolution. Who advocated revolution?",
            "demonstrators",
            "E2",
        ),
        (
            "The trophy does not fit in the suitcase because it is too big. What is too big?",
            "trophy",
            "E1",
        ),
        (
            "The trophy does not fit in the suitcase because it is too small. What is too small?",
            "suitcase",
            "E2",
        ),
    ]
    prompt, answer, winner = rng.choice(schemas)
    resolve = "RESOLVE_PRON_E1" if winner == "E1" else "RESOLVE_PRON_E2"
    answer_token = "ANS_E1" if winner == "E1" else "ANS_E2"
    trace = (
        "TASK_WINOGRAD",
        "BIND_E1",
        "BIND_E2",
        "LINK_CAUSAL",
        "PRONOUN_REF",
        resolve,
        "VERIFY_ID",
        answer_token,
    )
    return prompt, answer, trace


def _generate_multi_agent_problem(rng: random.Random) -> tuple[str, str, Tuple[str, ...]]:
    names = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank"]
    places = ["box", "drawer", "cabinet", "shelf", "desk"]
    mover, observer, bystander = rng.sample(names, 3)
    start_loc, end_loc = rng.sample(places, 2)
    seen_move = rng.choice([True, False, False])  # Bias toward false-belief traps.
    if seen_move:
        prompt = (
            f"{observer} puts the red ball in the {start_loc}. "
            f"{mover} moves it to the {end_loc} while {observer} is watching. "
            f"{bystander} takes the blue ball. "
            f"Where does {observer} think the red ball is?"
        )
        answer = end_loc
        belief_token = "MENTAL_STATE_E1_LOC2"
    else:
        prompt = (
            f"{observer} puts the red ball in the {start_loc}. "
            f"{mover} moves it to the {end_loc} while {observer} is outside. "
            f"{bystander} takes the blue ball. "
            f"Where does {observer} think the red ball is?"
        )
        answer = start_loc
        belief_token = "MENTAL_STATE_E1_LOC1"

    answer_token = "ANS_LOC2" if answer == end_loc else "ANS_LOC1"
    visibility = "OBS_PRESENT" if seen_move else "OBS_ABSENT"
    trace = (
        "TASK_MULTI_AGENT",
        "BIND_AGENT_E1",
        "BIND_AGENT_E2",
        "BIND_OBJ_O1",
        "STATE_OBJ_LOC1",
        "EVENT_MOVE_LOC2",
        visibility,
        belief_token,
        "VERIFY_ID",
        answer_token,
    )
    return prompt, answer, trace


def _generate_knights_knaves_problem(rng: random.Random) -> tuple[str, str, Tuple[str, ...]]:
    templates = [
        (
            "A says: 'B is a knave.' B says: 'A and C are both knights.' C says: 'A is a knave.' Who is what?",
            "A=knight,B=knave,C=knight",
            "SOLUTION_AK_BN_CK",
        ),
        (
            "A says: 'B is a knight.' B says: 'C is a knight.' C says: 'A is a knave.' Who is what?",
            "A=knight,B=knight,C=knave",
            "SOLUTION_AK_BK_CN",
        ),
        (
            "A says: 'B is a knave.' B says: 'C is a knight.' C says: 'A is a knight.' Who is what?",
            "A=knight,B=knave,C=knight",
            "SOLUTION_AK_BN_CK",
        ),
    ]
    prompt, answer, solution_token = rng.choice(templates)
    trace = (
        "TASK_KNIGHTS",
        "BIND_A",
        "BIND_B",
        "BIND_C",
        "CLAIM_GRAPH",
        "ASSUME_BRANCH",
        "CONSISTENCY_CHECK",
        solution_token,
        "VERIFY_ID",
        "ANS_ROLE_MAP",
    )
    return prompt, answer, trace


def _macro_patterns_sorted(macros: Dict[str, Tuple[str, ...]]) -> list[tuple[str, Tuple[str, ...]]]:
    return sorted(macros.items(), key=lambda kv: len(kv[1]), reverse=True)


def encode_trace(trace: Sequence[str], language: LanguageSpec) -> list[str]:
    encoded: list[str] = []
    i = 0
    patterns = _macro_patterns_sorted(language.macros)
    while i < len(trace):
        matched = False
        for macro_name, pattern in patterns:
            p_len = len(pattern)
            if p_len == 0:
                continue
            if tuple(trace[i : i + p_len]) == pattern:
                encoded.append(macro_name)
                i += p_len
                matched = True
                break
        if not matched:
            encoded.append(trace[i])
            i += 1
    return encoded


def decode_trace(encoded: Sequence[str], language: LanguageSpec) -> list[str]:
    decoded: list[str] = []
    for token in encoded:
        if token in language.macros:
            decoded.extend(language.macros[token])
        else:
            decoded.append(token)
    return decoded


def evaluate(problems: Sequence[Problem], language: LanguageSpec) -> Metrics:
    total = len(problems)
    if total == 0:
        return Metrics(accuracy=0.0, avg_tokens=0.0, parse_success_rate=0.0)
    correct = 0
    parseable = 0
    token_total = 0
    for p in problems:
        encoded = encode_trace(p.trace, language)
        decoded = decode_trace(encoded, language)
        ok = tuple(decoded) == p.trace
        if ok:
            parseable += 1
            correct += 1
        token_total += len(encoded)
    return Metrics(
        accuracy=correct / total,
        avg_tokens=token_total / total,
        parse_success_rate=parseable / total,
    )


def _all_ngrams(tokens: Sequence[str], min_n: int = 2, max_n: int = 4) -> Iterable[Tuple[str, ...]]:
    for n in range(min_n, max_n + 1):
        if n > len(tokens):
            break
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])


def propose_macros(
    language: LanguageSpec,
    training: Sequence[Problem],
    next_macro_index: int,
    top_k: int = 10,
    min_support: int = 30,
) -> tuple[list[Proposal], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    existing_patterns = set(language.macros.values())

    for p in training:
        encoded = encode_trace(p.trace, language)
        for ng in _all_ngrams(encoded):
            if ng in existing_patterns:
                continue
            if any(token in language.macros for token in ng):
                continue
            counts[ng] = counts.get(ng, 0) + 1

    candidates = [(pattern, support) for pattern, support in counts.items() if support >= min_support]
    candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    proposals: list[Proposal] = []

    for pattern, support in candidates[:top_k]:
        macro_name = f"M{next_macro_index:03d}"
        proposals.append(Proposal(macro_name=macro_name, pattern=pattern, support=support))
        next_macro_index += 1
    return proposals, next_macro_index


def is_pareto_improvement(new: Metrics, baseline: Metrics, tolerance: float = 1e-12) -> bool:
    improved = False
    if new.accuracy + tolerance < baseline.accuracy:
        return False
    if new.parse_success_rate + tolerance < baseline.parse_success_rate:
        return False
    if new.avg_tokens + tolerance < baseline.avg_tokens:
        improved = True
    if new.accuracy > baseline.accuracy + tolerance:
        improved = True
    if new.parse_success_rate > baseline.parse_success_rate + tolerance:
        improved = True
    return improved


def run_experiment(
    output_root: StoragePath,
    iterations: int = 6,
    seed: int = 7,
    dataset_size: int = 1000,
    max_accept_per_iteration: int = 3,
) -> dict:
    problems = generate_dataset(size=dataset_size, seed=seed)
    train, val, test = split_dataset(problems)
    language = LanguageSpec()
    history: list[IterationRecord] = []
    next_macro_index = 1

    for iteration in range(iterations):
        baseline = evaluate(val, language)
        before_metrics = baseline
        proposals, next_macro_index = propose_macros(language, train, next_macro_index=next_macro_index)
        accepted: list[Proposal] = []

        for proposal in proposals:
            temp = LanguageSpec(base_tokens=set(language.base_tokens), macros=dict(language.macros))
            temp.macros[proposal.macro_name] = proposal.pattern
            candidate_metrics = evaluate(val, temp)
            if is_pareto_improvement(candidate_metrics, baseline):
                language = temp
                baseline = candidate_metrics
                accepted.append(proposal)
                if len(accepted) >= max_accept_per_iteration:
                    break

        after = evaluate(val, language)
        history.append(
            IterationRecord(
                iteration=iteration,
                before=before_metrics,
                after=after,
                accepted=accepted,
                language_size=language.token_count(),
            )
        )

    test_metrics = evaluate(test, language)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "iterations": iterations,
            "seed": seed,
            "dataset_size": dataset_size,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "max_accept_per_iteration": max_accept_per_iteration,
        },
        "final_language": {
            "base_token_count": len(language.base_tokens),
            "macro_count": len(language.macros),
            "total_token_count": language.token_count(),
            "macros": {k: list(v) for k, v in language.macros.items()},
        },
        "history": [record.to_dict() for record in history],
        "test_metrics": test_metrics.to_dict(),
    }

    run_dir = join_path(output_root, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    make_dirs(run_dir, parents=True, exist_ok=True)
    payload["run_dir"] = str(run_dir)
    payload["dataset_fingerprint"] = _dataset_fingerprint(problems)
    write_text(join_path(run_dir, "history.json"), json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(join_path(run_dir, "summary.md"), payload)
    return payload


def _dataset_fingerprint(problems: Sequence[Problem]) -> str:
    h = hashlib.sha256()
    for p in problems:
        row = {
            "problem_id": p.problem_id,
            "prompt": p.prompt,
            "answer": p.answer,
            "trace": list(p.trace),
        }
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _write_summary(path: StoragePath, payload: dict) -> None:
    config = payload["config"]
    final_language = payload["final_language"]
    test = payload["test_metrics"]
    history = payload["history"]

    lines = [
        "# Language Evolution Run Summary",
        "",
        f"- Dataset size: {config['dataset_size']} (train={config['train_size']}, val={config['val_size']}, test={config['test_size']})",
        f"- Iterations: {config['iterations']}",
        f"- Final vocabulary: {final_language['total_token_count']} tokens ({final_language['base_token_count']} base + {final_language['macro_count']} macros)",
        f"- Test accuracy: {test['accuracy']:.4f}",
        f"- Test avg tokens: {test['avg_tokens']:.4f}",
        f"- Test parse success: {test['parse_success_rate']:.4f}",
        "",
        "## Iteration Snapshots",
        "",
    ]
    for h in history:
        lines.append(
            f"- Iteration {h['iteration']}: avg_tokens={h['after']['avg_tokens']:.4f}, "
            f"accuracy={h['after']['accuracy']:.4f}, accepted={len(h['accepted'])}, "
            f"language_size={h['language_size']}"
        )

    lines.extend(["", "## Accepted Macros", ""])
    macros = final_language["macros"]
    if macros:
        for name, pattern in sorted(macros.items()):
            lines.append(f"- {name}: {' '.join(pattern)}")
    else:
        lines.append("- None")

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
