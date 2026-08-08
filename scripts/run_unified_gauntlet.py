import argparse
import json
import os
import random
import subprocess
import sys
import gc
from pathlib import Path
from typing import Any
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lojban_evolution.data.datasets.emergent_bridi import generate_m25_emergent_bridi_examples
from lojban_evolution.m20.dictionary import SyntheticPredicateExample, predicate_specs, _slot_targets, DOMAIN_TO_ID, POLARITY_TO_ID, RELATION_TO_ID, ROLE_TO_ID
from lojban_evolution.m21.bridi import DynamicBridiExample
from lojban_evolution.m23.relevance import M23RelevanceExample


def adapt_m20_data(size: int, seed: int = 0, floating_fraction: float = 0.15, surfaces: Any = None):
    m25_examples = generate_m25_emergent_bridi_examples(int(size * 1.5), seed=seed) # generate extra
    specs = predicate_specs()
    spec_map = {s.name: s for s in specs}
    
    # Map M25 18 classes to M20 14 classes
    label_map = {
        "transfer_success": "ownership_transfer",
        "transfer_refused": "ownership_refusal",
        "visibility_blocked": "visibility_occluded",
    }
    
    out = []
    for ex in m25_examples:
        mapped_label = label_map.get(ex.answer_label, ex.answer_label)
        if mapped_label not in spec_map:
            continue
            
        spec = spec_map[mapped_label]
        is_floating = random.Random(seed + len(out)).random() < float(floating_fraction)
        
        out.append(SyntheticPredicateExample(
            prompt=ex.prompt,
            predicate_id=spec.predicate_id,
            predicate_name=spec.name,
            domain_id=DOMAIN_TO_ID[spec.domain],
            polarity_id=POLARITY_TO_ID[spec.polarity],
            relation_type_id=RELATION_TO_ID[spec.relation_type],
            arity_id=int(spec.arity) - 1,
            role_schema_id=ROLE_TO_ID[spec.role_schema],
            has_argument=not is_floating,
            slot_targets=_slot_targets(spec.arity, not is_floating),
            surface=ex.surface,
            counterfactual_group=spec.name,
            entity_signature=ex.entity_signature,
        ))
        if len(out) >= size:
            break
    return out


def adapt_m21_data(size: int, seed: int = 0, floating_fraction: float = 0.15, surfaces: Any = None):
    m25_examples = generate_m25_emergent_bridi_examples(size, seed=seed)
    out = []
    for ex in m25_examples:
        out.append(DynamicBridiExample(
            prompt=ex.prompt,
            frames=ex.frames,
            entities=ex.entities,
            answer_id=ex.answer_id,
            answer_label=ex.answer_label,
            surface=ex.surface,
            counterfactual_group=ex.counterfactual_group,
            entity_signature=ex.entity_signature,
            is_floating=False,
            template_id="",
            template_family="",
            is_relation_ood=False,
        ))
    return out

def adapt_m21_adversarial_data(size: int, seed: int = 0, surfaces: Any = None):
    m25_examples = generate_m25_emergent_bridi_examples(size, seed=seed)
    out = []
    for ex in m25_examples:
        out.append(DynamicBridiExample(
            prompt=ex.prompt,
            frames=ex.frames,
            entities=ex.entities,
            answer_id=ex.answer_id,
            answer_label=ex.answer_label,
            surface=ex.surface,
            counterfactual_group=ex.counterfactual_group,
            entity_signature=ex.entity_signature,
            is_floating=False,
            template_id="",
            template_family="",
            is_relation_ood=True,
        ))
    return out


def adapt_m23_data(size: int, seed: int = 0, clean_fraction: float = 0.35, max_frames: int = 6):
    m25_examples = generate_m25_emergent_bridi_examples(size, seed=seed)
    out = []
    for ex in m25_examples:
        out.append(M23RelevanceExample(
            prompt=ex.prompt,
            frames=ex.frames,
            entities=ex.entities,
            answer_id=ex.answer_id,
            answer_label=ex.answer_label,
            surface=ex.surface,
            counterfactual_group=ex.counterfactual_group,
            entity_signature=ex.entity_signature,
            relevant_frame_indices=ex.relevant_frame_indices,
            decoy_frame_indices=ex.decoy_frame_indices,
            relevance_surface=ex.relevance_surface,
            is_floating=False,
            template_id="",
            template_family="",
        ))
    return out


def run_worker(model_id: str):
    print(f"Running worker for {model_id}...")
    output_path = Path("artifacts") / "gauntlet_results" / f"{model_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {}

    if model_id == "M20":
        import lojban_evolution.m20.dictionary as m20_module
        m20_module.generate_synthetic_world_examples = adapt_m20_data
        result = m20_module.train_m20_dictionary(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = {"common14_accuracy": result["metrics"].get("strict_accuracy", 0.0), "metrics": result["metrics"]}

    elif model_id == "M21":
        import lojban_evolution.m21.bridi as m21_module
        m21_module.generate_dynamic_bridi_examples = adapt_m21_data
        m21_module.generate_dynamic_bridi_adversarial_examples = adapt_m21_adversarial_data
        result = m21_module.train_m21_dynamic_bridi(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = {"full18_accuracy": result["metrics"].get("dynamic_strict_accuracy", 0.0), "metrics": result["metrics"]}

    elif model_id == "M23":
        import lojban_evolution.m23.relevance as m23_module
        m23_module.generate_m23_relevance_examples = adapt_m23_data
        result = m23_module.train_m23_relevance_router(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = {"full18_accuracy": result["metrics"].get("accuracy", 0.0), "metrics": result["metrics"]}

    elif model_id == "M26":
        from lojban_evolution.models.components.language_backbone import train_m26_end_to_end_loafman
        result = train_m26_end_to_end_loafman(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = {"full18_accuracy": result["metrics"].get("strict_accuracy", 0.0), "metrics": result["metrics"]}

    elif model_id == "M27":
        from lojban_evolution.m27.runtime import train_m27_coconut_bridi_runtime
        result = train_m27_coconut_bridi_runtime(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = {"full18_accuracy": result["metrics"].get("strict_accuracy", 0.0), "metrics": result["metrics"]}

    elif model_id == "M28":
        from lojban_evolution.m28.model import train_logebonic_symbiote_model
        result = train_logebonic_symbiote_model(
            train_size=1000, eval_size=200, epochs=1, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(result, dict):
            metrics = result.get("metrics", {})
        else:
            metrics = result.metrics if hasattr(result, "metrics") else getattr(result, "metrics", {})
        payload = {"full18_accuracy": metrics.get("strict_accuracy", 0.0), "metrics": metrics}

    elif model_id == "M29":
        from lojban_evolution.m29.runtime import evaluate_m29_star_runtime
        from lojban_evolution.m29.model import M29StarQFormerSymbiote
        from lojban_evolution.m21.bridi import build_vocab
        eval_examples = generate_m25_emergent_bridi_examples(200, seed=0)
        vocab = build_vocab(eval_examples)
        model = M29StarQFormerSymbiote(vocab_size=len(vocab), hidden_dim=32, num_queries=5, target_vocab_size=7)
        result = evaluate_m29_star_runtime(
            model=model, examples=eval_examples, vocab=vocab, batch_size=32, seed=0
        )
        payload = {"full18_accuracy": result["metrics"].get("strict_accuracy", 0.0), "metrics": result["metrics"]}

    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    # Memory cleanup to prevent OOM
    if "result" in locals():
        if isinstance(result, dict) and "model" in result:
            del result["model"]
        elif hasattr(result, "model"):
            del result.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Worker {model_id} completed successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, help="Run specific worker (model_id)")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker)
        return

    models = ["M20", "M21", "M23", "M26", "M27", "M28", "M29"]
    results = {}
    
    print("Starting Unified Gauntlet")
    for model_id in models:
        print(f"\\n--- Spawning worker for {model_id} ---")
        cmd = [sys.executable, __file__, "--worker", model_id]
        subprocess.run(cmd, check=True)
        
        # Load results
        output_path = Path("artifacts") / "gauntlet_results" / f"{model_id}.json"
        if output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            results[model_id] = {
                "common14_accuracy": data.get("common14_accuracy", 0.0),
                "full18_accuracy": data.get("full18_accuracy", 0.0)
            }
        else:
            print(f"Warning: No results found for {model_id}")
            results[model_id] = {"error": "No results"}

    print("\\n=== Unified Accuracy Table ===")
    print(f"{'Model':<10} | {'Common 14':<15} | {'Full 18':<15}")
    print("-" * 45)
    for model_id, acc in results.items():
        if "error" in acc:
            print(f"{model_id:<10} | ERROR           | ERROR")
        else:
            common = acc["common14_accuracy"]
            full = acc["full18_accuracy"]
            print(f"{model_id:<10} | {common:<15.4f} | {full:<15.4f}")

    table_path = Path("artifacts") / "gauntlet_results" / "unified_accuracy_table.json"
    table_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\\nSaved unified accuracy table to {table_path}")

if __name__ == "__main__":
    main()
