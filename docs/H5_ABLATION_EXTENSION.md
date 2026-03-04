# H5/J Ablation Extension

This document defines the lightweight contract for H5 and J ablation extensions layered on top of the A-E matrix.

## Manifest Additions

Add an `h5_extensions` array to the ablation manifest.

Required run rows:

- `H5-PROV`
- `H5-OOD`
- `H5-DPTR`
- `J-1`
- `J-2`
- `J-3`
- `J-4`

Each row must include:

- `run_id`
- `name`
- `status`
- `return_code`
- `output`
- `metrics`
- `notes`

Example:

```json
{
  "h5_extensions": [
    {
      "run_id": "H5-PROV",
      "name": "H5 Provenance",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_prov.json",
      "metrics": {
        "final_acc": 0.39
      },
      "notes": "Lineage and grounding checks."
    },
    {
      "run_id": "H5-OOD",
      "name": "H5 Out-of-Distribution",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_ood.json",
      "metrics": {
        "final_acc": 0.31
      },
      "notes": "Distribution-shift stress test."
    },
    {
      "run_id": "H5-DPTR",
      "name": "H5 Distillation Pointer Transfer",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_dptr.json",
      "metrics": {
        "final_acc": 0.36
      },
      "notes": "Pointer transfer/distillation check."
    },
    {
      "run_id": "J-1",
      "name": "Graph Target (Factor Schema)",
      "status": "ok",
      "return_code": 0,
      "output": "runs/true_coconut_h_series/<timestamp>/j-1.json",
      "metrics": {
        "schema_valid_rate": 1.0,
        "graph_count": 20
      },
      "notes": "Converts prompts into Entities+Edges graphs and scores schema validity."
    },
    {
      "run_id": "J-2",
      "name": "Paraphrase Explosion (Invariance)",
      "status": "ok",
      "return_code": 0,
      "output": "runs/true_coconut_h_series/<timestamp>/j-2.json",
      "metrics": {
        "invariance_rate": 0.92,
        "variant_count": 20000
      },
      "notes": "Measures graph invariance under large paraphrase perturbations."
    },
    {
      "run_id": "J-3",
      "name": "Stop-Grad Isolation Gate",
      "status": "ok",
      "return_code": 0,
      "output": "runs/true_coconut_h_series/<timestamp>/j-3.json",
      "metrics": {
        "stopgrad_contract_pass": 1
      },
      "notes": "Asserts oracle isolation controls are structurally present."
    },
    {
      "run_id": "J-4",
      "name": "Operator Curriculum Build",
      "status": "ok",
      "return_code": 0,
      "output": "runs/true_coconut_h_series/<timestamp>/j-4.json",
      "metrics": {
        "sample_count": 1280,
        "operator_count": 5
      },
      "notes": "Builds operator-targeted micro-task curriculum dataset."
    }
  ]
}
```

## Markdown Summary Rows

When these rows are present in the manifest, the markdown summary must include:

- A row for `H5-PROV`
- A row for `H5-OOD`
- A row for `H5-DPTR`
- A row for `J-1`
- A row for `J-2`
- A row for `J-3`
- A row for `J-4`

## J-Series Targets

- `J-1` emits strict graph-schema artifacts from prompt text: `summary`, `metrics`, `graphs`, `relation_histogram`.
- `J-2` emits invariance artifacts from paraphrase mutation of `J-1`: `summary`, `metrics`, `samples`.
- `J-3` emits stop-grad gate artifacts: `summary`, `metrics`.
- `J-4` emits operator curriculum artifacts: `summary`, `metrics`, `operator_histogram` and a sidecar JSONL dataset.
